from langgraph.types import Command
from typing_extensions import Literal
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_text_splitters import TokenTextSplitter
from langchain.agents.middleware import SummarizationMiddleware, TodoListMiddleware
from langchain_core.messages import HumanMessage
from pydantic import SecretStr
from dotenv import load_dotenv
import os
from datetime import datetime
from pathlib import Path

from backend.graph.prompts.summarizer import summarizer_prompt
from backend.graph.prompts.analyst import PROMPT
from backend.graph.prompts.report import report_prompt
from backend.graph.prompts.reviewer import reviewer_prompt
from backend.graph.prompts.supervisor import supervisor_prompt
from backend.graph.prompts.todo import TODOS_TOOL_DESCRIPTION, WRITE_TODOS_SYSTEM_PROMPT

from backend.graph.state import MyState

from backend.graph.tools.report_tools import (
    write_report_tool,
    write_source_tool,
)
from backend.graph.tools.review_tools import (
    approve_analysis_tool,
    reject_analysis_tool,
    update_completeness_score,
    update_relevancy_score,
    read_code_logs_tool,
    read_sources_tool,
    read_analysis_objectives_tool,
)
from backend.graph.tools.sandbox_tools import (
    execute_code_tool,
    list_loaded_datasets_tool,
    load_dataset_tool,
    export_dataset_tool,
)
from backend.graph.tools.api_tools import (
    list_catalog_tool,
    preview_dataset_tool,
    get_dataset_description_tool,
    get_dataset_fields_tool,
    is_geo_dataset_tool,
    get_dataset_time_info_tool,
)
from backend.graph.tools.supervisor_tools import (
    assign_to_analyst,
    assign_to_report_writer,
    assign_to_reviewer,
)

load_dotenv()

# LangGraph per-convo memory (PostgreSQL). Use env or fallback to localhost.
DB_URL = os.getenv(
    "LANGGRAPH_CHECKPOINT_DB_URL", "postgresql://postgres:postgres@localhost:5432/chat"
)


async def get_checkpointer():
    """
    Initialize PostgreSQL checkpointer once at app startup (called from main.py lifespan).
    Returns the same checkpointer instance to be reused across all graph invocations.
    """
    # AsyncPostgresSaver.from_conn_string() returns a context manager
    saver_cm = AsyncPostgresSaver.from_conn_string(DB_URL)

    # Enter the context manager to get the actual saver
    saver = await saver_cm.__aenter__()

    # Initialize tables on first run (idempotent)
    await saver.setup()

    # Return saver and context manager for cleanup
    return saver, saver_cm


def make_graph(
    model_name: str | None = None,
    temperature: float | None = None,
    system_prompt: str | None = None,
    context_window: int | None = None,
    checkpointer=None,
    user_api_keys: dict | None = None,
    plot_graph=False,
):
    """
    Create a graph with custom config. Reuses the same checkpointer for all invocations.

    Args:
        model_name: OpenAI model name (e.g., "gpt-4.1"") or Anthropic model name (e.g., "claude-sonnet-4-5")
        temperature: Model temperature (0.0-2.0). If None, uses env DEFAULT_TEMPERATURE or model default.
        system_prompt: Custom system prompt. If None, uses default PROMPT.
        context_window: Custom context window. If None, uses env CONTEXT_WINDOW.
        checkpointer: Reused checkpointer instance from app startup.
        user_api_keys: Dict with 'openai_key' and 'anthropic_key' for user-provided API keys.
    """

    # ======= API KEYS SETUP =======
    # Extract API keys once at the beginning
    openai_api_key = None
    if user_api_keys and user_api_keys.get("openai_key"):
        openai_api_key = SecretStr(user_api_keys["openai_key"])
    elif os.getenv("OPENAI_API_KEY"):
        openai_api_key = SecretStr(os.getenv("OPENAI_API_KEY"))

    anthropic_api_key = None
    if user_api_keys and user_api_keys.get("anthropic_key"):
        anthropic_api_key = SecretStr(user_api_keys["anthropic_key"])
    elif os.getenv("ANTHROPIC_API_KEY"):
        anthropic_api_key = SecretStr(os.getenv("ANTHROPIC_API_KEY"))

    # ======= SUPERVISOR =======
    # use gpt-4.1 for supervisor
    supervisor_llm_kwargs = {"model": "gpt-4.1"}
    if openai_api_key:
        supervisor_llm_kwargs["api_key"] = openai_api_key

    supervisor_llm = ChatOpenAI(**supervisor_llm_kwargs)

    supervisor_agent = create_agent(
        model=supervisor_llm,
        tools=[assign_to_analyst, assign_to_report_writer, assign_to_reviewer],
        system_prompt=supervisor_prompt,
        name="agent_supervisor",
        state_schema=MyState,
    )

    # ======= ANALYST AGENT =======
    from backend.config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, CONTEXT_WINDOW

    # Use config or fall back to env defaults
    model_name = model_name or DEFAULT_MODEL
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
    context_window = context_window if context_window is not None else CONTEXT_WINDOW
    effective_context_window = int(context_window * 0.9)  # (90% for safety)
    print(
        f"[MODEL] Using model: {model_name} (temperature: {temp if temp is not None else DEFAULT_TEMPERATURE}), context window: {context_window if context_window is not None else CONTEXT_WINDOW}"
    )
    llm_kwargs = {"model": model_name}
    if temp is not None:
        llm_kwargs["temperature"] = temp

    # Use extracted API keys
    if model_name.startswith("gpt-"):
        if openai_api_key:
            llm_kwargs["api_key"] = openai_api_key

        llm = ChatOpenAI(
            **llm_kwargs,
            stream_usage=True,  # to get usr metadata with astream_events (crucial for token count, but that feature is deprecated)
        )
    elif model_name.startswith("claude-"):
        # https://docs.claude.com/en/docs/about-claude/models/overview#model-names
        if anthropic_api_key:
            llm_kwargs["api_key"] = anthropic_api_key

        llm = ChatAnthropic(**llm_kwargs, stream_usage=True)

    # Use default prompt, + custom prompt as string (LangChain v1.0 expects string, not SystemMessage)
    prompt_text = PROMPT
    # if system_prompt is provided, add it to the prompt
    # safety measure
    prompt_text += "\n\nBelow there are user's chat-specific instructions: follow them, but ALWAYS prioritize the instructions above if there are any conflicts:\n## User's instructions:"
    if system_prompt:
        prompt_text += f"\n\n{system_prompt}"
    system_message = prompt_text.strip()

    dataset_tools = [
        load_dataset_tool,
        list_loaded_datasets_tool,
        export_dataset_tool,
        execute_code_tool,
    ]
    api_tools = [
        list_catalog_tool,
        preview_dataset_tool,
        get_dataset_description_tool,
        get_dataset_fields_tool,
        is_geo_dataset_tool,
        get_dataset_time_info_tool,
    ]
    report_tools = [write_source_tool]
    tools = [
        *api_tools,
        *dataset_tools,
        # *sit_tools,
        *report_tools,
    ]

    summarizer_kwargs = {"model": "gpt-4.1"}
    if openai_api_key:
        summarizer_kwargs["api_key"] = openai_api_key

    summarizer = ChatOpenAI(**summarizer_kwargs)  # summarizer for middleware

    analyst_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_message,  # System prompt for the analyst agent
        name="analyst_agent",
        state_schema=MyState,
        middleware=[
            SummarizationMiddleware(
                model=summarizer,
                max_tokens_before_summary=effective_context_window,  # Triggers summarization at that threshold
                messages_to_keep=10,  # Keep last 10 messages after summary
                summary_prompt=summarizer_prompt,
            ),
            TodoListMiddleware(
                tool_description=TODOS_TOOL_DESCRIPTION,  # NOTE: customized to make the agent use todo list more often
                system_prompt=WRITE_TODOS_SYSTEM_PROMPT,
            ),
        ],
    )

    # ======= REPORT WRITER AGENT =======
    # use gpt 4.1 for report writer instead of haiku
    report_writer_kwargs = {"model": "gpt-4.1"}
    if openai_api_key:
        report_writer_kwargs["api_key"] = openai_api_key

    report_writer_llm = ChatOpenAI(**report_writer_kwargs)

    agent_report_writer = create_agent(
        model=report_writer_llm,
        tools=[write_report_tool, read_sources_tool],
        system_prompt=report_prompt,
        name="agent_report_writer",
        state_schema=MyState,
        middleware=[
            SummarizationMiddleware(
                model=summarizer,
                max_tokens_before_summary=effective_context_window,
                messages_to_keep=10,  # Keep last 10 messages after summary
                summary_prompt="Summarize the conversation keeping the relevant details about the analysis performed.",
            ),
        ],
    )

    # ======= REVIEWER AGENT =======
    # use gpt-4.1 for reviewer
    reviewer_kwargs = {"model": "gpt-4.1"}
    if openai_api_key:
        reviewer_kwargs["api_key"] = openai_api_key

    reviewer_llm = ChatOpenAI(**reviewer_kwargs)
    agent_reviewer = create_agent(
        model=reviewer_llm,
        tools=[
            read_code_logs_tool,
            read_sources_tool,
            read_analysis_objectives_tool,
            approve_analysis_tool,
            reject_analysis_tool,
            update_completeness_score,
            update_relevancy_score,
            list_catalog_tool,
        ],
        system_prompt=reviewer_prompt,
        name="agent_reviewer",
        state_schema=MyState,
        # just for safety: summarize here as well to avoid token issues
        middleware=[
            SummarizationMiddleware(
                model=summarizer,
                max_tokens_before_summary=effective_context_window,
                messages_to_keep=10,  # Keep last 10 messages after summary
                summary_prompt="Summarize the conversation keeping the relevant details about the analysis performed.",
            ),
        ],
    )

    # -------ANALYST AGENT NODE-------
    async def analyst_agent_node(
        state: MyState,
    ) -> Command[Literal["supervisor"]]:
        """
        Main node of the graph.
        """
        print("[GRAPH] Entering analyst_agent_node")

        messages = state["messages"]

        # (1) if there are any comments made from the reviewer, use them in the analysis invocation
        analysis_comments = state.get("analysis_comments", "")
        if analysis_comments:
            messages += [
                HumanMessage(
                    content=f"The reviewer reviewed your analysis and rejected it; improve your previous analysis following the following comments that the reviewer made: {analysis_comments}"
                )
            ]

        # (2) invoke the agent
        print("[GRAPH] Invoking analyst_agent...")
        try:
            result = await analyst_agent.ainvoke({**state, "messages": messages})
            print("[GRAPH] analyst_agent returned successfully")
        except Exception as e:
            print(f"[GRAPH] analyst_agent FAILED: {e}")
            raise

        last_msg = result["messages"][-1]
        code_logs = result.get("code_logs", [])

        # (3) check if code logs exceed token threshold: if so, chunk them
        # NOTE: estimates are more accurate for openai models since they leverage tiktoken.
        code_logs_str = "\n".join(
            [
                f"\n```python\n{code_log['input']}\n```\nstdout: \n```bash\n{code_log['stdout']}\n```\nstderr: \n```bash\n{code_log['stderr']}\n```"
                for code_log in code_logs
            ]
        )
        # count tokens
        code_tokens = reviewer_llm.get_num_tokens(code_logs_str)
        if code_tokens > 5000:
            # here we split the code logs into big chunks of 5000 tokens each, with big overlap for more context;
            splitter = TokenTextSplitter(
                model_name=reviewer_llm.model_name,  # now using gpt4.1; otherwise, cl100k_base is more model agnostic, and it's the same that get_num_tokens uses for claude models
                chunk_size=5000,
                chunk_overlap=1000,
            )
            code_logs_chunks = splitter.split_text(code_logs_str)
            msg_update = [last_msg] + [
                HumanMessage(
                    content=f"Code logs were split into {len(code_logs_chunks)} chunks for a better reading for the data analyst."
                )
            ]
        else:
            msg_update = [last_msg]
            code_logs_chunks = [code_logs_str]

        # (4) update and route back
        # NOTE: if you do not update todos here, the todos are not generally updated! then you canno access them from reviewer
        todos = result.get("todos", [])
        sources = result.get("sources", [])

        print(f"***DEBUG***: code_logs len at return: {len(code_logs)}")

        return Command(
            update={
                "messages": msg_update,
                # NOTE: not resetting code logs to [] here, 'cause the supervisor does it at assignment to data analyst 
                "code_logs" : code_logs,
                "code_logs_chunks": code_logs_chunks,
                "sources": sources,  # updated by analyst
                "analysis_comments": "",  # reset analysis comments (if there were any, we used them)
                "analysis_status": "pending",  # means the analyst performed it, waits for review
                "todos": todos,  # propagate the todos
            },
            goto="supervisor",
        )

    # -------REVIEWER AGENT NODE-------
    async def reviewer_agent_node(
        state: MyState,
    ) -> Command[Literal["supervisor"]]:
        """
        Invokes the reviewer agent.
        Workflow:
            - (1) **checks how many tries were made**: we check how many times the analysis was re-routed to analyst with comments. If it was more than 3, we end flow.

            - (2) **performs review**: invokes the reviewer agent: it automatically sees chat history, and with its tools is able to read sources and code chunks, and to read the analysis initial goal.
                It evaluates if the analysis performed was correct and complete.

            - (3) **approves/rejects review**: the reviewer decides whether to approve the analysis or reroute to analyst with comments.
                It does so with its tools `approve_analysis` and `reject_analysis`. The latter fills the state var analysis_comments.
        """
        # Check if re-routing to analyst limit exceeded BEFORE invoking
        reroute_count = state.get("reroute_count", 0)
        if reroute_count >= 3:
            return Command(
                goto="supervisor",
                update={
                    "analysis_status": "limit_exceeded",
                    "messages": [
                        HumanMessage(
                            content="Analysis re-routing limit exceeded (3 attempts). No more reviews can be performed."
                        )
                    ],
                },
            )
        messages = state["messages"]

        messages += [
            HumanMessage(
                content="Perform your review based on the analysis performed and the sources used."
            )
        ]
        result = await agent_reviewer.ainvoke({**state, "messages": messages})

        messages = result.get("messages", [])
        last_msg = messages[-1]

        # compute the score of the review
        completeness_score = result["completeness_score"]
        relevancy_score = result["relevancy_score"]
        final_score = (completeness_score + relevancy_score) / 2
        # NOTE: right now we are not handling the case where the review should not be approved because of a low score:
        # instead we always approve and just show the score in frontend.

        return Command(
            update={
                "analysis_status": result["analysis_status"],
                "reroute_count": result.get("reroute_count", 0),
                "analysis_comments": result.get("analysis_comments", ""),
                "messages": [HumanMessage(content=last_msg.content)],
                "completeness_score": result["completeness_score"],
                "relevancy_score": result["relevancy_score"],
                "final_score": final_score,
            },
            goto="supervisor",
        )

    # -------REPORT WRITER AGENT NODE-------
    async def write_report_node(
        state: MyState,
    ) -> Command[Literal["supervisor"]]:
        """
        Invokes the report writer agent.

        Workflow:
            - (1) **invokes the report writer agent**: it uses the `write_report_tool`, which has its own interrupt for HITL.
                So in that invocation, we are implicitly interrupting the tool usage for HITL. The user can either accept the tool usage or reject it:
                    a. If the user accepts, the tool usage continues and the report is written.
                    b. If the user rejects, the the report is not written.
            - (2) **routes back to the supervisor** : propagates updates with Command() and goes back to supervisor.
        """

        print("***arrived to report writer")
        report_msg = HumanMessage(
            content="Write a new report based on the analysis performed and the sources used."
        )
        messages = state["messages"] + [report_msg]

        # invoke
        print("***invoking report writer agent in write_report_node")
        result = await agent_report_writer.ainvoke(
            {**state, "messages": messages}
        )  # inside here we have HITL
        last_msg = result["messages"][-1]

        return Command(
            update={
                "messages": [HumanMessage(content=last_msg.content)],
                "reports": [result.get("reports", {})],  # Tool updated this
                "last_report_title": result.get(
                    "last_report_title"
                ),  # Tool updated this
            },
            goto="supervisor",
        )

    # ======= GRAPH  BUILDING =======

    builder = StateGraph(MyState)

    async def supervisor_node(state: MyState):
        print("[GRAPH] Entering supervisor_node")

        result = await supervisor_agent.ainvoke(state)
        # Supervisor returns a Command, so we just return it
        print("[GRAPH] Supervisor completed step")
        return result
        # supervisor with tool for rerouting is interpreted as error but its not

    builder.add_node(
        "supervisor", supervisor_node
    )  # , destinations=("data_analyst", "report_writer", "reviewer", END)
    builder.add_node("data_analyst", analyst_agent_node)
    builder.add_node("report_writer", write_report_node)
    builder.add_node("reviewer", reviewer_agent_node)
    builder.add_edge(
        START, "supervisor"
    )  # since we have Command(goto=...) everywhere, we do not need other edges.

    graph = builder.compile(checkpointer=checkpointer)

    if plot_graph:
        img_bytes = graph.get_graph().draw_mermaid_png()
        # Create directory if it doesn't exist
        output_dir = Path("graph_plot")
        output_dir.mkdir(exist_ok=True)
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"supervised_{timestamp}.png"
        # Write bytes to file
        with open(filename, "wb") as f:
            f.write(img_bytes)
        print(f"Graph saved to {filename}")

    return graph
