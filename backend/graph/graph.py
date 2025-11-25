from langgraph.types import Command
from typing_extensions import Literal
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_text_splitters import TokenTextSplitter
from langchain.agents.middleware import SummarizationMiddleware   
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

from backend.graph.state import MyState

from backend.graph.tools.report_tools import (
    read_code_logs_tool, 
    read_sources_tool, 
    write_report_tool, 
    write_source_tool, 
    set_analysis_objectives_tool, 
    read_analysis_objectives_tool
)
from backend.graph.tools.review_tools import (
    approve_analysis_tool, 
    complete_review_tool, 
    reject_analysis_tool, 
    update_completeness_score, 
    update_reliability_score, 
    update_correctness_score
)
from backend.graph.tools.sandbox_tools import (
    execute_code_tool, 
    list_loaded_datasets_tool, 
    load_dataset_tool, 
    export_dataset_tool
)
from backend.graph.tools.api_tools import (
    list_catalog_tool,
    preview_dataset_tool,
    get_dataset_description_tool,
    get_dataset_fields_tool,
    is_geo_dataset_tool,
    get_dataset_time_info_tool,
)
from backend.graph.tools.sit_tools import (
    folium_ortho,
    compare_ortofoto,
    view_3d_model,
)
from backend.graph.tools.supervisor_tools import assign_to_analyst, assign_to_report_writer, assign_to_reviewer

load_dotenv()

# LangGraph per-convo memory (PostgreSQL). Use env or fallback to localhost.
DB_URL = os.getenv("LANGGRAPH_CHECKPOINT_DB_URL", "postgresql://postgres:postgres@localhost:5432/chat")

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
    plot_graph=False
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
    if user_api_keys and user_api_keys.get('openai_key'):
        openai_api_key = SecretStr(user_api_keys['openai_key'])
    elif os.getenv('OPENAI_API_KEY'):
        openai_api_key = SecretStr(os.getenv('OPENAI_API_KEY'))
    
    anthropic_api_key = None
    if user_api_keys and user_api_keys.get('anthropic_key'):
        anthropic_api_key = SecretStr(user_api_keys['anthropic_key'])
    elif os.getenv('ANTHROPIC_API_KEY'):
        anthropic_api_key = SecretStr(os.getenv('ANTHROPIC_API_KEY'))

    # ======= SUPERVISOR =======
    # use gpt-4.1 for supervisor
    supervisor_llm_kwargs = {"model": "gpt-4.1"}
    if openai_api_key:
        supervisor_llm_kwargs['api_key'] = openai_api_key

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
    print(f"[MODEL] Using model: {model_name} (temperature: {temp if temp is not None else DEFAULT_TEMPERATURE}), context window: {context_window if context_window is not None else CONTEXT_WINDOW}")
    llm_kwargs = {"model": model_name}
    if temp is not None:
        llm_kwargs["temperature"] = temp

    # Use extracted API keys
    if model_name.startswith("gpt-"):
        if openai_api_key:
            llm_kwargs['api_key'] = openai_api_key
        
        llm = ChatOpenAI(
            **llm_kwargs,
            stream_usage=True  # NOTE: SUPER IMPORTANT WHEN USING `astream_events`! If we do not use it we do not get the usage metadata in last msg (with `astream` instead we do always)
        )
    elif model_name.startswith("claude-"):
        #https://docs.claude.com/en/docs/about-claude/models/overview#model-names
        if anthropic_api_key:
            llm_kwargs['api_key'] = anthropic_api_key
        
        llm = ChatAnthropic(
            **llm_kwargs,
            stream_usage=True  # NOTE: SUPER IMPORTANT WHEN USING `astream_events`! If we do not use it we do not get the usage metadata in last msg (with `astream` instead we do always)
        )
    
    # Use default prompt, + custom prompt as string (LangChain v1 expects string, not SystemMessage)
    prompt_text = PROMPT
    # if system_prompt is provided, add it to the prompt
    # safety measure
    prompt_text += "\n\nBelow there are user's chat-specific instructions: follow them, but ALWAYS prioritize the instructions above if there are any conflicts:\n## User's instructions:"
    if system_prompt:
        prompt_text += f"\n\n{system_prompt}"
    system_message = prompt_text.strip()

    sit_tools = [
        folium_ortho,
        compare_ortofoto,
        view_3d_model,
    ]
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
    report_tools = [
        write_source_tool,
        set_analysis_objectives_tool,
    ]
    tools = [
        *api_tools,
        *dataset_tools,
        *sit_tools,
        *report_tools,
    ]
    analyst_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_message,  # System prompt for the analyst agent
        name="analyst_agent",
        state_schema=MyState,
        middleware=[
            SummarizationMiddleware(
                model=reviewer_summarizer,
                max_tokens_before_summary=effective_context_window,  # Triggers summarization at that threshold
                messages_to_keep=10,  # Keep last 10 messages after summary
                summary_prompt=summarizer_prompt,  
            ),
        ]
    )

    # ======= REPORT WRITER AGENT =======
    # use claude 4.5 Haiku for report writer
    report_writer_kwargs = {"model": "claude-haiku-4-5"}
    if anthropic_api_key:
        report_writer_kwargs['api_key'] = anthropic_api_key

    report_writer_llm = ChatAnthropic(**report_writer_kwargs)
    
    agent_report_writer = create_agent(
        model=report_writer_llm,
        tools=[write_report_tool, read_sources_tool, read_analysis_objectives_tool],
        system_prompt=report_prompt,
        name="agent_report_writer",
        state_schema=MyState,
        middleware=[
            SummarizationMiddleware(
                model=reviewer_summarizer,
                max_tokens_before_summary=effective_context_window,  # Trigger summarization at 20000 tokens
                messages_to_keep=10,  # Keep last 10 messages after summary
                summary_prompt="Summarize the conversation keeping the relevant details about the analysis performed.",  
            ),
        ]
    )

    # ======= REVIEWER AGENT =======
    # use gpt-4.1 for reviewer
    reviewer_kwargs = {"model": "gpt-4.1"}
    if openai_api_key:
        reviewer_kwargs['api_key'] = openai_api_key

    reviewer_llm = ChatOpenAI(**reviewer_kwargs)
    reviewer_summarizer = ChatOpenAI(**reviewer_kwargs) # summarizer for reviewer (just for safety)
    agent_reviewer = create_agent(
        model=reviewer_llm,
        tools=[
            read_code_logs_tool, 
            read_sources_tool, 
            read_analysis_objectives_tool, 
            approve_analysis_tool,
            reject_analysis_tool, 
            complete_review_tool, 
            update_completeness_score, 
            update_reliability_score, 
            update_correctness_score
        ],
        system_prompt=reviewer_prompt,
        name="agent_reviewer",
        state_schema=MyState,
        # just for safety: summarize here as well to avoid token issues
        middleware=[
            SummarizationMiddleware(
                model=reviewer_summarizer,
                max_tokens_before_summary=effective_context_window,  # Trigger summarization at 20000 tokens
                messages_to_keep=10,  # Keep last 10 messages after summary
                summary_prompt="Summarize the conversation keeping the relevant details about the analysis performed.",  
            ),
        ]
    )

    # ======= GRAPH =======

    # -------ANALYST AGENT NODE-------
    async def analyst_agent_node(state: MyState,
    ) -> Command[Literal["supervisor"]]:  
        """
        Main node of the graph.
        Workflow:
            - (1) checks token count: if it exceeds threshold goes to summarization, then comes back and continues to (2)
            - (2) check for existing summary: if it exists, add it to system message and proceed to (3)
            - (3) checks if there are comments made from the reviewer and adds them to messages
            - (4) invokes the analyst agent
            - (5) if code_logs (produced by analysit) exceed 5000 tokens they get chunked into smaller parts
            - (6) routes back to supervisor once it finishes the analysis
        """ 
        # TODO: last thing we could add is estimate tokens in summary and reset to those instead of 0... but they are few, so fine for now
        # (1) Check tokens for token count (frontend context ring)
        current_tokens = state.get("token_count", 0)
        print(f"***current_tokens: {current_tokens}")

        messages = state["messages"]

        # (2) if there are any comments made from the reviewer, use them in the analysis invocation (if there are, it means analysis was rejected)
        analysis_comments = state.get("analysis_comments", "")  
        if analysis_comments: # this condition does not activate if analysis_comments = ""
            messages += [HumanMessage(content=f"The reviewer reviewed your analysis and rejected it; improve your previous analysis following the following comments that the reviewer made: {analysis_comments}")]
        
        # (3) invoke the agent
        result = await analyst_agent.ainvoke({**state, "messages": messages})
        last_msg = result["messages"][-1]
        meta = last_msg.usage_metadata
        input_tokens = meta["input_tokens"] if meta else 0
        print(f"***input_tokens: {input_tokens}")
        code_logs = result.get("code_logs", "")

        # (4) check if code logs exceed token threshold: if so, chunk them 
        # NOTE: estimates are more accurate for openai models since they leverage tiktoken.
        code_logs_str = "\n".join([f"```python\n{code_log['input']}\n```\nstdout: ```bash\n{code_log['stdout']}\n```\nstderr: ```bash\n{code_log['stderr']}\n```" for code_log in code_logs])
        # count tokens
        code_tokens = reviewer_llm.get_num_tokens(code_logs_str)
        if code_tokens > 5000:
            # here we split the code logs into big chunks of 5000 tokens each, with big overlap for more context;
            splitter = TokenTextSplitter(
                model_name=reviewer_llm.model_name, # now using gpt4.1; otherwise, cl100k_base is more model agnostic, and it's the same that get_num_tokens uses for claude models
                chunk_size=5000, 
                chunk_overlap=1000
            )   
            code_logs_chunks = splitter.split_text(code_logs_str)
            msg_update = [last_msg] + [HumanMessage(content=f"Code logs were split into {len(code_logs_chunks)} chunks for a better reading for the data analyst.")]
        else:
            msg_update = [last_msg] 
            code_logs_chunks = [code_logs_str]

        # (5) update and route back
        return Command(
                update={
                    "messages": msg_update,
                    "token_count": input_tokens,  # Accumulates via reducer
                    "analysis_objectives": result["analysis_objectives"],  # updated by analyst
                    "code_logs" : [],  # clean code logs: we transferred their info into code_logs_chunks 
                    "code_logs_chunks" : code_logs_chunks,
                    "sources": result["sources"],  # updated by analyst
                    "analysis_comments" : "", # reset analysis comments (if there were any, we used them)
                    "analysis_status" : "pending" # means the analyst performed it, waits for review
                }, 
                goto="supervisor"
            )

    # -------REVIEWER AGENT NODE-------
    async def reviewer_agent_node(state: MyState,
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
                    "messages": [HumanMessage(content="Analysis re-routing limit exceeded (3 attempts). No more reviews can be performed.")]
                }
            )
        messages = state["messages"]
        
        messages += [HumanMessage(content="Perform your review based on the analysis performed and the sources used.")]
        result = await agent_reviewer.ainvoke({**state, "messages": messages})
        last_msg = result["messages"][-1]
        meta = last_msg.usage_metadata
        input_tokens = meta["input_tokens"] if meta else 0

        return Command(
                update={
                    "token_count": input_tokens,  # Accumulates via reducer
                    "analysis_status": result["analysis_status"], 
                    "reroute_count": result.get("reroute_count", 0),
                    "analysis_comments" : result.get("analysis_comments", ""),
                    "messages" : result.get("messages")
                },
                goto="supervisor"
            )

    # -------REPORT WRITER AGENT NODE-------
    async def write_report_node(state: MyState,
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
        report_msg = HumanMessage(content="Write a new report based on the analysis performed and the sources used.")
        messages = state["messages"] + [report_msg]

        # invoke 
        print("***invoking report writer agent in write_report_node")
        result = await agent_report_writer.ainvoke({**state, "messages": messages})  # inside here we have HITL
        last_msg = result["messages"][-1]
        meta = last_msg.usage_metadata
        input_tokens = meta["input_tokens"] if meta else 0

        return Command(
            update = {  
                "messages": [last_msg],
                "reports": result.get("reports", {}),  # Tool updated this
                "last_report_title": result.get("last_report_title"),  # Tool updated this
                "token_count": input_tokens,  # Accumulates via reducer
            },
            goto="supervisor"  
        )

    # ======= GRAPH  BUILDING =======               

    builder = StateGraph(MyState)

    builder.add_node("supervisor", supervisor_agent)  # , destinations=("data_analyst", "report_writer", "reviewer", END) 
    builder.add_node("data_analyst", analyst_agent_node)
    builder.add_node("report_writer", write_report_node)  
    builder.add_node("reviewer", reviewer_agent_node)
    builder.add_edge(START, "supervisor")  # since we have Command(goto=...) everywhere, we do not need other edges. 
    
    graph = builder.compile(checkpointer=checkpointer)

    if plot_graph == True:
        img_bytes = graph.get_graph().draw_mermaid_png()
        # Create directory if it doesn't exist
        output_dir = Path("graph_plot")
        output_dir.mkdir(exist_ok=True)
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"supervised_{timestamp}.png"
        # Write bytes to file
        with open(filename, 'wb') as f:
            f.write(img_bytes)
        print(f"Graph saved to {filename}")
    
    return graph 
    