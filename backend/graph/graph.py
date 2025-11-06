from langgraph.types import Command, interrupt
from typing_extensions import Literal
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_text_splitters import TokenTextSplitter
from pydantic import SecretStr
from dotenv import load_dotenv
import os

from backend.graph.prompts.summarizer import summarizer_prompt
from backend.graph.prompts.analyst import PROMPT
from backend.graph.tools.report_tools import assign_to_report_writer_tool, read_code_logs_tool, read_sources_tool, write_report_tool, write_source_tool
from backend.graph.prompts.report import report_prompt
from backend.graph.prompts.reviewer import reviewer_prompt
from backend.graph.tools.sandbox_tools import execute_code_tool, list_loaded_datasets_tool, load_dataset_tool, export_dataset_tool
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
from backend.graph.state import MyState


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


def make_graph(model_name: str | None = None, temperature: float | None = None, system_prompt: str | None = None, context_window: int | None = None, checkpointer=None, user_api_keys: dict | None = None):
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

    # ======= ANALYST AGENT =======
    from backend.config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, CONTEXT_WINDOW
    # Use config or fall back to env defaults
    model_name = model_name or DEFAULT_MODEL
    llm_kwargs = {"model": model_name}
    
    # Only pass temperature if explicitly set (config) or if env default exists
    temp = temperature if temperature is not None else DEFAULT_TEMPERATURE
    if temp is not None:
        llm_kwargs["temperature"] = temp

    if model_name == "gpt-5":
        llm_kwargs["temperature"] = 1.0  # gpt-5 only accepts 1

    # Use user API keys if available, otherwise fall back to environment variables
    if model_name.startswith("gpt-"):
        api_key = None
        if user_api_keys and user_api_keys.get('openai_key'):
            api_key = user_api_keys['openai_key']
        elif os.getenv('OPENAI_API_KEY'):
            api_key = os.getenv('OPENAI_API_KEY')
        
        if api_key:
            llm_kwargs['api_key'] = SecretStr(api_key)
        
        llm = ChatOpenAI(
            **llm_kwargs,
            stream_usage=True  # NOTE: SUPER IMPORTANT WHEN USING `astream_events`! If we do not use it we do not get the usage metadata in last msg (with `astream` instead we do always)
        )
    elif model_name.startswith("claude-"):
        #https://docs.claude.com/en/docs/about-claude/models/overview#model-names
        api_key = None
        if user_api_keys and user_api_keys.get('anthropic_key'):
            api_key = user_api_keys['anthropic_key']
        elif os.getenv('ANTHROPIC_API_KEY'):
            api_key = os.getenv('ANTHROPIC_API_KEY')
        
        if api_key:
            llm_kwargs['api_key'] = SecretStr(api_key)
        
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
        assign_to_report_writer_tool,
        write_source_tool,
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
    )

    # ======= SUMMARIZER AGENT =======
    # Use same API key configuration as main LLM for gpt-4.1
    summarizer_kwargs = {"model": "gpt-4.1", "temperature": 0.0}
    if user_api_keys and user_api_keys.get('openai_key'):
        summarizer_kwargs['api_key'] = SecretStr(user_api_keys['openai_key'])
    elif os.getenv('OPENAI_API_KEY'):
        summarizer_kwargs['api_key'] = SecretStr(os.getenv('OPENAI_API_KEY'))
    
    summarizer_llm = ChatOpenAI(**summarizer_kwargs)
    agent_summarizer = create_agent(
        model=summarizer_llm,
        tools=[],
        system_prompt=summarizer_prompt,  
        name="agent_summarizer",
        state_schema=MyState,
    )

    # ======= REPORT WRITER AGENT =======
    # use claude 4.5 Haiku for report writer
    report_writer_kwargs = {"model": "claude-haiku-4-5"}
    if user_api_keys and user_api_keys.get('anthropic_key'):
        report_writer_kwargs['api_key'] = SecretStr(user_api_keys['anthropic_key'])
    elif os.getenv('ANTHROPIC_API_KEY'):
        report_writer_kwargs['api_key'] = SecretStr(os.getenv('ANTHROPIC_API_KEY'))

    report_writer_llm = ChatAnthropic(**report_writer_kwargs)
    
    agent_report_writer = create_agent(
        model=report_writer_llm,
        tools=[write_report_tool], #, get_sources_tool],
        system_prompt=report_prompt,
        name="agent_report_writer",
        state_schema=MyState,
    )

    # ======= REVIEWER AGENT =======
    # use claude 4.5 sonnet for reviewer
    reviewer_kwargs = {"model": "claude-sonnet-4-5"}
    if user_api_keys and user_api_keys.get('anthropic_key'):
        reviewer_kwargs['api_key'] = SecretStr(user_api_keys['anthropic_key'])
    elif os.getenv('ANTHROPIC_API_KEY'):
        reviewer_kwargs['api_key'] = SecretStr(os.getenv('ANTHROPIC_API_KEY'))
    
    reviewer_llm = ChatAnthropic(**reviewer_kwargs)
    agent_reviewer = create_agent(
        model=reviewer_llm,
        tools=[read_code_logs_tool, read_sources_tool],
        system_prompt=reviewer_prompt,
        name="agent_reviewer",
        state_schema=MyState,
    )

    # ======= GRAPH =======

    # -------ROUTING FUNCTIONS-------
    def review_routing(state: MyState):
        """
        Used in reviewer_agent_node to route to next node based on the `analysis_status` flag.
        It decides whether to: 
            a. re-route back to analyst agent 
            b. continue flow
            c. end flow
        
        Specifically: 
            (a) **If the flag is rejected** --> **re-routes back to analyst agent.** (analysis needs revision)
            (b) **If the flag is approved** --> **continues flow.** (analysis is correct and complete)
            (c) **If the flag is limit_exceeded** --> **ends flow.** (too many re-routings to analyst)
        """
        analysis_status = state.get("analysis_status", "none")  # defaults to none it it wasn't initialized yet (correct, none is default value)
        print(f"***routing function in review_routing: analysis status is {analysis_status}")
        if analysis_status == "rejected":
            return "analyst_agent"
        elif analysis_status == "approved":
            return "continue_flow"
        elif analysis_status == "limit_exceeded":
            return "__end__"

    def write_report_or_end_flow(state: MyState):
        """
        Used in analyst_agent_node to route to next node based on the report status flag.
        It decides whether to go to report writer or end flow:
        **If it is none -> ends flow.** (initial state)
        **If it is assigned -> goes to report writer.**
        
        This may seem like overkill at first, but actually it allows us not to interrupt the flow at every data analyst answer.
        Only if the data analyst thinks it should write a report, it sets the flag to assigned and goes to report writer.
        
        NOTE: this workaround was needed because nesting commands is bad behaviour - so we make a tool update a flag and then check it here.
        Basically an alternative to a conditional edge. 
        """
        report_status = state.get("report_status", "none")  # defaults to none it it wasn't initialized yet (correct, none is default value)
        print(f"***routing function in get_next_node: report status is {report_status}")
        if report_status == "assigned":  # means the data analyst thinks it should write a report, called assign_to_report_writer_tool
            print("***routing to report writer in get_next_node")
            return "report_writer"
        elif report_status == "none":  # initial state, no calls to assign_to_report_writer_tool yet -> ends flow
            print("***routing to end flow in get_next_node")
            return "__end__"
        else:
            raise ValueError(f"Invalid report status: {report_status}")
    
    def edit_report_or_end_flow(report_status: Literal["pending", "rejected"]):
        """
        Used in write_report_node to route to next node based on the report status. The report status is updated by the write_report_tool.
        **If it's pending -> goes to human approval.** (write_report_tool interuupted and user confirmed -> pending)
        **If it's rejected -> ends flow.** (write_report_tool interrupted and user rejected -> rejected)
        """
        if report_status == "pending":
            return "human_approval"
        elif report_status == "rejected":
            return "__end__"

    # -------SUMMARIZATION NODE-------
    async def summarize_conversation(state: MyState,
    ) -> Command[Literal["analyst_agent"]]:  # after summary we go back to the analyst agent
        """
        Summarizes the conversation with the agent_summarizer
        (!) NOTE: the summary does not persist in chat history, it's only added as system message dynamically, at invokation, when needed.
        It only persists in state.
        """
        # First, we get any existing summary
        summary = state.get("summary", "")
        # Create our summarization prompt 
        if summary:
            # A summary already exists
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
        else:
            summary_message = "Create a summary of the conversation above:"

        # Add prompt to our history
        messages = state["messages"] + [HumanMessage(content=summary_message)]
        response = await agent_summarizer.ainvoke({"messages": messages})

        summary = response["messages"][-1].content

        # Delete all but the 4 most recent messages
        delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-4]]  
        
        return Command(
                update={
                    "summary": summary, 
                    "messages": delete_messages,
                    "token_count": -1  # reset token count to zero
                    }, 
                goto="analyst_agent"  # go back to the analyst agent to answer the question
            )   

    # -------ANALYST AGENT NODE-------
    async def analyst_agent_node(state: MyState,
    ) -> Command[Literal["summarize_conversation", "report_writer", "__end__"]]:  # if summary is needed go to summarize_conversation, otherwise either continues flow to report writer or ends flow
        """
        Main node of the graph.
        Workflow:
            - (1) checks token count: if it exceeds threshold goes to summarization, then comes back and continues to (2)
            - (2) check for existing summary: if it exists, add it to system message and proceed to (3)
            - (3) invokes the analyst agent
            - (4) routes to next node, which is code_chunking_node (checks if code is too long, then goes to reviewer agent)
        """ 
        # TODO: last thing we could add is estimate tokens in summary and reset to those instead of 0... but they are few, so fine for now
        # Check tokens BEFORE invoking analyst agent (Cursor-style: summarize first, then answer)
        current_tokens = state.get("token_count", 0)
        # Use thread-specific context_window or fall back to env default
        effective_context_window = context_window if context_window is not None else CONTEXT_WINDOW
        threshold = effective_context_window * 0.9
        if current_tokens >= threshold:
            # Route to summarization FIRST, then back to analyst agent
            return Command(
                goto="summarize_conversation"
            )
        # If we're here, tokens are fine (either we summarized or we were under the threshold); proceed with analyst agent
        # get the summary
        summary = state.get("summary", "")

        # if the summary is not empty add it 
        if summary:
            # Add summary to system message **just for the invocation** - it will not be persisted in messages history, only persists in state
            system_message = f"Summary of conversation earlier: {summary}"
            # Let's just add the summary as a human message at the beginning
            messages_with_summary = [HumanMessage(content=system_message)] + state["messages"]
            result = await analyst_agent.ainvoke({"messages": messages_with_summary})
        else:
            messages = state["messages"]

        # invoke the agent
        result = await analyst_agent.ainvoke({"messages": messages})
        last_msg = result["messages"][-1]
        meta = last_msg.usage_metadata
        input_tokens = meta["input_tokens"] if meta else 0

        # update the token count and add message
        return Command(
                update={
                    "messages": [last_msg],
                    "token_count": input_tokens,  # Accumulates via reducer,
                },
                goto="code_chunking_node"
            )

    # -------CODE CHUNKING NODE-------
    async def code_chunking_node(state: MyState,
    ) -> Command[Literal["reviewer_agent"]]:
        """
        Chunks the code logs into smaller chunks for the reviewer agent, if the code logs are too long to be processed in one go.
        We consider the code too long if it exceeds 5000 tokens. 
        We need to estimate these tokens, since we do not want to split first and then count. 
        NOTE: estimates are more accurate for openai models since they leverage tiktoken. Still, we will probably use Sonnet 4.5 as a reviewer (stronger) 
        """
        code_logs = state["code_logs"]

        code_logs_str = "\n".join([f"```python\n{code_log['input']}\n```\nstdout: ```bash\n{code_log['stdout']}\n```\nstderr: ```bash\n{code_log['stderr']}\n```" for code_log in code_logs])

        # count tokens for the logs
        token_count = reviewer_llm.get_num_tokens(code_logs_str)

        if token_count > 5000:
            # here we split the code logs into big chunks of 5000 tokens each, with big overlap for more context;
            splitter = TokenTextSplitter(
                encoding_name="cl100k_base", # cl100k_base is more model agnostic, and it's the same that get_num_tokens uses for claude models
                chunk_size=5000, 
                chunk_overlap=1000
            )  
            code_logs_chunks = splitter.split_text(code_logs_str)
            return Command(
                update = {
                    "messages" : [HumanMessage(content=f"Code logs were split into {len(code_logs_chunks)} chunks to be reviewed by the reviewer agent. You can read chunks with the read_code_logs_tool, specifying the index of the chunk you want to read.")],
                    "code_logs_chunks" : code_logs_chunks,
                },
                goto="reviewer_agent"
            )
        else:
            return Command(
                update = {
                    "code_logs_chunks" : [code_logs_str],
                },
                goto="reviewer_agent"
            )

    # -------REVIEWER AGENT NODE-------
    # here we need to invoke the reviewer agent on the chat and the code chunks
    # then this agent, based on the `report_status` state flag, decides whether to go to report writer or reroute to analyst with comments
    # NOTE: the state flag is updated by the reviewer agent' tool `assign_to_report_writer`.
    async def reviewer_agent_node(state: MyState,
    ) -> Command[Literal["analyst_agent", "report_writer", "__end__"]]:
        """
        Invokes the reviewer agent.
        Workflow:
            - (1) **performs review**: invokes the reviewer agent: it automatically sees chat history, and with its tools is able to read sources and code chunks, and to read the analysis initial goal.
            It evaluates if the analysis performed was correct and complete.
            - (2) **routing decision**: the reviewer internally decides whether to go to report writer or reroute to analyst with comments. It does so with its tools, 
            `assign_to_report_writer` and `reroute_to_analyst`. These change state flags, respectively: report_status and analysis_status.
            - (3) **checks how many tries were made**: we check how many times the analysis was re-routed to analyst with comments. If it was more than 3, we end flow.
        """

        # invoke the reviewer agent
        result = await agent_reviewer.ainvoke(state)
        # remember to update state with result when routing to next node

        # check review decision - always invoke on result !
        review_route = review_routing(result)  # will be either "analyst_agent", "continue_flow", or "__end__"
        
        if review_route == "analyst_agent":
            return Command(
                goto="analyst_agent",
                update={
                    "analysis_status": "pending", # reset analysis status to pending (default value)
                    "reroute_count": 1,  # increment reroute count
                    "analysis_comments": result["analysis_comments"], # add comments from reviewer
                }
            )
        elif review_route == "__end__":
            return Command(
                goto="__end__"
            )
        elif review_route == "continue_flow":
            pass
        else:
            raise ValueError(f"Invalid route: {review_route}")

        # if we got here, the route is "continue_flow". Thus, the analysis is correct and complete, and the reviewer may have used assign_to_report_writer.
        # if it did, our report_status flag is updated to "assigned" -> go to report writer
        # otherwise it's still "none" -> end flow
        # always invoke on result !
        report_route = write_report_or_end_flow(result)  # will be either "report_writer", "continue_flow", or "__end__"
        if report_route == "report_writer":
            return Command(
                goto="report_writer",
                update={
                    "report_status": "assigned",
                    # update any other state flags as needed
                }
            )
        elif report_route == "__end__":
            return Command(
                goto="__end__"
            )

    # -------REPORT WRITER AGENT NODE-------
    async def write_report_node(state: MyState,
    ) -> Command[Literal["human_approval", "__end__"]]:   # this can actually go to human approval or end, but the end part is done by the write_report_tool. So we only put "human_approval" in Literal.
        """
        Invokes the report writer agent.

        Workflow:
            - (1) **checks for edit instructions:** if they exist, add them to the messages and invoke the agent with the new messages; otherwise, write a new report
            - (2) **invokes the report writer agent**. It uses the `write_report_tool`, which has its own interrupt for HITL.
                So in that invocation, we are implicitly interrupting the tool usage for HITL. The user can either accept the tool usage or reject it:
                    a. If the user accepts, the tool usage continues and the report is written. We are still in this node, so we go to (3)
                    b. If the user rejects, the flow ends - this is done directly in the tool with return Command(goto="__end__")
            - (3) **propagates the write_report_tool updates** with Command(update={...}) and goes to the last human approval node. 
                There, we show the report to the user and ask for approval or edits.
        """

        print("***arrived to report writer")
        report_status = state.get("report_status", "assigned")  # defaults to assigned if it wasn't initialized yet (correct, assigned is default value)
        # report status can now be only: assigned or pending, since we got to the write_report_node from the reviewer agent node
        print(f"***report status in write_report_node: {report_status}")
        # If there are edit instructions, add them to the messages and invoke the agent with the new messages
        if report_status == "pending": # edits: revise existing report
            print("***revising existing report in write_report_node")
            msg = f"Revise the report based on the following instructions: {state['edit_instructions']}. The report you need to revise is: {state['reports'][state['last_report_title']]}"
            messages = state["messages"] + [HumanMessage(content=msg)]
        elif report_status == "assigned": # we got here so the user wants report to be written, first time -> no edits: write a new report
            print("***writing new report in write_report_node")
            msg = "Write a new report based on the analysis performed and the sources used."
            messages = state["messages"] + [HumanMessage(content=msg)]
        else:
            raise ValueError(f"Invalid report status: {report_status}. Since we got to the write_report_node from the analyst agent node, the report status can only be assigned or pending.")

        # invoke on full state but use messages with new sys msg
        print("***invoking report writer agent in write_report_node")
        result = await agent_report_writer.ainvoke({**state, "messages": messages})  # here the agent uses the write_report_tool: report status can be either rejected or pending now
        last_msg = result["messages"][-1]

        goto = edit_report_or_end_flow(result["report_status"])  # if report = pending -> human approval, if report = rejected -> end flow

        return Command(
            update = {  # propagate possible updates
                "messages": [last_msg],
                "reports": result.get("reports", {}),  # Tool updated this
                "last_report_title": result.get("last_report_title"),  # Tool updated this
                "report_status": result["report_status"],  # Tool updated this - update here not really needed (goto uses it) but good to have for safety
                "edit_instructions": ""  # clear edit instructions (if there were any, report writer already used them)
            },
            goto=goto  # can either be human approval (status="pending") or end flow (status="rejected") 
        )

    # -------HUMAN APPROVAL NODE-------
    async def human_approval_node(state: MyState,
    ) -> Command[Literal["report_writer", "__end__"]]:   # this can either go next to report writer (if edits are requested) or end flow
        """
        Last human approval step before ending flow.
        Workflow:
            - (1) **safety check**: if no report has been written yet, raise an error;
            - (2) **interrupt for HITL**: ask the user if they approve the report or request edits;
            - (3) **route based on user input**: if the user approves, end flow; if the user requests edits, go back to report writer node for edits.
        """
        # Safety check
        if state["report_status"] == "pending":
            # pending means that it should have been written by now
            if not state["last_report_title"] or state["last_report_title"] not in state["reports"]:
                raise ValueError("No report has been written yet!")  # if we got to human approval node, it means the report has been written, so we raise an error

        # This message below is only for backend, can be simplified - it's not shown in frontend
        human_input = interrupt({
            "question": "The report has been generated. If you approve the report, input 'yes' - once approved, you can manually edit it. If instead you want the model to edit it, input your desired changes.",
            "report": state["reports"][state["last_report_title"]]
        })
        print(f"***human input in human_approval_node: {human_input}")
        if human_input["type"] == "accept":
            return Command(
                goto="__end__", 
                update={"report_status": "accepted"}
            )  # accepted: therefore, end flow   

        elif human_input["type"] == "edit":
            return Command(
                goto="report_writer", 
                update={
                    "edit_instructions": human_input["edit_instructions"], 
                    "report_status": "pending"
                }
            )  # edit: goes back to report writer node for edits

        else:
            raise ValueError(
                f"Invalid response type: {human_input['type']}"
            )

    # ======= GRAPH  BUILDING =======               

    builder = StateGraph(MyState)
    builder.add_node("analyst_agent", analyst_agent_node)
    builder.add_node("summarize_conversation", summarize_conversation)
    builder.add_edge(START, "analyst_agent")    # notice we do not add an edge to summarize_conversation because we have Command[Literal[...]] in analyst agent node
    builder.add_node("report_writer", write_report_node)  #again, no edge because of Command(goto="...") in write_report_node
    builder.add_node("human_approval", human_approval_node) # no edge because of Command(goto="...") in human_approval_node
    builder.add_node("code_chunking_node", code_chunking_node)
    builder.add_node("reviewer_agent", reviewer_agent_node)
    return builder.compile(checkpointer=checkpointer)
    