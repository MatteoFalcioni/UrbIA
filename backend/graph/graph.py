from langgraph.types import Command
from typing_extensions import Literal
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from pydantic import SecretStr
from dotenv import load_dotenv
import os

from backend.graph.summarizer_prompt import summarizer_prompt
from backend.graph.prompt import PROMPT
from backend.graph.tools import internet_search, make_code_sandbox
from backend.graph.api_tools import (
    list_catalog_tool,
    preview_dataset_tool,
    get_dataset_description_tool,
    get_dataset_fields_tool,
    is_geo_dataset_tool,
    get_dataset_time_info_tool,
)
from backend.graph.dataset_tools import (
    select_dataset_tool,
    export_datasets_tool,
    list_datasets_tool,
)
from backend.graph.sit_tools import (
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

    # Create code sandbox tool (will be bound to thread_id later)
    code_sandbox = make_code_sandbox()
    
    # main agent
    agent = create_agent(
        model=llm,
        tools=[
            # Core tools
            # internet_search,
            code_sandbox,
            # Bologna OpenData API tools
            list_catalog_tool,
            preview_dataset_tool,
            get_dataset_description_tool,
            get_dataset_fields_tool,
            is_geo_dataset_tool,
            get_dataset_time_info_tool,
            # Dataset management tools
            select_dataset_tool,
            list_datasets_tool,
            export_datasets_tool,
            # SIT (Geographic Information System) tools
            folium_ortho,
            compare_ortofoto,
            view_3d_model,
        ],
        system_prompt=system_message,  # System prompt for the agent
        name="agent",
        state_schema=MyState,
    )

    # summarization agent
    # Use same API key configuration as main LLM for gpt-4o-mini
    summarizer_kwargs = {"model": "gpt-4.1", "temperature": 0.0}
    if user_api_keys and user_api_keys.get('openai_key'):
        summarizer_kwargs['api_key'] = SecretStr(user_api_keys['openai_key'])
    elif os.getenv('OPENAI_API_KEY'):
        summarizer_kwargs['api_key'] = SecretStr(os.getenv('OPENAI_API_KEY'))
    
    agent_summarizer = create_agent(
        model=ChatOpenAI(**summarizer_kwargs),
        tools=[],
        system_prompt=summarizer_prompt,  
        name="agent_summarizer",
        state_schema=MyState,
    )

    # summarization node
    async def summarize_conversation(state: MyState,
    ) -> Command[Literal["agent"]]:  # after summary we go back to the agent
        """
        Summarizes the conversation with the agent_summarizer
        (!) NOTE: the summary does not persist in chat history, it's only added as system message dynamically, at invokation, when needed.
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
                    "token_count": -1  # reset token count
                    }, 
                goto="agent"  # go back to the agent to answer the question
            )   

    # agent node
    async def agent_node(state: MyState,
    ) -> Command[Literal["summarize_conversation", "__end__"]]:  # if summary is needed go to summarize_conversation, otherwise end flow
        """
        Check token count, if it exceeds threshold goes to summarization, then comes back and invokes the agent
        """ 
        # NOTE: last thing we could add is estimate tokens in summary andreset to those instead of 0... but they are few, so fine for now
        # Check tokens BEFORE invoking agent (Cursor-style: summarize first, then answer)
        current_tokens = state.get("token_count", 0)
        # Use thread-specific context_window or fall back to env default
        effective_context_window = context_window if context_window is not None else CONTEXT_WINDOW
        threshold = effective_context_window * 0.9
        if current_tokens >= threshold:
            # Route to summarization FIRST, then back to agent
            return Command(
                goto="summarize_conversation"
            )
        # If we're here, tokens are fine - proceed with agent
        # get the summary
        summary = state.get("summary", "")

        # if the summary is not empty add it 
        if summary:
            # Add summary to system message **just for the invocation** - it will not be persisted in messages history
            system_message = f"Summary of conversation earlier: {summary}"
            # Let's just add the summary as a human message at the beginning
            messages_with_summary = [HumanMessage(content=system_message)] + state["messages"]
            result = await agent.ainvoke({"messages": messages_with_summary})
        else:
            messages = state["messages"]

        # invoke the agent
        result = await agent.ainvoke({"messages": messages})
        last_msg = result["messages"][-1]
        meta = last_msg.usage_metadata
        input_tokens = meta["input_tokens"] if meta else 0

        # update the token count and add message
        return Command(
                update={
                    "messages": [last_msg],
                    "token_count": input_tokens  # Accumulates via reducer
                },
                goto="__end__"
            )

    builder = StateGraph(MyState)
    builder.add_node("agent", agent_node)
    builder.add_node("summarize_conversation", summarize_conversation)
    builder.add_edge(START, "agent")    # notice we do not add an edge to summarize_conversation because we have Command[Literal[...]]
    return builder.compile(checkpointer=checkpointer)
    