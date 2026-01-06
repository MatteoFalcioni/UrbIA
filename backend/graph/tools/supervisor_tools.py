from typing import Annotated
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.types import Command, interrupt

# ------ developer note -------
# NOTE: we are considering 'an analysis' as one conversation turn: user -> graph -> user. 
# This may be an oversimplification:
# what if the user is not satisfied with an operation performed by the analyst and wants to change a specific action?
# Like, say, remake a plot. The analyst loses sources, code and todos when this new task is assigned, but the 'analysys' is still the same.
# Right now we are sweeping this under the rug, but may be causing problems in the future.
# A fix could be to use a key for each analysis, say an 'analysis title'. It could be AI generated as well.
# -----------------------------

# ------ developer note -------
# NOTE: When we use handoff tools, we pass the state updates as well in the Command. 
# That is fine because we want to add always a tool message, but there is a problem with multi-turn conversations:
# if we pass state as is in the handoff, after having modified state previously, it will not be reset - of course, that's how LG is supposed to work: be stateful.
# but this means then that we may want to perform some additional state management in the handoffs, like resetting state values.
# (!) We will only do it for the analyst because it's the first agent that is hit in our workflow 
# -----------------------------


# === Handoff Tools ===
# NOTE: these are structured with graph.PARENT because we do not have a supervisor node right now.
# That means that the subagents are considered by langgraph as **subgraphs** and therefore we need Command.PARENT in the handoff.

# helper function to create handoff tool
def create_handoff_tool(
    *, agent_name: str, description: str | None = None
):  #  * means: from here on, all arguments must be passed as keyword arguments
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    # the actual handoff tool
    @tool(name, description=description)
    def handoff_tool(
        task: Annotated[str, "The task that the subagent should perform"],
        runtime: ToolRuntime,
    ) -> Command:

        tool_msg = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            tool_call_id=runtime.tool_call_id,
        )
        task_msg = HumanMessage(
            content=f"The agent supervisor advices you to perform the following task : \n{task}"
        )
        state = runtime.state

        return Command(
            goto=agent_name,
            update={"messages": state["messages"] + [tool_msg] + [task_msg]},
            graph=Command.PARENT,
        )

    return handoff_tool

# === Handoff Tools with Human In The Loop ===
def create_handoff_tool_HITL(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool_HITL(
        task: Annotated[str, "The task that the subagent should perform"],
        runtime: ToolRuntime,
    ) -> Command:

        usr_response = interrupt(
            value=f"The agent supervisor wants to call the {agent_name} to perform the following task: *{task}*\nDo you approve?"
        )

        # Handle case where usr_response might be None or not properly structured
        if usr_response is None:
            error_msg = "Resume value is None - user might have cancelled the interrupt"
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Error: {error_msg}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )
        
        if not isinstance(usr_response, dict):
            error_msg = f"Resume value must be a dict, got {type(usr_response)}: {usr_response}"
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Error: {error_msg}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )
        
        if 'decision' not in usr_response:
            error_msg = f"Resume value missing 'decision' key. Got: {usr_response}"
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Error: {error_msg}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

        if usr_response['decision'] == "accept":
            goto = agent_name
            tool_msg = [
                ToolMessage(
                    content=f"Successfully transferred to {agent_name}",
                    tool_call_id=runtime.tool_call_id,
                )
            ]
            task_msg = [
                HumanMessage(
                    content=f"The agent supervisor advices you to perform the following task : \n{task}"
                )
            ]
            msgs = tool_msg + task_msg
        elif usr_response['decision'] == 'reject':
            goto = "supervisor"
            msgs = [
                ToolMessage(
                    content=f"Routing to {agent_name} was rejected by the user.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        else: 
            error_msg = f"Invalid user response: {usr_response['decision']}"
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Error: {error_msg}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

        state = runtime.state

        return Command(
            goto=goto,
            update={"messages": state["messages"] + msgs},
            graph=Command.PARENT,
        )

    return handoff_tool_HITL


# === Handoff Tool with state management for data analyst agent ===
def create_handoff_to_data_analyst():

    # hardcoded for data analyst agent
    name = "transfer_to_data_analyst"
    description = "Assign task to the data analyst agent."

    @tool(name, description=description)
    def handoff_to_data_analyst(
        task: Annotated[str, "The task that the subagent should perform"],
        runtime: ToolRuntime,
    ) -> Command:
        """
        Handoff tool with "state management".
        "State management" means that we apply operations to the state passed from the supervisor to the data analyst before assignment.
        Specifically, we want to reset all state vars that should be treated as independent between different analysis, i.e.:
            - review scores
            - todos
            - code logs
            - code logs chunks
            - sources
        """

        tool_msg = ToolMessage(
            content="Successfully transferred to data analyst",
            tool_call_id=runtime.tool_call_id,
        )
        task_msg = HumanMessage(
            content=f"The agent supervisor advices you to perform the following task : \n{task}"
        )
        state = runtime.state

        return Command(
            goto="data_analyst",
            update={
                "messages": state["messages"] + [tool_msg] + [task_msg],  # do not reset: msgs, reports, analysis comments and reroute_count
                "completeness_score": 0,
                "relevancy_score": 0,
                "final_score": 0, 
                "todos" : [],  
                "code_logs" : [],
                "code_logs_chunks" : [],
                "sources" : []  
                },
            graph=Command.PARENT,
        )

    return handoff_to_data_analyst


# Handoffs
assign_to_analyst = create_handoff_to_data_analyst()

assign_to_reviewer = create_handoff_tool(
    agent_name="reviewer",
    description="Assign task to the reviewer agent.",
)

# Handoff with Human In The Loop for report writer
assign_to_report_writer = create_handoff_tool_HITL(
    agent_name="report_writer",
    description="Assign task to the report writer agent.",
)