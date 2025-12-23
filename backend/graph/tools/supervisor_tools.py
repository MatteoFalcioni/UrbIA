from typing import Annotated
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.types import Command, interrupt


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
            update={**state, "messages": state["messages"] + [tool_msg] + [task_msg]},
            graph=Command.PARENT,
        )

    return handoff_tool


def create_handoff_tool_HITL(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool_HITL(
        task: Annotated[str, "The task that the subagent should perform"],
        runtime: ToolRuntime,
    ) -> Command:

        usr_response = interrupt(
            value=f"The agent supervisor wants to call the {agent_name} to perform the following task: *{task}*.\nDo you approve?"
        )

        # Handle case where usr_response might be None or not properly structured
        if usr_response is None:
            raise ValueError("Resume value is None - user might have cancelled the interrupt")
        
        if not isinstance(usr_response, dict):
            raise ValueError(f"Resume value must be a dict, got {type(usr_response)}: {usr_response}")
        
        if 'decision' not in usr_response:
            raise ValueError(f"Resume value missing 'decision' key. Got: {usr_response}")

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
            raise ValueError(f"Invalid user response: {usr_response['decision']}")

        state = runtime.state

        return Command(
            goto=goto,
            update={**state, "messages": state["messages"] + msgs},
            graph=Command.PARENT,
        )

    return handoff_tool_HITL


# Handoffs
assign_to_analyst = create_handoff_tool(
    agent_name="data_analyst",
    description="Assign task to the data analyst agent.",
)

assign_to_reviewer = create_handoff_tool(
    agent_name="reviewer",
    description="Assign task to the reviewer agent.",
)

# Handoff with Human In The Loop for report writer
assign_to_report_writer = create_handoff_tool_HITL(
    agent_name="report_writer",
    description="Assign task to the report writer agent.",
)

# === Show code tool ===
# Just a tool to show to the user the code executed by the analyst 