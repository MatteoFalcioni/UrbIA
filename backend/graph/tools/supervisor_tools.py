from typing import Annotated
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.types import Command


# helper function to create handoff tool
def create_handoff_tool(*, agent_name: str, description: str | None = None):    #  * means: from here on, all arguments must be passed as keyword arguments
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    # the actual handoff tool
    @tool(name, description=description)
    def handoff_tool(task: Annotated[str, "The task that the subagent should perform"], runtime : ToolRuntime) -> Command:

        tool_msg = ToolMessage(content=f"Successfully transferred to {agent_name}", tool_call_id=runtime.tool_call_id)
        task_msg = HumanMessage(content=f"The agent supervisor adivces you to perform the following task : \n{task}")
        state = runtime.state
        
        return Command(
            goto=agent_name,  
            update={**state, "messages": state["messages"] + [tool_msg] + [task_msg]},   
        )

    return handoff_tool

# Handoffs
assign_to_analyst = create_handoff_tool(
    agent_name="data_analyst",
    description="Assign task to the data analyst agent.",
)

assign_to_report_writer = create_handoff_tool(
    agent_name="report_writer",
    description="Assign task to the report writer agent.",
)

assign_to_reviewer = create_handoff_tool(
    agent_name="reviewer",
    description="Assign task to the reviewer agent.",
)