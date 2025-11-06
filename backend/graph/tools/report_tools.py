from typing_extensions import Annotated
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command

@tool(
    name_or_callable="assign_to_report_writer",
    description="Use this to assign the task to the report writer when analysis is complete."
)
def assign_to_report_writer_tool(
    reason: Annotated[str, "Brief reason why analysis is complete and report should be written"],
    runtime: ToolRuntime
) -> Command:
    """
    Assigns the task to the report writer when analysis is complete. 
    Since literal tool assingment was not working, we use this to update the state flag and conditional edge to route to the report writer.
    """
    print("***assigning to report writer in assign_to_report_writer_tool")
    return Command(
        update={
            "messages": [ToolMessage(
                content=f"Analysis complete. {reason}. Assigning to report writer.", 
                tool_call_id=runtime.tool_call_id
            )],
            "report_status" : "assigned"   # first assignment: assigned to report writer
        }
    )


from langgraph.types import interrupt

@tool
def assign_to_report_writer(
    report_title: Annotated[str, "The title of the report"],
    report_content: Annotated[str, "The content of the report"],
    runtime: ToolRuntime
)->Command:
    """
    Write a report of the analysis performed.
    Interrupts the model to ask for approval before writing the report.
    """
    state = runtime.state   
    print("***writing report in assign_to_report_writer")
    # interrupt only if the writer is not editing an existing report
    if state["report_status"] == "assigned":  # means this is the first time the report is being written
        print(f"***report status is 'assigned' in assign_to_report_writer: asking for report writing approval in assign_to_report_writer")

        # refine this message in frontend and simplify it here in backend (the user will not see this below)
        response = interrupt(f"The model has finished its analysis and wants to write a report. To continue, input 'yes'. To reject, input 'no'.")

        if response["type"] == "accept":
            print("***accepted write report in assign_to_report_writer")
            pass  # accepted write report: therefore, continue flow (will update status to pending at the end of the tool)
        elif response["type"] == "reject":
            print("***rejected write report in assign_to_report_writer")
            return Command(update={
                "messages": [
                    ToolMessage(
                        content="Report writing rejected by the user.", 
                        tool_call_id=runtime.tool_call_id
                    )], 
                    "report_status" : "rejected"  # rejected write report
                })  
        else:
            raise ValueError(f"Invalid response type: {response['type']}")
    
    elif state["report_status"] == "pending":  # means this is an edit to an existing report
        print(f"***report status is 'pending' in assign_to_report_writer: editing existing report in assign_to_report_writer")
        pass  # edit does not need interruptions
    
    else:
        raise ValueError(f"Invalid report status: {state['report_status']}. Since we got to the assign_to_report_writer from the report writer node, the report status can only be assigned or pending.")

    report_dict = {report_title: report_content}  # show this to the user in frontend in a nice way (like sidebar with the full report in markdown format)

    return Command(  
        update = {
            "messages" : [ToolMessage(content="Report written successfully.", tool_call_id=runtime.tool_call_id)],
            "reports" : report_dict,
            "last_report_title" : report_title,
            "edit_instructions" : "",  # clear if there were any 
            "report_status" : "pending"  # pending - can be edited
        }
    )

# probably should make a modify_report() tool that can be used to modify an existing report after it was approved. 
# But maybe that can be done in the frontend only.

@tool
def write_source_tool(dataset_id: Annotated[str, "the dataset_id, i.e. the source"], runtime: ToolRuntime) -> Command:
    """
    Write a source to the list of sources. If the dataset is already in the list, return a message that says so..
    """
    state = runtime.state
    sources = state["sources"]
    if dataset_id not in sources:
        sources_update = [dataset_id]
        return Command(update={"messages" : [ToolMessage(content=f"Dataset {dataset_id} added to sources.", tool_call_id=runtime.tool_call_id)], "sources" : sources_update})
    else:
        return Command(update={"messages" : [ToolMessage(content=f"Dataset {dataset_id} already in sources.", tool_call_id=runtime.tool_call_id)]})

@tool
def read_sources_tool(
    runtime: ToolRuntime
) -> Command:
    """
    Get the sources used in the analysis.
    """
    state = runtime.state
    sources = state["sources"]
    sources_str = "\n".join([f"- {source}" for source in sources])

    return Command(
        update = {
            "messages" : [ToolMessage(content=f"Sources: {sources_str}", tool_call_id=runtime.tool_call_id)],
        }
    )

@tool
def read_code_logs_tool(
    index: Annotated[int, "The index of the code log chunk to read"],
    runtime: ToolRuntime
) -> Command:
    """
    Read a chunk of code logs.
    """
    state = runtime.state
    code_logs_chunks = state["code_logs_chunks"]

    if index < 0 or index >= len(code_logs_chunks):
        return Command(update={"messages" : [ToolMessage(content=f"Invalid index: {index}. Index must be between 0 and {len(code_logs_chunks) - 1}.", tool_call_id=runtime.tool_call_id)]})
    
    return Command(update={"messages" : [ToolMessage(content=f"Code log chunk {index}: \n\n{code_logs_chunks[index]}", tool_call_id=runtime.tool_call_id)]})
