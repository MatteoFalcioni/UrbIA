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
    
    return Command(
        update={
            "messages": [ToolMessage(
                content=f"Analysis complete. {reason}. Assigning to report writer.", 
                tool_call_id=runtime.tool_call_id
            )],
            "write_report" : True
        }
    )


from langgraph.types import interrupt

@tool
def write_report_tool(
    report_title: Annotated[str, "The title of the report"],
    report_content: Annotated[str, "The content of the report"],
    runtime: ToolRuntime
)->Command:
    """
    Write a report of the analysis performed.
    Interrupts the model to ask for approval before writing the report.
    """
    state = runtime.state

    # interrupt only if the writer is not editing an existing report
    if state["edit_instructions"] == "":

        # refine this message in frontend and simplify it here in backend (the user will not see this below)
        response = interrupt(f"The model has finished its analysis and wants to write a report. To continue, input 'yes'. To reject, input 'no'.")

        if response["type"] == "accept":
            pass  # accepted write report: therefore, continue flow
        elif response["type"] == "reject":
            return Command(goto="__end__")  # rejected write report: therefore, end flow
        else:
            raise ValueError(f"Invalid response type: {response['type']}")

    report_dict = {report_title: report_content}  # show this to the user in frontend in a nice way (like sidebar with the full report in markdown format)

    return Command(  # accepted write report: therefore, continue flow
        update = {
            "messages" : [ToolMessage(content="Report written successfully.", tool_call_id=runtime.tool_call_id)],
            "reports" : report_dict,
            "last_report_title" : report_title,
            "edit_instructions" : ""  # clear if there were any 
        }
    )

# probably should make a modify_report() tool that can be used to modify an existing report after it was approved. But maybe that can 
# be frontend only.


@tool
def get_sources_tool(
    runtime: ToolRuntime
) -> Command:
    """
    Get the sources used in the analysis.
    """
    state = runtime.state

    sources = state["sources"]

    sources_str = "\n".join([f"- {source['desc']}: {source['url']}" for source in sources.values()])

    return Command(
        update = {
            "messages" : [ToolMessage(content=f"Sources: {sources_str}", tool_call_id=runtime.tool_call_id)],
        }
    )

@tool 
def get_code_logs_tool(
    runtime: ToolRuntime
) -> Command:
    """
    Get the code logs used in the analysis. You should call this only once at the beginning of your fact checking workflow. 
    You can then read the chunks with the read_code_logs_tool, specifying the index of the chunk you want to read.
    """
    state = runtime.state

    code_logs = state["code_logs"]

    code_logs_str = "\n".join([f"```python\n{code_log['input']}\n```\nstdout: ```bash\n{code_log['stdout']}\n```\nstderr: ```bash\n{code_log['stderr']}\n```" for code_log in code_logs])

    # here we split the code logs into big chunks of 7500 tokens each;
    from langchain_text_splitters import TokenTextSplitter
    splitter = TokenTextSplitter(model_name="gpt-4.1", chunk_size=7500, chunk_overlap=1000)
    code_logs_chunks = splitter.split_text(code_logs_str)
    num_chunks = len(code_logs_chunks)

    return Command(
        update = {
            "messages" : [ToolMessage(content=f"Code logs retrieved successfully, and split into {num_chunks} chunks. You can read chunks with the read_code_logs_tool, specifying the index of the chunk you want to read.", tool_call_id=runtime.tool_call_id)],
            "code_logs_chunks" : code_logs_chunks,
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
