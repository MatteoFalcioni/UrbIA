from typing_extensions import Annotated
from langchain.tools import tool, ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.types import Command


@tool
def read_sources_tool(runtime: ToolRuntime) -> Command:
    """
    Get the sources used in the analysis.
    """
    state = runtime.state
    sources = state["sources"]
    sources_str = "\n".join([f"- {source}" for source in sources])

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Sources: {sources_str}", tool_call_id=runtime.tool_call_id
                )
            ],
        }
    )


@tool
def read_code_logs_tool(
    index: Annotated[
        int,
        "The index of the code log chunk to read. At least index 0 will always exist.",
    ],
    runtime: ToolRuntime,
) -> Command:
    """
    Read a chunk of code logs by specifying the index of the chunk.
    """
    state = runtime.state
    code_logs_chunks = state["code_logs_chunks"]

    if index < 0 or index >= len(code_logs_chunks):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Invalid index: {index}. Index must be between 0 and {len(code_logs_chunks) - 1}.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Code log chunk {index}: \n\n{code_logs_chunks[index]}",
                    tool_call_id=runtime.tool_call_id,
                )
            ]
        }
    )


@tool
def read_analysis_objectives_tool(runtime: ToolRuntime) -> Command:
    """
    Use this to read the analysis objectives and their status.
    """
    state = runtime.state
    objectives = state.get("todos", "")
    if objectives:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Todos:\n {objectives}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    else:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="No todos found.", tool_call_id=runtime.tool_call_id
                    )
                ]
            }
        )


@tool
async def approve_analysis_tool(runtime: ToolRuntime) -> Command:
    """
    Use this to approve the analysis.

    """
    print("***approving analysis in approve_analysis_tool")
    return Command(
        update={
            "analysis_status": "approved",
            "analysis_comments": "",  # reset any analysis comments (they are for rejected analyses)
            "messages": [
                ToolMessage(
                    content="Analysis approved from the reviewer.",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "reroute_count" : -1  # resets to 0
        }
    )


@tool
async def reject_analysis_tool(
    comments: Annotated[str, "Comments for the analyst to improve the analysis"],
    runtime: ToolRuntime,
) -> Command:
    """
    Use this to reject the analysis, with constructive criticism for the analyst to improve the analysis.
    Arguments:
        comments: Constructive criticism for the analyst to improve the analysis.
    """
    print(f"***rejecting analysis in reject_analysis_tool: {comments}")
    return Command(
        update={
            "analysis_status": "rejected",
            "analysis_comments": comments,
            "reroute_count": 1,
            "messages": [
                ToolMessage(
                    content=f"Analysis rejected by reviewer, with the following comments for the analyst:\n {comments}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
async def update_completeness_score(
    grade: Annotated[int, "The grade of the completeness score"], runtime: ToolRuntime
) -> Command:
    """
    Use this to update the completeness score.
    Arguments:
        grade: The grade of the completeness score
    """

    state = runtime.state
    num_todos = len(state.get("todos", []))

    if num_todos == 0:  # no update to score means default value, which is 0 
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="todo list was empty: completeness score cannot be computed and therefore willbe penalized with a score = 0",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    print(f"***updating completeness score in update_completeness_score: {grade}")

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Completeness score updated to: {grade/num_todos:.2f}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "completeness_score": grade / num_todos,
        }
    )


@tool
async def update_relevancy_score(
    grade: Annotated[int, "The grade of the relevancy score"], runtime: ToolRuntime
) -> Command:
    """
    Use this to update the relevancy score.
    Arguments:
        grade: The grade of the relevancy score
    """

    state = runtime.state
    num_sources = len(state.get("sources", []))

    if num_sources == 0:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="source list was empty: relevancy score cannot be computed, and will be penalized with a score = 0",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    print(f"***updating relevancy score in update_relevancy_score: {grade}")
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Relevancy score updated to: {grade/num_sources:.2f}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "relevancy_score": grade / num_sources,
        }
    )
