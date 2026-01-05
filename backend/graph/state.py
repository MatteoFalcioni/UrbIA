from langchain.agents import AgentState
from typing import Annotated, Literal


def list_add_dicts(
    left: list[dict] | None = None, right: list[dict] | None = None
) -> list[dict]:
    """
    Add a new item to a list. No deduplication.
    Used for:
        * code: running the same code twice is meaningful;
    
    Added reset: if we pass an empty list, and left is not empty, it will return [].
    """
    if left is None:
        left = []
    if right is None:
        right = []

    if left is not None and len(right) == 0:
        return []

    return left + right


def dict_merge(
    left: dict[str, str] | None = None, right: dict[str, str] | None = None
) -> dict[str, str]:
    """
    Merge two dicts. Right overwrites left for duplicate keys.
    Used for:
        * reports: we want to accumulate several reports over different analyses
    
    Added reset: if we pass an empty dict, and left is not empty, it will return {}.
    """
    if left is None:
        left = {}
    if right is None:
        right = {}

    if left is not None and len(right) == 0:
        return {}

    return {**left, **right}


def list_replace(left: list[str] | None, right: list[str] | None) -> list[str]:
    """
    Replace list of strings entirely instead of concatenating. 
    Used for:
        * code logs chunks: these are the logs read by the agent at each run - overwrite
        * sources: sources are the datasets used in the current analysis - overwriting the previous ones is needed;
    """
    if left is None:
        left = []
    if right is None:
        right = []

    return right


def str_replace(left: str | None, right: str | None) -> str:
    """Update a string just by replacing it. Reducer needed to initialize when None"""
    if left is None:
        left = ""
    if right is None:
        right = ""

    return right


def status_replace(
    left: (
        Literal["pending", "approved", "rejected", "limit_exceeded", "end_flow"] | None
    ),
    right: (
        Literal["pending", "approved", "rejected", "limit_exceeded", "end_flow"] | None
    ),
) -> Literal["pending", "approved", "rejected", "limit_exceeded", "end_flow"]:
    if left is None:
        left = "pending"
    if right is None:
        right = "pending"
    return right


def int_add(left: int | None, right: int | None) -> int:
    """
    Increment a counter. Used for reroute_count
    Added reset: if value is -1 and lft is not None, reset counter to 0.
    """
    if left is None:
        left = 0
    if right is None:
        right = 0

    if left is not None and right == -1:
        return 0
        
    return left + right


def float_replace(left: float | None, right: float | None) -> float:
    if left is None:
        left = 0.0
    if right is None:
        right = 0.0
    return right


# NOTE: (!) CRUCIAL
# If we want to propagate the todos state var, added by the Middleware, to the general state,
# we need to still define the todos in state with a reducer.
# If we try to pass the todos update to the general state, this will fail because the middleware
# automatically adds the state var only to the agent that has that middleware!

class MyState(AgentState):
    """
    Custom state for the graph. Inherits from AgentState -> automatically contains messages.

    Additional state variables:
        * sources (`list[str]`): 
            list of dataset ids used in the analysis;
        * reports (`dict[str, str]`): 
            dict of reports written by the report writer agent. 
            Key is title, value is content. They accumulate over different analyses;
        * last_report_title (`str`): 
            title of the last report written;
        * code_logs (`list[dict[str, str]]`): 
            list of dicts containing input code and output+err logs;
        * code_logs_chunks (`list[str]`): 
            list of strings, each string is a chunk of already ordered code logs. 
            Needed for better code reading for the agents;
        * analysis_status (`Literal["pending", "approved", "rejected", "limit_exceeded", "end_flow"]`): 
            status of the analysis. Can be: pending, approved, rejected, limit_exceeded, end_flow;
        * analysis_comments (`str`): 
            comments for the analyst to improve the analysis;
        * reroute_count (`int`): 
            counter of how many times the analysis was re-routed to analyst with comments;
        * completeness_score (`float`): 
            score of the completeness of the analysis;
        * relevancy_score (`float`): 
            score of the relevancy of the analysis;
        * final_score (`float`): 
            final score of the analysis;
        * todos (`list[dict]`): 
            list of todos for the analyst to perform. 
    """

    # ---- report features ----
    sources: Annotated[
        list[str], list_replace
    ]  # list of dataset ids; NOTE: we are replacing the list of sources entirely after each analysis
    reports: Annotated[
        dict[str, str], dict_merge
    ]  # key is the title, value is the content
    last_report_title: Annotated[str, str_replace]  # title of the last report written
    code_logs: Annotated[
        list[dict], list_add_dicts
    ]  # list of dicts (we need chronological order!), each dicts is input and output of a code block (out can be stdout or stderr or both)
    code_logs_chunks: Annotated[
        list[str], list_replace
    ]  # list of strings, each string is a chunk of already ordered code logs - we first stringify code_logs correclty, then separate it in chunks (see get_code_logs_tool in report_tools.py)
    # ---- review features ----
    ## analysys
    analysis_status: Annotated[
        Literal["pending", "approved", "rejected", "limit_exceeded", "end_flow"],
        status_replace,
    ]
    analysis_comments: Annotated[
        str, str_replace
    ]  # comments for the analyst to improve the analysis
    ## reroute after review
    reroute_count: Annotated[
        int, int_add
    ]  # counter of how many times the analysis was re-routed to analyst with comments
    ## scores
    completeness_score: Annotated[float, float_replace]
    relevancy_score: Annotated[float, float_replace]
    final_score: Annotated[float, float_replace]
    # ---- todos ----
    todos: list[dict]
