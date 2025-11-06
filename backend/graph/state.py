from langchain.agents import AgentState
from typing import Annotated, Literal

def update_token_count(token_count: int | None = None, token_used: int | None = None) -> int:
    """
    Updates the token count
    """
    # init safeguards
    if token_count is None:
        token_count = 0
    if token_used is None:
        token_used = 0
        
    # a value of -1 means reset to 0
    if token_used == -1:
        return 0
    else:
        return token_count + token_used

def merge_dicts(
    left: dict[str, str] | None = None,
    right: dict[str, str] | None = None
) -> dict[str, str]:
    """Merge two dictionaries. Left takes precedence over right. Used for reports."""
    if left is None:
        left = {}
    if right is None:
        right = {}
    return {**left, **right}

def merge_dicts_nested(
    left: dict[str, dict[str, str]] | None = None, 
    right: dict[str, dict[str, str]] | None = None
) -> dict[str, dict[str, str]]:
    """Merge two nested dictionaries. Left takes precedence over right. Used for sources."""
    if left is None:
        left = {}
    if right is None:
        right = {}
    return {**left, **right}

def list_add(
    left: list[dict[str, str]] | None = None,
    right: list[dict[str, str]] | None = None
) -> list[dict[str, str]]:
    """Add a new item to a list. Used for code. No deduplication - running the same code twice is meaningful."""
    if left is None:
        left = []
    if right is None:
        right = []
    
    return left + right


def list_replace_str(
    left: list[str] | None,
    right: list[str] | None
) -> list[str]:
    """Replace list of strings entirely instead of concatenating. Used for code logs chunks"""
    if left is None:
        left = []
    if right is None:
        right = []

    return right

def str_replace(
    left: str | None,
    right: str | None
) -> str:
    """Update a string just by replacing it. Reducer needed to initialize when None"""
    if left is None:
        left = ""
    if right is None:
        right = ""

    return right

def status_replace(
    left: Literal["none", "assigned", "pending", "rejected", "accepted"] | None,
    right: Literal["none", "assigned", "pending", "rejected", "accepted"] | None
) -> Literal["none", "assigned", "pending", "rejected", "accepted"]: # "none" is the initial state
    """Update the report status just by replacing strings. Reducer needed to initialize when None"""
    if left is None:
        left = "none"
    if right is None:
        right = "none"

    return right

class MyState(AgentState):
    # summary and token count features (core)
    summary : Annotated[str, str_replace]
    token_count : Annotated[int, update_token_count]
    # report features 
    sources : Annotated[list[str], list_add] # list of dataset ids
    reports: Annotated[dict[str, str], merge_dicts]  # key is the title, value is the content 
    report_status : Annotated[Literal["none", "assigned", "pending", "rejected", "accepted"], status_replace]
    last_report_title : Annotated[str, str_replace]  # title of the last report written
    edit_instructions : Annotated[str, str_replace]  # instructions for the report writer to edit the report
    code_logs: Annotated[list[dict[str, str]], list_add]  # list of dicts (we need chronological order!), each dicts is input and output of a code block (out can be stdout or stderr or both)
    code_logs_chunks: Annotated[list[str], list_replace_str]  # list of strings, each string is a chunk of already ordered code logs - we first stringify code_logs correclty, then separate it in chunks (see get_code_logs_tool in report_tools.py)