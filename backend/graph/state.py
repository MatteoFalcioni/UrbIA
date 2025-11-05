from langchain.agents import AgentState
from typing import Annotated

# we add reducers not mainly for conurrency of update, but more to define rules for updates
# we want string updates for summary (if a summary exists, we want to extend it) so no reducers needed
# but we want to count tokens AND reset to 0 after each summary update

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
    return right if right is not None else (left if left is not None else [])


def str_replace(
    left: str | None,
    right: str | None
) -> str:
    """Update a string just by replacing it. Reducer needed to initialize when None"""
    if left is None:
        return ""
    else:
        return right

def bool_replace(
    left: bool | None,
    right: bool | None
) -> bool:
    """Update a boolean just by replacing it. Reducer needed to initialize when None"""
    if left is None:
        return False
    else:
        return right


class MyState(AgentState):
    # summary and token count features (core)
    summary : str   # No reducer - just replace
    token_count : Annotated[int, update_token_count]
    # write report features 
    sources : Annotated[dict[str, dict[str, str]], merge_dicts_nested] # key is the dataset id, value is a dict with desc, url
    reports: Annotated[dict[str, str], merge_dicts]  # key is the title, value is the content 
    write_report : Annotated[bool, bool_replace]
    last_report_title : Annotated[str, str_replace]
    edit_instructions : Annotated[str, str_replace]
    code_logs: Annotated[list[dict[str, str]], list_add]  # list of dicts (we need chronological order!), each dicts is input and output of a code block (out can be stdout or stderr or both)
    code_logs_chunks: Annotated[list[str], list_replace_str]  # list of strings, each string is a chunk of already ordered code logs - we first stringify code_logs correclty, then separate it in chunks (see get_code_logs_tool in report_tools.py)