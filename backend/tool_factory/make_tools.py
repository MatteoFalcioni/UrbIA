# langgraph_sandbox/tool_factory/make_tools.py
from __future__ import annotations

import json
from typing import Callable, Optional, Awaitable, Any

from pydantic import BaseModel, Field, ConfigDict
from typing_extensions import Annotated
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain_core.messages import ToolMessage
import os

try:
    # Try relative imports first (when used as a module)
    from ..sandbox.session_manager import SessionManager
except ImportError:
    # Fall back to absolute imports (when run directly)
    from backend.sandbox.session_manager import SessionManager


def _default_get_session_key() -> str:
    return "conv"  # TODO: user/thread id


def make_code_sandbox_tool(
    *,
    session_manager: SessionManager,
    session_key_fn: Callable[[], str] = _default_get_session_key,
    name: str = "code_sandbox",
    description: str = (
        "Execute Python code in a session-pinned Docker sandbox. "
        "Returns stdout and any artifacts from /session/artifacts. "
        "Always use print(...) to show results."
    ),
    timeout_s: int = 120,
) -> Callable:
    """
    Factory that returns a LangChain Tool for executing code inside the sandbox.
    This tool only executes code - dataset loading is handled separately.

    Usage:
        session_manager = SessionManager(...)
        code_sandbox = make_code_sandbox_tool(
            session_manager=session_manager,
            session_key_fn=lambda: "conv",
        )
    """

    class ExecuteCodeArgs(BaseModel):
        code: str = Field(description="Python code to execute in the sandbox.")
        runtime: ToolRuntime
        model_config = ConfigDict(arbitrary_types_allowed=True)

    # The implementation closes over session_manager, session_key_fn
    async def _impl(
        code: Annotated[str, "Python code to run"],
        runtime: ToolRuntime,
    ) -> Command:

        sid = session_key_fn()
        session_manager.start(sid)

        # Get database session and thread_id from context (set by the API layer)
        try:
            from backend.graph.context import get_db_session, get_thread_id
            db_session = get_db_session()
            thread_id = get_thread_id()
        except ImportError:
            # Fallback if context module not available
            db_session = None
            thread_id = None

        # Execute code - no dataset loading here anymore
        result = await session_manager.exec(
            sid, code, timeout=timeout_s, db_session=db_session, thread_id=thread_id, tool_call_id=runtime.tool_call_id
        )

        payload = {
            "stdout": result.get("stdout", ""),
            "stderr": result.get("error", "") or result.get("stderr", ""),
            "session_dir": result.get("session_dir", ""),
        }

        artifacts = result.get("artifacts", [])

        # Create structured artifact information for the UI. ALways include it in artifacts to be retrieved in main 
        structured_artifacts = []
        if artifacts:
            for artifact in artifacts:
                artifact_data = {
                    "name": artifact.get('name', 'unknown'),
                    "mime": artifact.get('mime', 'unknown'),
                    "url": artifact.get('url', ''),
                    "size": artifact.get('size', 0)
                }
                structured_artifacts.append(artifact_data)

        # Don't include artifact URLs in chat content (keeps model focused on stdout/stderr)
        # If you want to show URLs to the model, set this to True
        in_chat_url = False
        
        if in_chat_url:
        # Create human-readable artifact summary for content
            artifact_summary = ""
            if artifacts:
                artifact_summary = "\n\n📁 Generated Artifacts:\n"
                for artifact in artifacts:
                    filename = artifact.get('name', 'unknown')
                    size = artifact.get('size', 0)
                    mime = artifact.get('mime', 'unknown')
                    download_url = artifact.get('url', '')
                    artifact_summary += f"  • {filename} ({mime}, {size} bytes)\n"
                    if download_url:
                        artifact_summary += f"    Download: {download_url}\n"
                artifact_summary += "\n"        
            # Combine stdout with artifact information
            content = result.get("stdout", "") + artifact_summary + json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            # ELSE, only include it in artifacts to be retrieved in main (in_chat_url=False)
            content = json.dumps(payload, ensure_ascii=False, indent=2)

        # Only include artifact parameter if there are actually artifacts
        if len(structured_artifacts) > 0:
            tool_msg = ToolMessage(
                content=content,
                artifact=structured_artifacts,  
                tool_call_id=runtime.tool_call_id,
            )
        else:
            tool_msg = ToolMessage(
                content=content,
                tool_call_id=runtime.tool_call_id,
            )
        return Command(update={"messages": [tool_msg]})

    # Return a LangChain Tool by applying the decorator at factory time
    return tool(
        name_or_callable=name,
        description=description,
        args_schema=ExecuteCodeArgs,
    )(_impl)


def make_select_dataset_tool(
    *,
    session_manager: SessionManager,
    session_key_fn: Callable[[], str] = _default_get_session_key,
    fetch_fn: Any,
    client: Optional[object] = None,
    name: str = "select_dataset",
    description: str = (
        "Select a dataset to load into sandbox as a parquet file. "
        "This will fetch and stage the dataset immediately. "
        "In HYBRID mode, this skips datasets already mounted at /data/."
    ),
) -> Callable:
    """
    Factory that returns a LangChain Tool for selecting and loading datasets into the sandbox.
    
    Args:
        session_manager: SessionManager instance to use for container operations
        session_key_fn: Function to get current session key
        fetch_fn: Function to fetch dataset bytes by ID (could have also client as input parameter)
        client: Optional client object to pass to fetch_fn
        name: Tool name
        description: Tool description
        
    Returns:
        LangChain tool function
    """
    
    class SelectDatasetArgs(BaseModel):
        dataset_id: Annotated[str, Field(description="The dataset ID")]
        runtime: ToolRuntime
        model_config = ConfigDict(arbitrary_types_allowed=True)
    
    async def _impl(
        dataset_id: Annotated[str, "The dataset ID"], 
        runtime: ToolRuntime
    ) -> Command:
        """Select and load a dataset into the sandbox."""
        try:
            from ..config import Config
        except ImportError:
            from config import Config
        from ..dataset_manager.cache import DatasetStatus, add_entry
        from ..dataset_manager.sync import load_pending_datasets
        
        # Load configuration
        cfg = Config.from_env()
        session_id = session_key_fn()

        # safety check: clean dataset id of any extension (like .parquet, .csv, etc.)
        # if mistakenly set by llm
        dataset_id = dataset_id.split(".")[0]
        
        # Add the dataset to cache with PENDING status
        cache_path = add_entry(cfg, session_id, dataset_id, status=DatasetStatus.PENDING)
        
        try:
            # Start the session if not already started
            session_manager.start(session_id)
            
            # Create wrapper function for fetch_fn that includes client if provided
            if client is not None:
                async def fetch_dataset_wrapper(ds_id: str) -> bytes:
                    return await fetch_fn(client, ds_id)
            else:
                async def fetch_dataset_wrapper(ds_id: str) -> bytes:
                    return await fetch_fn(ds_id)
            
            # Load the dataset into the sandbox
            container = session_manager.container_for(session_id)
            
            # hybrid mode is handled inside load_pending_datasets
            # if ds is in local it is not fetched
            loaded_datasets = await load_pending_datasets(
                cfg=cfg,
                session_id=session_id,
                container=container,
                fetch_fn=fetch_dataset_wrapper,
                ds_ids=[dataset_id],
            )
            
            if loaded_datasets:
                dataset_info = loaded_datasets[0]
                path_in_container = dataset_info["path_in_container"]
                
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=f"Dataset '{dataset_id}' successfully loaded into sandbox at {path_in_container}",
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
                                content=f"Failed to load dataset '{dataset_id}' - no datasets were loaded",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
                
        except Exception as e:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Failed to load dataset '{dataset_id}': {str(e)}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )
    
    # Return a LangChain Tool by applying the decorator at factory time
    return tool(
        name_or_callable=name,
        description=description,
        args_schema=SelectDatasetArgs,
    )(_impl)


def make_export_datasets_tool(
    *,
    session_manager: SessionManager,
    session_key_fn: Callable[[], str] = _default_get_session_key,
    name: str = "export_datasets",
    description: str = (
        "Export a modified dataset from the container filepath to ./exports/modified_datasets/ "
        "with timestamp prefix. Use this to save processed or modified datasets "
        "from the sandbox to the host filesystem."
    ),
) -> Callable:
    """
    Create a tool for exporting files from the container's filepath.
    
    Parameters:
        session_manager: SessionManager instance to use for container operations
        session_key_fn: Function to get current session key
        name: Tool name
        description: Tool description
        
    Returns:
        LangChain tool function
    """
    
    class ExportDatasetArgs(BaseModel):
        container_path: Annotated[str, Field(description="Path to file inside container (e.g., '/to_export/<name>.parquet')")]
        runtime: ToolRuntime
        model_config = ConfigDict(arbitrary_types_allowed=True)
    
    async def _impl(
        container_path: Annotated[str, "Path to file inside container in the to_export/ directory (e.g., '/to_export/<name>.parquet')"], 
        runtime: ToolRuntime
    ) -> Command:
        """Export a file from container to host filesystem."""
        session_key = session_key_fn()
        
        # Get database session and thread_id from context (same as code_sandbox)
        try:
            from backend.graph.context import get_db_session, get_thread_id
            db_session = get_db_session()
            thread_id = get_thread_id()
        except ImportError:
            # Fallback if context module not available
            db_session = None
            thread_id = None
        
        # Call the session manager's export method with DB context for artifact persistence
        result = await session_manager.export_file(
            session_key, 
            container_path,
            db_session=db_session,
            thread_id=thread_id,
            tool_call_id=runtime.tool_call_id
        )
        
        if result["success"]:
            # Create structured artifact for the exported file
            structured_artifacts = [{
                "name": result['host_path'].split('/')[-1],  # Extract filename from host path
                "mime": "application/octet-stream",  # Generic binary file
                "url": result['download_url'],
                "size": 0  # Size not available from export result
            }]
            
            tool_msg = ToolMessage(
                content=(
                    f"Successfully exported dataset:\n"
                    f"  Container path: {container_path}\n"
                    f"  Host path: {result['host_path']}"
                ),
                artifact=structured_artifacts,  # Add artifact field like code_exec does
                tool_call_id=runtime.tool_call_id,
            )
        else:
            tool_msg = ToolMessage(
                content=f"Failed to export dataset: {result['error']}",
                tool_call_id=runtime.tool_call_id,
            )
        
        return Command(update={"messages": [tool_msg]})
    
    # Return a LangChain Tool by applying the decorator at factory time
    return tool(
        name_or_callable=name,
        description=description,
        args_schema=ExportDatasetArgs,
    )(_impl)

def make_list_datasets_tool(
    *,
    session_manager: SessionManager,
    session_key_fn: Callable[[], str] = _default_get_session_key,
    name: str = "list_datasets",
    description: str = (
        "List all datasets available in the sandbox. "
        "In API mode: lists datasets loaded in /data. "
        "In LOCAL_RO mode: lists statically mounted files in /data. "
        "In HYBRID mode: lists both local mounted files and API-loaded datasets in /data."
    ),
) -> Callable:
    """
    Factory that returns a LangChain Tool for listing datasets in the sandbox.
    
    Args:
        session_manager: SessionManager instance to use for container operations
        session_key_fn: Function to get current session key
        name: Tool name
        description: Tool description
        
    Returns:
        LangChain tool function
    """
    
    class ListDatasetsArgs(BaseModel):
        runtime: ToolRuntime
        model_config = ConfigDict(arbitrary_types_allowed=True)
    
    async def _impl(
        runtime: ToolRuntime
    ) -> Command:
        """List all datasets in the sandbox."""
        import os
        
        # Get dataset access mode from environment
        dataset_access = os.getenv("DATASET_ACCESS", "NONE")
        
        session_key = session_key_fn()
        
        # Start the session if not already started
        session_manager.start(session_key)
        
        # Determine the path to list based on dataset access mode
        if dataset_access == "API":
            # API mode: list files in /data
            list_path = "/data"
            mode_description = "API mode (loaded datasets)"
        elif dataset_access == "LOCAL_RO":
            # LOCAL_RO mode: list files in /data
            list_path = "/data"
            mode_description = "LOCAL_RO mode (statically mounted files)"
        elif dataset_access == "HYBRID":
            # HYBRID mode: list files in both /data (API) and /heavy_data (local)
            list_path = "/data"
            mode_description = "HYBRID mode (local + API datasets)"
        else:
            # NONE mode: no datasets available
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="No datasets available - sandbox is in NONE mode",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )
        
        # Execute code to list files in the appropriate directory
        if dataset_access == "HYBRID":
            # HYBRID mode: list both /data and /heavy_data
            list_code = f"""
import os
import json
from pathlib import Path

files = []

# List /data (API datasets)
data_path = "/data"
if os.path.exists(data_path):
    for item in os.listdir(data_path):
        item_path = os.path.join(data_path, item)
        if os.path.isfile(item_path):
            stat = os.stat(item_path)
            files.append({{
                "name": item,
                "path": item_path,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "source": "API"
            }})

# List /heavy_data (local datasets)
heavy_data_path = "/heavy_data"
if os.path.exists(heavy_data_path):
    for item in os.listdir(heavy_data_path):
        item_path = os.path.join(heavy_data_path, item)
        if os.path.isfile(item_path):
            stat = os.stat(item_path)
            files.append({{
                "name": item,
                "path": item_path,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "source": "Local"
            }})

result = {{
    "mode": "{mode_description}",
    "path": "/data and /heavy_data",
    "files": files,
    "count": len(files)
}}

print(json.dumps(result, indent=2))
"""
        else:
            # Other modes: list single directory
            list_code = f"""
import os
import json
from pathlib import Path

list_path = "{list_path}"
files = []

if os.path.exists(list_path):
    for item in os.listdir(list_path):
        item_path = os.path.join(list_path, item)
        if os.path.isfile(item_path):
            stat = os.stat(item_path)
            files.append({{
                "name": item,
                "path": item_path,
                "size": stat.st_size,
                "modified": stat.st_mtime
            }})
        elif os.path.isdir(item_path):
            files.append({{
                "name": item + "/",
                "path": item_path,
                "type": "directory"
            }})
else:
    files = []

result = {{
    "mode": "{mode_description}",
    "path": list_path,
    "files": files,
    "count": len([f for f in files if not f.get("type") == "directory"])
}}

print(json.dumps(result, indent=2))
"""
        
        # Execute the listing code
        result = await session_manager.exec(
            session_key, 
            list_code, 
            timeout=10,
            db_session=None,  # Not needed for listing
            thread_id=None,
        )
        
        if result.get("error"):
            content = f"Error listing datasets: {result['error']}"
        else:
            stdout = result.get("stdout", "")
            # Parse the JSON output
            try:
                result_data = json.loads(stdout)
                content = f"Datasets in {mode_description}:\n\n{json.dumps(result_data, indent=2)}"
            except json.JSONDecodeError as e:
                content = f"Error parsing JSON output: {e}\nRaw output: {stdout}"
        
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    
    # Return a LangChain Tool by applying the decorator at factory time
    return tool(
        name_or_callable=name,
        description=description,
        args_schema=ListDatasetsArgs,
    )(_impl)
