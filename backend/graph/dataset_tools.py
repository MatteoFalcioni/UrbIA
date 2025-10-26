"""Dataset loading tools with custom logic for heavy dataset handling and hybrid mode support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Callable, Any

from typing_extensions import Annotated

from langchain_core.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

# Import the shared session manager from tools.py
from backend.graph.tools import get_session_manager
from backend.tool_factory.make_tools import (
    make_export_datasets_tool,
    make_list_datasets_tool
)
from backend.opendata_api.helpers import get_dataset_bytes, is_dataset_too_heavy
from backend.opendata_api.init_client import client


def _tar_single_file_bytes(filename: str, data: bytes, mode: int = 0o644) -> bytes:
    """Create a tar archive containing a single file."""
    import tarfile
    import io
    
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode='w') as tar:
        tarinfo = tarfile.TarInfo(name=filename)
        tarinfo.size = len(data)
        tarinfo.mode = mode
        tar.addfile(tarinfo, io.BytesIO(data))
    
    tar_stream.seek(0)
    return tar_stream.read()


def put_bytes(container, container_path: str, data: bytes, *, mode: int = 0o644) -> None:
    """
    Write `data` to `container_path` inside the container by streaming a single-file
    tar to Docker's put_archive. Overwrites any existing file.
    
    Falls back to base64 encoding if put_archive fails.
    
    Parameters:
        container: Docker container object
        container_path: Absolute path to the destination file in the container
        data: File content (bytes)
        mode: File mode for the created file (default 0o644)
    """
    from pathlib import Path
    import base64
    
    if not container_path or container_path.endswith("/"):
        raise ValueError("container_path must be a file path, not a directory")
    
    parent = str(Path(container_path).parent)
    name_in_tar = str(Path(container_path).name)
    
    # Ensure parent directory exists
    exec_result = container.exec_run(
        ["python3", "-c", f"import os; os.makedirs('{parent}', exist_ok=True)"],
        user="root"
    )
    if exec_result.exit_code != 0:
        print(f"Warning: mkdir failed: {exec_result.output.decode()}")
    
    # Set permissions on parent
    exec_result = container.exec_run(
        ["chmod", "777", parent],
        user="root"
    )
    
    # Create tar with just the filename (no directory structure)
    tar_bytes = _tar_single_file_bytes(name_in_tar, data, mode=mode)
    
    # Try put_archive first
    try:
        ok = container.put_archive(path=parent, data=tar_bytes)
        
        # Verify the file was actually written
        exec_result = container.exec_run(["ls", "-la", container_path], user="root")
        
        if exec_result.exit_code == 0:
            print(f"Successfully wrote {container_path} using put_archive")
            print(f"File info: {exec_result.output.decode()}")
            return
        else:
            print(f"File not found after put_archive, trying base64 fallback...")
    except Exception as e:
        print(f"put_archive exception: {e}, trying base64 fallback...")
    
    # Fallback to base64 method with chunking (for files that exceed arg list limit)
    print(f"Using base64 chunked fallback to write {container_path} ({len(data)} bytes)")
    data_b64 = base64.b64encode(data).decode('ascii')
    
    # Use 2000 base64 characters per chunk to avoid "argument list too long" error
    chunk_size = 2000
    chunks = [data_b64[i:i+chunk_size] for i in range(0, len(data_b64), chunk_size)]
    
    print(f"Writing {len(chunks)} chunks...")
    
    # Create empty file first
    exec_result = container.exec_run(
        ["sh", "-c", f"cat /dev/null > {container_path}"],
        user="root"
    )
    if exec_result.exit_code != 0:
        raise RuntimeError(f"Failed to create empty file {container_path}: {exec_result.output.decode()}")
    
    # Write chunks one by one
    for i, chunk in enumerate(chunks):
        exec_result = container.exec_run(
            ["sh", "-c", f"echo -n '{chunk}' | base64 -d >> {container_path}"],
            user="root"
        )
        if exec_result.exit_code != 0:
            raise RuntimeError(f"Failed to write chunk {i+1}/{len(chunks)}: {exec_result.output.decode()}")
        
        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{len(chunks)} chunks written")
    
    print(f"All {len(chunks)} chunks written successfully")
    
    # Final verification
    exec_result = container.exec_run(["ls", "-lh", container_path], user="root")
    if exec_result.exit_code != 0:
        raise RuntimeError(f"File verification failed after chunked write: {exec_result.output.decode()}")
    
    print(f"File verified: {exec_result.output.decode()}")


def get_session_key():
    """Get session key from the conversation context."""
    try:
        # Try to get thread_id from context (set by API layer)
        from backend.graph.context import get_thread_id
        tid = get_thread_id()
        return str(tid) if tid else "default"
    except Exception:
        return "default"


# Expose session_manager for backward compatibility with sit_tools
session_manager = get_session_manager()


# Create dataset management tools using the shared session manager
export_datasets_tool = make_export_datasets_tool(
    session_manager=get_session_manager(),
    session_key_fn=get_session_key
)

list_datasets_tool = make_list_datasets_tool(
    name="list_loaded_datasets",
    description="List already loaded datasets available in sandbox",
    session_manager=get_session_manager(),
    session_key_fn=get_session_key
)


async def load_dataset_with_size_check(
    *,
    session_id: str,
    dataset_id: str,
    container,
) -> Dict[str, Any]:
    """
    Load a dataset into the sandbox with HYBRID mode support.
    
    HYBRID mode logic:
    1. First checks if dataset exists in /heavy_data (mounted read-only in sandbox)
    2. If found locally, uses it directly at /heavy_data/{dataset_id}.parquet
    3. If not found locally, downloads from API to /data/{dataset_id}.parquet (with size check)
    
    Args:
        session_id: The current session identifier
        dataset_id: The dataset ID to load
        container: Docker container handle
        
    Returns:
        Dict with:
            - "id": dataset id
            - "path_in_container": absolute path to the dataset file inside the container
            - "too_heavy": bool, True if dataset was not loaded due to size
            - "source": "local" or "api"
            
    Raises:
        Exception: If loading fails for any reason other than being too heavy
    """
    from backend.config import DATASET_ACCESS
    
    # HYBRID mode: Check if dataset exists in /heavy_data (mounted in sandbox)
    if DATASET_ACCESS == "HYBRID":
        # Execute code in the sandbox container to check if file exists at /heavy_data
        check_code = f"""
import os
heavy_data_path = "/heavy_data"
file_path = os.path.join(heavy_data_path, "{dataset_id}.parquet")
exists = os.path.exists(file_path)
print(f"File exists in container: {{exists}}")
if exists:
    stat = os.stat(file_path)
    print(f"File size: {{stat.st_size}} bytes")
"""
        try:
            result = await session_manager.exec(
                session_id, 
                check_code, 
                timeout=10,
                db_session=None,  # Not needed for this check
                thread_id=None,
            )
            
            # Check if the result indicates the file exists
            if "File exists in container: True" in result.get('stdout', ''):
                # Dataset exists locally at /heavy_data, use it directly (no copy needed!)
                path_in_container = f"/heavy_data/{dataset_id}.parquet"
                print(f"Dataset {dataset_id} found in local /heavy_data at {path_in_container}")
                
                return {
                    "id": dataset_id,
                    "path_in_container": path_in_container,
                    "too_heavy": False,
                    "source": "local",
                }
        except Exception as e:
            print(f"Warning: Error checking local file for {dataset_id}: {e}")
            print(f"Falling back to API download...")
            # Fall through to API download below
        
        print(f"Dataset {dataset_id} not found in /heavy_data, will download from API")
    
    # API mode or HYBRID fallback: Download from Bologna OpenData API
    try:
        # Check if dataset is too heavy before loading (2MB threshold by default)
        is_heavy = await is_dataset_too_heavy(client, dataset_id, threshold=2_000_000)
        if is_heavy:
            print(f"Dataset {dataset_id} is too heavy and was not loaded")
            return {
                "id": dataset_id,
                "path_in_container": None,
                "too_heavy": True,
                "source": "api",
            }
    except Exception as e:
        # If size check fails, log but continue with normal loading
        print(f"Warning: Could not check size for {dataset_id}: {e}")
    
    # Download the dataset as parquet bytes
    try:
        parquet_bytes = await get_dataset_bytes(client, dataset_id)
    except Exception as e:
        raise Exception(f"Failed to download dataset {dataset_id}: {e}")
    
    # Stage the dataset into the container at /data/{dataset_id}.parquet
    try:
        path_in_container = f"/data/{dataset_id}.parquet"
        
        # Use robust put_bytes function (handles tar extraction issues)
        put_bytes(container, path_in_container, parquet_bytes)
        
        return {
            "id": dataset_id,
            "path_in_container": path_in_container,
            "too_heavy": False,
            "source": "api",
        }
        
    except Exception as e:
        raise Exception(f"Failed to stage dataset {dataset_id} into container: {e}")


@tool(
    name_or_callable="select_dataset",
    description="Select and load a dataset from Bologna OpenData into the sandbox for analysis.",
)
async def select_dataset_tool(
    dataset_id: Annotated[str, "The dataset ID from Bologna OpenData"], 
    runtime: ToolRuntime
) -> Command:
    """
    Select and load a dataset into the sandbox.
    
    This tool:
    - Checks if the dataset is too large (>2MB) before loading
    - Downloads the dataset as parquet
    - Makes it available at /data/{dataset_id}.parquet in the sandbox
    """
    session_id = get_session_key()
    
    try:
        # Start the session if not already started
        session_manager.start(session_id)
        
        # Get the container for this session
        container = session_manager.container_for(session_id)
        
        # Load the dataset with size checking
        result = await load_dataset_with_size_check(
            session_id=session_id,
            dataset_id=dataset_id,
            container=container,
        )
        
        if result.get("too_heavy"):
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Dataset '{dataset_id}' is too large (>2MB) and was not loaded. Consider using a smaller subset or filtering the data via the API first.",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )
        
        path_in_container = result["path_in_container"]
        source = result.get("source", "unknown")
        source_msg = " (from local storage)" if source == "local" else " (downloaded from API)" if source == "api" else ""
        
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Dataset '{dataset_id}' successfully loaded{source_msg} into sandbox at {path_in_container}. You can now read it with pandas: pd.read_parquet('{path_in_container}')",
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

