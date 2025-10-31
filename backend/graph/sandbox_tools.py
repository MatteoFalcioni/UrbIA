"""Dataset loading tools with custom logic for heavy dataset handling and hybrid mode support."""

from __future__ import annotations

import os
import json
import base64
from typing import Dict
from typing_extensions import Annotated

from langchain_core.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

import modal

from backend.opendata_api.helpers import is_dataset_too_heavy, get_dataset_bytes  # change heavy detection this to be more reliable
from backend.opendata_api.init_client import client
from backend.modal_runtime.executor import SandboxExecutor
from backend.graph.context import get_thread_id

# Lookup deployed Modal functions (using from_name)
def _get_modal_function(name: str):
    """Get a deployed Modal function by name."""
    try:
        return modal.Function.from_name("lg-urban-executor", name)
    except Exception:
        # Fallback to import for local development
        raise Exception(f"Modal function {name} not found")

# Session-based executor cache: one sandbox per session
_executor_cache: Dict[str, SandboxExecutor] = {}


def get_or_create_executor(session_id: str) -> SandboxExecutor:
    """Get existing executor for session or create new one."""
    if session_id not in _executor_cache:
        _executor_cache[session_id] = SandboxExecutor(session_id=session_id)
    return _executor_cache[session_id]


def terminate_session_executor(session_id: str) -> None:
    """Terminate and cleanup executor for a session."""
    if session_id in _executor_cache:
        executor = _executor_cache.pop(session_id)
        executor.terminate()


@tool(
    name_or_callable="execute_code",
    description="Use this to execute python code."
)
def execute_code_tool(code: Annotated[str, "The python code to execute."],
                 runtime: ToolRuntime) -> Command:
    """Use this to execute python code."""
    session_id = str(get_thread_id())
    executor = get_or_create_executor(session_id)
    result = executor.execute(code)

    # wrap the result in a ToolMessage
    return Command(update={"messages": [ToolMessage(
        content=json.dumps(result, ensure_ascii=False), 
        tool_call_id=runtime.tool_call_id
    )]})

@tool(
    name_or_callable="load_dataset",
    description="Use this to load a dataset into the sandbox environment. Returns the written path (relative to /workspace)."
)
async def load_dataset_tool(
    dataset_id: Annotated[str, "The dataset ID to load."],
    runtime: ToolRuntime
) -> Command:
    """
    Load a dataset into the Modal workspace:
    - If exists in S3 input bucket, download from there.
    - Else, if not too heavy, fetch from the OpenData API.
    Returns the written path (relative to /workspace).
    """
    import boto3

    s3 = boto3.client("s3")
    input_bucket = os.getenv("S3_BUCKET")
    if not input_bucket:
        return Command(update={"messages": [ToolMessage(
            content="Missing S3_BUCKET env var",
            tool_call_id=runtime.tool_call_id
        )]})

    # Try S3 first (input/datasets/{dataset_id}.parquet)
    data_bytes = None
    s3_key = f"input/datasets/{dataset_id}.parquet"
    try:
        s3.head_object(Bucket=input_bucket, Key=s3_key)
        data_bytes = s3.get_object(Bucket=input_bucket, Key=s3_key)["Body"].read()
    except Exception:
        # fetching from api: is it too heavy?
        too_heavy = await is_dataset_too_heavy(client=client, dataset_id=dataset_id)
        if too_heavy:
            return Command(update={"messages": [ToolMessage(
                content=f"Dataset '{dataset_id}' is too large to fetch from API.",
                tool_call_id=runtime.tool_call_id
            )]})
        # fetch from api
        data_bytes = await get_dataset_bytes(client=client, dataset_id=dataset_id)
    
    # write into modal workspace
    session_id = str(get_thread_id())
    write_dataset_bytes = _get_modal_function("write_dataset_bytes")
    summary = write_dataset_bytes.remote(
        dataset_id=dataset_id,
        data_b64=base64.b64encode(data_bytes).decode("utf-8"),
        session_id=session_id,
        ext='parquet',
        subdir="datasets",
    )    

    return Command(update={"messages": [ToolMessage(
        content=json.dumps(summary, ensure_ascii=False),
        tool_call_id=runtime.tool_call_id
    )]})

@tool(
    name_or_callable="list_datasets",
    description="Use this to list all datasets currently loaded in the sandbox."
)
def list_datasets_tool(runtime: ToolRuntime) -> Command:
    """Use this to list all datasets currently loaded in the sandbox."""
    session_id = str(get_thread_id())
    list_loaded_datasets = _get_modal_function("list_loaded_datasets")
    datasets = list_loaded_datasets.remote(session_id=session_id)
    return Command(update={"messages": [ToolMessage(content=json.dumps(datasets, ensure_ascii=False), tool_call_id=runtime.tool_call_id)]})

@tool(
    name_or_callable="export_dataset",
    description="Use this to export a dataset from the sandbox given its path."
)
def export_dataset_tool(dataset_path: Annotated[str, "The path of the dataset to export."],
                   runtime: ToolRuntime) -> Command:
    """Use this to export a dataset from the sandbox given its path."""
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return Command(update={"messages": [ToolMessage(
            content="Missing S3_BUCKET env var",
            tool_call_id=runtime.tool_call_id
        )]})
    session_id = str(get_thread_id())
    export_dataset = _get_modal_function("export_dataset")
    result = export_dataset.remote(dataset_path, bucket, session_id=session_id)
    return Command(update={"messages": [ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=runtime.tool_call_id)]})