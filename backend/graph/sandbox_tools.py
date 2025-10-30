"""Dataset loading tools with custom logic for heavy dataset handling and hybrid mode support."""

from __future__ import annotations

import os
import json
import base64
from typing_extensions import Annotated

from langchain_core.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

from backend.modal_runtime.functions import list_loaded_datasets, export_dataset, write_dataset_bytes
from backend.opendata_api.helpers import is_dataset_too_heavy, get_dataset_bytes  # change heavy detection this to be more reliable
from backend.opendata_api.init_client import client
from backend.modal_runtime.executor import SandboxExecutor
from backend.graph.context import get_thread_id


@tool(
    name_or_callable="execute_code",
    description="Use this to execute python code."
)
def execute_code_tool(code: Annotated[str, "The python code to execute."],
                 runtime: ToolRuntime) -> Command:
    """Use this to execute python code."""
    session_id = get_thread_id()
    result = SandboxExecutor(session_id=session_id).execute(code)

    # wrap the result in a ToolMessage
    return Command(update={"messages": [ToolMessage(content=result, tool_call_id=runtime.tool_call_id)]})

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
    session_id = get_thread_id()
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
    session_id = get_thread_id()
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
    session_id = get_thread_id()
    result = export_dataset.remote(dataset_path, bucket, session_id=session_id)
    return Command(update={"messages": [ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=runtime.tool_call_id)]})