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


from backend.opendata_api.init_client import client
from backend.modal_runtime.executor import SandboxExecutor
from backend.modal_runtime.session import session_base_dir
from backend.graph.context import get_thread_id

# ===== helpers =====

# Session-based executor cache: one sandbox per session
_executor_cache: Dict[str, SandboxExecutor] = {}


# ===== executor management =====
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


# ===== tools =====


# -----------------
# execute code tool
# -----------------
@tool(name_or_callable="execute_code", description="Use this to execute python code.")
def execute_code_tool(
    code: Annotated[str, "The python code to execute."], runtime: ToolRuntime
) -> Command:
    """Use this to execute python code."""
    thread_id = get_thread_id()
    if not thread_id:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Error: thread_id not set in context. Cannot execute code.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    session_id = str(thread_id)
    executor = get_or_create_executor(session_id)
    result = executor.execute(code)

    # take out artifacts from result and use artifact field of ToolMessage to return them
    artifacts = result.pop("artifacts")
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=runtime.tool_call_id,
                    artifact=artifacts,
                )
            ],
            "code_logs": [
                {
                    "input": code,
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                }
            ],
        }
    )


# -----------------
# load dataset tool
# -----------------
@tool(
    name_or_callable="load_dataset",
    description="Load a dataset by ID into the workspace. After loading, you can access it in code with the code execution tool at the path 'datasets/{dataset_id}.parquet' from the working directory.",
)
async def load_dataset_tool(
    dataset_id: Annotated[str, "The dataset ID to load."], runtime: ToolRuntime
) -> Command:
    """
    First, checks if the dataset was already loaded in the workspace.

    If not, then loads a dataset into the sandbox:
    - If the dataset exists in S3 input bucket, download from there.
    - Else, fetch it from the OpenData API AND THEN upload it to S3 (so it's faster next time).
    Returns the written path (relative to /workspace).
    """

    # get session id from thread id
    thread_id = get_thread_id()
    if not thread_id:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Error: thread_id not set in context. Cannot load dataset.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )
    session_id = str(thread_id)
    executor = get_or_create_executor(session_id)

    print(f"Checking if dataset {dataset_id} is already loaded...")

    # First, check if the dataset was already loaded in the workspace
    check_code = f"""
import os
import json
from pathlib import Path

dataset_path = Path('datasets/{dataset_id}.parquet')
exists = dataset_path.exists()
if exists:
    size_bytes = dataset_path.stat().st_size
    result = {{"exists": True, "path": str(dataset_path), "size_bytes": size_bytes}}
else:
    result = {{"exists": False, "path": str(dataset_path)}}
print(json.dumps(result))
"""
    check_result = executor.execute(check_code)
    stdout = check_result.get("stdout", "").strip()

    try:
        check_data = json.loads(stdout) if stdout else {}
        if check_data.get("exists", False):
            # Construct absolute path for consistency with newly loaded datasets
            base_dir = session_base_dir(session_id)
            abs_path = f"{base_dir}/datasets/{dataset_id}.parquet"
            size_bytes = check_data.get("size_bytes", 0)
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=json.dumps(
                                {
                                    "dataset_id": dataset_id,
                                    "path": abs_path,
                                    "rel_path": f"datasets/{dataset_id}.parquet",
                                    "size_bytes": size_bytes,
                                    "size_mb": round(size_bytes / (1024 * 1024), 3),
                                    "ext": "parquet",
                                    "note": f"Dataset '{dataset_id}' already loaded. In code, use: pd.read_parquet('datasets/{dataset_id}.parquet')",
                                }
                            ),
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )
    except (json.JSONDecodeError, KeyError) as e:
        # If check fails, continue to load anyway
        print(f"Check failed for dataset {dataset_id}; error: {e}. Continuing to load...")
        raise e

    # If not, load from S3 or API
    try:
        import boto3
        import tempfile
        import time
        from botocore.client import Config

        print(f"[LOAD_DATASET] Starting load for {dataset_id}")

        region = os.getenv("AWS_REGION", "eu-central-1")
        s3 = boto3.client(
            "s3", region_name=region, config=Config(signature_version="s3v4")
        )
        input_bucket = os.getenv("S3_BUCKET")
        if not input_bucket:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content="Error: Missing S3_BUCKET environment variable.",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

        # Try S3 first (input/datasets/{dataset_id}.parquet)
        data_bytes = None
        s3_key = f"input/datasets/{dataset_id}.parquet"
        
        # Create a temp file to store the dataset locally (avoids RAM spikes)
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_path = tmp_file.name
            print(f"Created temp file: {temp_path}")

        try:
            # Check S3
            try:
                print("Checking S3...")
                s3.head_object(Bucket=input_bucket, Key=s3_key)
                
                # Download from S3 to file
                print("Downloading from S3 to file...")
                s3.download_file(input_bucket, s3_key, temp_path)
                print("S3 download complete")
                
                with open(temp_path, "rb") as f:
                    data_bytes = f.read()
                print("Read bytes from file into RAM")
                
            except Exception:
                # Not in S3, download from API to file
                try:
                    print("Not in S3. Downloading from API to file...")
                    start_dl = time.time()
                    await client.export_to_file(
                        dataset_id=dataset_id, path=temp_path
                    )
                    dl_time = time.time() - start_dl
                    print(f"API Download complete in {dl_time:.2f}s")
                    
                    # Read bytes for sandbox injection
                    print("Reading file into RAM...")
                    with open(temp_path, "rb") as f:
                        data_bytes = f.read()
                    print(f"Read {len(data_bytes) / 1024 / 1024:.2f} MB into RAM")

                    if not data_bytes:
                         return Command(
                            update={
                                "messages": [
                                    ToolMessage(
                                        content=f"Error: Dataset '{dataset_id}' returned empty data.",
                                        tool_call_id=runtime.tool_call_id,
                                    )
                                ]
                            }
                        )
                except Exception as api_err:
                     print(f"API Error: {str(api_err)}")
                     return Command(
                        update={
                            "messages": [
                                ToolMessage(
                                    content=f"Error: Failed to fetch dataset '{dataset_id}' from API. Error: {str(api_err)}",
                                    tool_call_id=runtime.tool_call_id,
                                )
                            ]
                        }
                    )

                # After downloading from API, upload to S3 from file
                try:
                    print("Uploading to S3...")
                    s3.upload_file(
                        Filename=temp_path,
                        Bucket=input_bucket,
                        Key=s3_key,
                        ExtraArgs={"ContentType": "application/parquet"},
                    )
                    print("S3 Upload complete")
                except Exception as upload_err:
                    print(
                        f"Warning: Failed to upload dataset to S3: {upload_err}. Dataset is being loaded into workspace anyway..."
                    )
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            print("Cleanup complete")

        # Write dataset directly to sandbox using executor.execute()
        print("Encoding base64...")
        data_b64 = base64.b64encode(data_bytes).decode("utf-8")
        # Use repr() to safely pass the base64 string in the f-string
        write_code = f"""
import base64
import os
import json
from pathlib import Path

# Decode and write the dataset
data_b64 = {repr(data_b64)}
data = base64.b64decode(data_b64)
datasets_dir = Path('datasets')
datasets_dir.mkdir(exist_ok=True)
path = datasets_dir / '{dataset_id}.parquet'
path.write_bytes(data)

# Get metadata
size_bytes = len(data)
# rel_path is relative to workdir (datasets/{dataset_id}.parquet)
rel_path = str(path)
# path should be absolute - resolve() gives absolute path
abs_path = str(path.resolve())

result = {{
    "dataset_id": "{dataset_id}",
    "path": abs_path,
    "rel_path": rel_path,
    "size_bytes": size_bytes,
    "size_mb": round(size_bytes / (1024 * 1024), 3),
    "ext": "parquet"
}}

print(json.dumps(result))
"""
        write_result = executor.execute(write_code)

        stdout = write_result.get("stdout", "").strip()
        stderr = write_result.get("stderr", "")

        if stderr:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Error: Failed to write dataset to sandbox: {stderr}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

        try:
            result = json.loads(stdout) if stdout else {}
            if "error" in result:
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=f"Error: {result['error']}",
                                tool_call_id=runtime.tool_call_id,
                            )
                        ]
                    }
                )
        except json.JSONDecodeError:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Error: Failed to parse write result. Output: {stdout}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

        # Add a clear note about the path to use in code
        result["note"] = (
            f"Dataset loaded. In code, use: pd.read_parquet('{result['rel_path']}')"
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    except Exception as e:
        # Catch-all for any unexpected errors
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Error: Unexpected error loading dataset '{dataset_id}': {str(e)}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )


# -----------------
# list loaded datasets tool
# -----------------
@tool(
    name_or_callable="list_loaded_datasets",
    description="List datasets already loaded in the current workspace.",
)
def list_loaded_datasets_tool(runtime: ToolRuntime) -> Command:
    """
    Lists datasets already loaded in the current workspace.
    NOTE: we do not list S3 datasets because we don't want the model to get confused.
    BUT when we load, we check if datasets are present in S3 first and, if so, we download from there.
    """

    thread_id = get_thread_id()
    if not thread_id:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Error: thread_id not set in context. Cannot list datasets.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    try:
        session_id = str(thread_id)
        executor = get_or_create_executor(session_id)

        # List datasets directly in the sandbox
        list_code = """
import os
import json
from pathlib import Path

datasets_dir = Path('datasets')
if datasets_dir.exists():
    files = [f.stem for f in datasets_dir.glob('*.parquet')]
else:
    files = []

print(json.dumps(files))
"""
        result = executor.execute(list_code)

        # Parse JSON from stdout
        stdout = result.get("stdout", "").strip()
        stderr = result.get("stderr", "")

        if stderr:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Error: Failed to list loaded datasets: {stderr}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

        try:
            dataset_ids = json.loads(stdout) if stdout else []
        except json.JSONDecodeError:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=f"Error: Failed to parse dataset list. Output: {stdout}",
                            tool_call_id=runtime.tool_call_id,
                        )
                    ]
                }
            )

        # Return the result as a list of dataset_ids
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps(dataset_ids, ensure_ascii=False),
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
                        content=f"Error: Failed to list loaded datasets: {str(e)}",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )


# -----------------
# export dataset tool
# -----------------
@tool(
    name_or_callable="export_dataset",
    description="Use this to export a dataset from the sandbox given its path.",
)
def export_dataset_tool(
    dataset_path: Annotated[str, "The path of the dataset to export."],
    runtime: ToolRuntime,
) -> Command:
    """Exports a dataset from the sandbox to S3 by executing upload code inside the sandbox.

    This avoids Modal volume sync issues by reading the file directly from the sandbox
    filesystem where it was created, rather than trying to access it from a separate Modal function.
    """
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Missing S3_BUCKET env var",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    thread_id = get_thread_id()
    if not thread_id:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content="Error: thread_id not set in context. Cannot export dataset.",
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    session_id = str(thread_id)

    # Generate Python code to export from inside the sandbox
    # This avoids volume sync issues since the file is read from the same container that created it
    export_code = f"""
import hashlib
import mimetypes
import boto3
import json
from pathlib import Path
from botocore.client import Config

# Read the file
file_path = Path('{dataset_path}')
if not file_path.exists():
    result = {{"error": "File not found: {dataset_path}"}}
else:
    data = file_path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    size = len(data)
    
    # Upload to S3
    s3_key = f"output/datasets/{{sha256[:2]}}/{{sha256[2:4]}}/{{sha256}}"
    region = "eu-central-1"
    s3_client = boto3.client("s3", region_name=region, config=Config(signature_version='s3v4'))
    s3_client.put_object(
        Bucket="{bucket}",
        Key=s3_key,
        Body=data,
        ContentType=mime
    )
    
    result = {{
        "name": file_path.name,
        "path": str(file_path),
        "sha256": sha256,
        "mime": mime,
        "size": size,
        "s3_key": s3_key,
        "s3_url": f"s3://{bucket}/{{s3_key}}"
    }}

print(json.dumps(result))
"""

    # Execute the export code in the sandbox
    executor = get_or_create_executor(session_id)
    result = executor.execute(export_code)

    # The result will be in stdout as JSON
    stdout = result.get("stdout", "").strip()
    stderr = result.get("stderr", "")

    # Parse the JSON result
    try:
        export_result = json.loads(stdout) if stdout else {}
    except json.JSONDecodeError:
        export_result = {"error": "Failed to parse export result"}

    # If there's an error, return it as content
    if "error" in export_result or (stderr and not stdout):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=json.dumps(
                            export_result
                            if "error" in export_result
                            else {"error": f"Export failed: {stderr}"}
                        ),
                        tool_call_id=runtime.tool_call_id,
                    )
                ]
            }
        )

    # Return as artifact so frontend can display download link
    # Format matches what the code execution tool returns
    artifacts = [
        export_result
    ]  # List of artifact dicts with s3_key, name, mime, size, sha256

    # Return JSON with success message and export details
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=json.dumps(
                        {
                            "success": True,
                            "message": f"Dataset exported successfully: {export_result.get('name', 'unknown')}",
                            **export_result,  # Include all export details (name, path, sha256, mime, size, s3_key, s3_url)
                        },
                        ensure_ascii=False,
                    ),
                    tool_call_id=runtime.tool_call_id,
                    artifact=artifacts,  # for frontend to display download link, artifacts go here, not in content
                )
            ]
        }
    )
