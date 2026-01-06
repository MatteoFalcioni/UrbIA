"""
Test loading datasets into the sandbox using SandboxExecutor directly.

This test verifies that we can load files both from the API and from S3
into the sandbox using the same approach as the load_dataset_tool.

It also verifies that loading more than one dataset in the same session works.
"""

import os
import base64
import json
import pytest
import time
from dotenv import load_dotenv
from backend.modal_runtime.executor import SandboxExecutor
from backend.opendata_api.init_client import client
from backend.opendata_api.helpers import get_dataset_bytes

load_dotenv()


# ===== helpers =====
def check_modal_tokens():
    """Check that Modal tokens are configured."""
    if not (os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET")):
        raise ValueError("Modal tokens not configured")


def check_s3_bucket():
    """Check that S3 bucket is configured."""
    if not os.getenv("S3_BUCKET") or not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY") or not os.getenv("AWS_REGION"):
        raise ValueError("S3 bucket not configured")


def load_dataset_bytes_to_sandbox(executor: SandboxExecutor, dataset_id: str, data_bytes: bytes) -> dict:
    """Load dataset bytes into sandbox (replicates load_dataset_tool logic)."""
    data_b64 = base64.b64encode(data_bytes).decode("utf-8")
    
    write_code = f"""
import base64
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
rel_path = str(path)
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
    result = executor.execute(write_code)
    stdout = result.get("stdout", "").strip()
    stderr = result.get("stderr", "")
    
    if stderr:
        raise RuntimeError(f"Failed to write dataset: {stderr}")
    
    return json.loads(stdout)


def list_loaded_datasets(executor: SandboxExecutor) -> list:
    """List datasets in sandbox (replicates list_loaded_datasets_tool logic)."""
    list_code = """
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
    stdout = result.get("stdout", "").strip()
    return json.loads(stdout) if stdout else []

# --- fixtures ---
@pytest.fixture(scope="module")
def test_session_id():
    """Create a single test session ID shared across all tests in this module."""
    session_id = "test-load-to-sandbox-session"
    yield session_id
    # Cleanup: terminate executor if created
    print(f"\n🧹 Cleaning up test session: {session_id}")
    from backend.graph.tools.sandbox_tools import terminate_session_executor, _executor_cache
    terminate_session_executor(session_id)
    _executor_cache.pop(session_id, None)


@pytest.fixture(scope="module")
def test_executor(test_session_id):
    """Create a single executor shared across all tests in this module."""
    check_modal_tokens()  # Check before creating executor
    executor = SandboxExecutor(session_id=test_session_id)
    yield executor
    executor.terminate()

# --- actual tests ---
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_load_dataset_from_api(test_executor, test_session_id):
    """Test that we can load a dataset from the API into the sandbox."""
    executor = test_executor
    session_id = test_session_id
    print(f"Session ID 1: {session_id}")

    dataset_id = "temperature_bologna"
    
    # Download from API
    print(f"Starting download of dataset: {dataset_id}")
    data_bytes = await get_dataset_bytes(client=client, dataset_id=dataset_id)
    print(f"Download completed. Size: {len(data_bytes)} bytes")
    
    # Load into sandbox using helper function (same logic as tool)
    res = load_dataset_bytes_to_sandbox(executor, dataset_id, data_bytes)
    
    assert res["dataset_id"] == dataset_id
    assert res["rel_path"].endswith(f"datasets/{dataset_id}.parquet")
    print(f"Loaded dataset from API successfully: {res}")
    
    # Verify we can access the dataset in code
    code = f"""
import pandas as pd
df = pd.read_parquet('{res["rel_path"]}')
print(f"Loaded dataset with shape: {{df.shape}}")
print(f"Columns: {{list(df.columns)}}")
print(f"First few rows:")
print(df.head())
"""
    result = executor.execute(code)
    
    assert "stdout" in result
    assert "shape" in result["stdout"].lower() or "loaded dataset" in result["stdout"].lower()
    assert not result.get("stderr") or result["stderr"] == ""
    
    print("✅ Loaded dataset from API and accessed from sandbox successfully")

@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_load_dataset_from_s3(test_executor, test_session_id):
    """Test that we can load a dataset from S3 into the sandbox."""
    check_s3_bucket()

    executor = test_executor
    session_id = test_session_id
    print(f"Session ID 2: {session_id}")
    
    dataset_id = "temperature_bologna"
    
    # Download from S3
    print(f"Downloading dataset from S3: {dataset_id}")
    import boto3
    from botocore.client import Config
    region = os.getenv("AWS_REGION", "eu-central-1")
    s3 = boto3.client(
        "s3",
        region_name=region,
        config=Config(signature_version='s3v4')
    )
    input_bucket = os.getenv("S3_BUCKET")
    s3_key = f"input/datasets/{dataset_id}.parquet"
    s3.head_object(Bucket=input_bucket, Key=s3_key)
    data_bytes = s3.get_object(Bucket=input_bucket, Key=s3_key)["Body"].read()
    print(f"Download completed. Size: {len(data_bytes)} bytes")

    # Load into sandbox using helper function (same logic as tool)
    res = load_dataset_bytes_to_sandbox(executor, dataset_id, data_bytes)
    
    assert res["dataset_id"] == dataset_id
    assert res["rel_path"].endswith(f"datasets/{dataset_id}.parquet")
    print(f"Loaded dataset from S3 successfully: {res}")

    # Verify we can access the dataset in code
    code = f"""
import pandas as pd
df = pd.read_parquet('{res["rel_path"]}')
print(f"Dataset shape: {{df.shape}}")
print(f"Memory usage: {{df.memory_usage(deep=True).sum() / 1024**2:.2f}} MB")
"""
    result = executor.execute(code)
    
    assert "Dataset shape" in result["stdout"]
    assert not result.get("stderr") or result["stderr"] == ""
    print("✅ Loaded dataset from S3 and accessed from sandbox successfully")

@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_load_multiple_datasets_in_same_session(test_executor, test_session_id):
    """Test that we can load multiple datasets into the sandbox in the same session."""
    executor = test_executor
    session_id = test_session_id
    print(f"Session ID 3: {session_id}")

    # Load first dataset from API
    dataset_id1 = "temperature_bologna"
    print(f"Loading dataset 1: {dataset_id1}")
    bytes1 = await get_dataset_bytes(client=client, dataset_id=dataset_id1)
    res1 = load_dataset_bytes_to_sandbox(executor, dataset_id1, bytes1)
    
    assert res1["dataset_id"] == dataset_id1
    assert res1["rel_path"].endswith(f"datasets/{dataset_id1}.parquet")
    print(f"✅ Dataset 1 loaded: {res1}")

    # Load second dataset from API
    dataset_id2 = "precipitazioni_bologna"
    print(f"Loading dataset 2: {dataset_id2}")
    bytes2 = await get_dataset_bytes(client=client, dataset_id=dataset_id2)
    res2 = load_dataset_bytes_to_sandbox(executor, dataset_id2, bytes2)
    
    assert res2["dataset_id"] == dataset_id2
    assert res2["rel_path"].endswith(f"datasets/{dataset_id2}.parquet")
    print(f"✅ Dataset 2 loaded: {res2}")

    # List datasets using helper function
    datasets = list_loaded_datasets(executor)
    print(f"Datasets in sandbox: {datasets}")
    assert dataset_id1 in datasets
    assert dataset_id2 in datasets

    # Verify both datasets are accessible in code
    code = f"""
import pandas as pd
df1 = pd.read_parquet('{res1["rel_path"]}')
df2 = pd.read_parquet('{res2["rel_path"]}')
print(f"Dataset 1 shape: {{df1.shape}}")
print(f"Dataset 2 shape: {{df2.shape}}")
print("Both datasets loaded successfully!")
"""

    result = executor.execute(code)
    assert "stdout" in result
    assert "shape" in result["stdout"].lower()
    assert "both datasets loaded successfully" in result["stdout"].lower()
    assert not result.get("stderr") or result["stderr"] == ""
    print("✅ Both datasets accessible from sandbox successfully")