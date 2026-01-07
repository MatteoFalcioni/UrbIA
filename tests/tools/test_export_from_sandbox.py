"""
Test exporting datasets from inside the sandbox (avoiding volume sync issues).

This test verifies that we can export files by executing export code
directly inside the sandbox, rather than trying to read files from
a separate Modal function.
"""
import os
import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.mark.timeout(300)
def test_export_dataset_from_inside_sandbox():
    """Test that we can export a dataset by running export code inside the sandbox."""
    # Require Modal tokens to run this real integration test
    if not (os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET")):
        pytest.skip("Modal tokens not configured; skipping real Modal integration test")

    from backend.modal_runtime.executor import SandboxExecutor
    import json

    session_id = "test-export-session"
    executor = SandboxExecutor(session_id=session_id)
    
    try:
        # Step 1: Create a test dataset file in the sandbox
        create_code = """
import pandas as pd
from pathlib import Path

# Create datasets directory
datasets_dir = Path('datasets')
datasets_dir.mkdir(exist_ok=True)

# Create a simple test dataset
df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [4, 5, 6]
})

# Save as parquet
file_path = datasets_dir / 'test_export.parquet'
df.to_parquet(file_path)

print(f"Created dataset at: {file_path}")
print(f"File exists: {file_path.exists()}")
"""
        
        result1 = executor.execute(create_code)
        print("Create result:", result1)
        assert result1["stderr"] == "" or "FutureWarning" in result1["stderr"]
        assert "Created dataset at:" in result1["stdout"]
        assert "File exists: True" in result1["stdout"]
        
        # Step 2: Export the dataset from inside the sandbox
        # We'll test with and without S3 credentials
        have_s3 = bool(
            os.getenv("AWS_ACCESS_KEY_ID") and 
            os.getenv("AWS_SECRET_ACCESS_KEY") and 
            os.getenv("S3_BUCKET")
        )
        
        if have_s3:
            # Real S3 export
            bucket = os.environ["S3_BUCKET"]
            export_code = f"""
import hashlib
import mimetypes
import boto3
import json
from pathlib import Path
from botocore.client import Config

# Read the file
file_path = Path('datasets/test_export.parquet')
if not file_path.exists():
    result = {{"error": "File not found: datasets/test_export.parquet"}}
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
            result2 = executor.execute(export_code)
            print("Export result:", result2)
            
            # Parse the JSON output
            output = result2["stdout"].strip()
            export_result = json.loads(output)
            
            assert "error" not in export_result
            assert export_result["name"] == "test_export.parquet"
            assert "sha256" in export_result
            assert "s3_key" in export_result
            assert "s3_url" in export_result
            assert export_result["s3_url"].startswith(f"s3://{bucket}/")
            print(f"✓ Successfully exported to S3: {export_result['s3_url']}")
        
        else:
            # Mock S3 export - just verify file can be read and hashed
            export_code = """
import hashlib
import mimetypes
import json
from pathlib import Path

# Read the file
file_path = Path('datasets/test_export.parquet')
if not file_path.exists():
    result = {"error": "File not found: datasets/test_export.parquet"}
else:
    data = file_path.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    size = len(data)
    
    result = {
        "name": file_path.name,
        "path": str(file_path),
        "sha256": sha256,
        "mime": mime,
        "size": size,
        "mock": True
    }

print(json.dumps(result))
"""
            result2 = executor.execute(export_code)
            print("Mock export result:", result2)
            
            # Parse the JSON output
            output = result2["stdout"].strip()
            export_result = json.loads(output)
            
            assert "error" not in export_result
            assert export_result["name"] == "test_export.parquet"
            assert "sha256" in export_result
            assert export_result["mock"] is True
            print(f"✓ Successfully read and hashed file (mock S3): {export_result['sha256']}")
    
    finally:
        # Clean up
        executor.terminate()


@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_export_dataset_tool_integration():
    """Test the actual export_dataset_tool works with the sandbox."""
    # Require Modal tokens to run this real integration test
    if not (os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET")):
        pytest.skip("Modal tokens not configured; skipping real Modal integration test")
    
    from backend.graph.tools.sandbox_tools import export_dataset_tool, get_or_create_executor, terminate_session_executor
    from backend.graph.context import set_thread_id
    from unittest.mock import Mock
    import json
    import uuid
    
    # Create a test session - use a proper UUID as thread_id
    thread_uuid = uuid.uuid4()
    set_thread_id(thread_uuid)
    session_id = str(thread_uuid)  # session_id matches thread_id
    
    try:
        # Step 1: Create a dataset in the sandbox
        executor = get_or_create_executor(session_id)
        create_code = """
import pandas as pd
from pathlib import Path

datasets_dir = Path('datasets')
datasets_dir.mkdir(exist_ok=True)

df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
df.to_parquet('datasets/tool_test.parquet')
print("Dataset created")
"""
        result = executor.execute(create_code)
        assert "Dataset created" in result["stdout"]
        print(f"✓ Dataset created in session {session_id}")
        
        # Step 2: Use the export_dataset_tool
        mock_runtime = Mock()
        mock_runtime.tool_call_id = "test-call-123"
        
        # Call the async tool via its coroutine
        command = await export_dataset_tool.coroutine("datasets/tool_test.parquet", mock_runtime)
        
        # Extract the result
        tool_message = command.update["messages"][0]
        
        # The export result is in the artifact field, not content
        # Content just has a success message
        assert "Dataset exported successfully" in tool_message.content
        assert tool_message.artifact is not None
        assert len(tool_message.artifact) > 0
        
        export_result = tool_message.artifact[0]
        print(f"Export result: {export_result}")
        
        # Verify success
        assert "error" not in export_result
        assert export_result["name"] == "tool_test.parquet"
        assert "sha256" in export_result
        assert "s3_url" in export_result
        print(f"✓ Tool successfully exported to: {export_result['s3_url']}")
        
    finally:
        # Clean up
        terminate_session_executor(session_id)


if __name__ == "__main__":
    test_export_dataset_from_inside_sandbox()
    test_export_dataset_tool_integration()

