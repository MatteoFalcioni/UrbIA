"""
Real integration tests for sandbox tools without requiring Postgres.

These tests actually:
- Execute Python code in Modal sandboxes
- Call deployed Modal functions
- Upload/download from S3
- Test the full stack

Requirements:
- MODAL_TOKEN_ID, MODAL_TOKEN_SECRET (for Modal functions)
- S3_BUCKET, AWS credentials (for S3 operations)
- Modal functions must be deployed: `modal deploy backend/modal_runtime/functions.py`
"""

import os
import uuid
import base64
import json
import pytest
import boto3
import pandas as pd
import io
from unittest.mock import Mock
from dotenv import load_dotenv

from backend.graph.context import set_thread_id, get_thread_id
from backend.graph.sandbox_tools import (
    execute_code_tool,
    load_dataset_tool,
    list_datasets_tool,
    export_dataset_tool,
    terminate_session_executor,
    _executor_cache,
)

load_dotenv()


@pytest.fixture(scope="module")
def test_session_id():
    """Create a single test session ID shared across all tests in this module."""
    # Use a fixed UUID for all tests so they share the same Modal workspace
    test_uuid = uuid.UUID("00000000-0000-0000-0000-000000000001")
    set_thread_id(test_uuid)
    session_id = str(test_uuid)
    yield session_id
    # Cleanup: terminate executor if created
    print(f"\n🧹 Cleaning up test session: {session_id}")
    terminate_session_executor(session_id)
    # Also clear from cache just in case
    _executor_cache.pop(session_id, None)


@pytest.fixture
def mock_runtime():
    """Create a mock ToolRuntime object (only thing we mock)."""
    runtime = Mock()
    runtime.tool_call_id = f"test-tool-{uuid.uuid4()}"
    return runtime


@pytest.fixture
def test_dataset_bytes():
    """Create test dataset bytes for mocking."""
    df = pd.DataFrame({
        "city": ["Milano", "Roma", "Torino", "Napoli"],
        "population": [1352000, 2873000, 875000, 959000],
        "region": ["Lombardia", "Lazio", "Piemonte", "Campania"]
    })
    buffer = io.BytesIO()
    df.to_parquet(buffer)
    return buffer.getvalue()


def _have_modal_tokens() -> bool:
    """Check if Modal tokens are configured."""
    return bool(os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"))


def _have_s3_config() -> bool:
    """Check if S3 is configured."""
    return bool(
        os.getenv("S3_BUCKET")
        and os.getenv("AWS_ACCESS_KEY_ID")
        and os.getenv("AWS_SECRET_ACCESS_KEY")
    )


def _have_full_config() -> bool:
    """Check if both Modal and S3 are configured."""
    return _have_modal_tokens() and _have_s3_config()


class TestExecuteCodeTool:
    """Tests for execute_code_tool - actually executes code in Modal sandbox."""
    
    @pytest.mark.skipif(not _have_modal_tokens(), reason="Modal tokens not configured")
    def test_execute_simple_code(self, test_session_id, mock_runtime):
        """Test executing simple Python code in real Modal sandbox."""
        print(f"\n🚀 Testing simple code execution in session {test_session_id[:8]}...")
        
        code = "result = 2 + 2\nprint(f'Result: {result}')"
        
        # Call the underlying function using .func
        command = execute_code_tool.func(code, mock_runtime)
        
        # Check that a Command was returned
        assert command is not None
        assert "messages" in command.update
        
        # Extract the tool message
        tool_message = command.update["messages"][0]
        assert tool_message.tool_call_id == mock_runtime.tool_call_id
        
        # Parse the result (should be JSON)
        result = json.loads(tool_message.content)
        print(f"📊 Result: {result}")
        
        assert "stdout" in result
        assert "Result: 4" in result["stdout"] or "4" in result["stdout"]
        # Should not have errors
        assert not result.get("stderr") or result["stderr"] == ""
    
    @pytest.mark.skipif(not _have_modal_tokens(), reason="Modal tokens not configured")
    def test_execute_code_persists_state(self, test_session_id, mock_runtime):
        """Test that Python state actually persists across executions in same session."""
        print(f"\n🔄 Testing state persistence in session {test_session_id[:8]}...")
        
        # First execution: define a variable
        code1 = "x = 42\nprint('Set x to 42')"
        command1 = execute_code_tool.func(code1, mock_runtime)
        result1 = json.loads(command1.update["messages"][0].content)
        
        print(f"📊 First execution: {result1}")
        assert "Set x to 42" in result1["stdout"]
        
        # Second execution: use the variable (proves state persisted)
        code2 = "print(f'x + 1 = {x + 1}')"
        command2 = execute_code_tool.func(code2, mock_runtime)
        result2 = json.loads(command2.update["messages"][0].content)
        
        print(f"📊 Second execution: {result2}")
        assert "x + 1 = 43" in result2["stdout"]
    
    @pytest.mark.skipif(not _have_modal_tokens(), reason="Modal tokens not configured")
    def test_execute_code_with_pandas(self, test_session_id, mock_runtime):
        """Test executing code with pandas (verify packages available)."""
        print(f"\n🐼 Testing pandas in session {test_session_id[:8]}...")
        
        code = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
print(f'DataFrame shape: {df.shape}')
print(df.to_string())
"""
        
        command = execute_code_tool.func(code, mock_runtime)
        result = json.loads(command.update["messages"][0].content)
        
        print(f"📊 Result: {result}")
        assert "DataFrame shape: (3, 2)" in result["stdout"]
    
    @pytest.mark.skipif(not _have_modal_tokens(), reason="Modal tokens not configured")
    def test_execute_code_with_error(self, test_session_id, mock_runtime):
        """Test executing code that raises an error."""
        print(f"\n❌ Testing error handling in session {test_session_id[:8]}...")
        
        code = "x = 1 / 0  # Division by zero"
        
        command = execute_code_tool.func(code, mock_runtime)
        result = json.loads(command.update["messages"][0].content)
        
        print(f"📊 Error result: {result}")
        # Should have stderr with error message
        assert "stderr" in result
        assert "division by zero" in result["stderr"].lower()


@pytest.mark.asyncio
class TestLoadDatasetTool:
    """Tests for load_dataset_tool - mocks API fallback since S3 input/ is read-only."""
    
    @pytest.mark.skipif(not _have_full_config(), reason="Modal and S3 not fully configured")
    async def test_load_dataset_from_api(self, test_session_id, mock_runtime, test_dataset_bytes):
        """Test loading a dataset via API fallback (not in S3)."""
        print(f"\n📥 Testing load dataset from API in session {test_session_id[:8]}...")
        
        from unittest.mock import patch, AsyncMock
        
        dataset_id = f"test-api-dataset-{test_session_id[:8]}"
        
        # Mock the API helpers
        with patch("backend.graph.sandbox_tools.is_dataset_too_heavy", new_callable=AsyncMock) as mock_heavy, \
             patch("backend.graph.sandbox_tools.get_dataset_bytes", new_callable=AsyncMock) as mock_get:
            
            # Configure mocks
            mock_heavy.return_value = False  # Not too heavy
            mock_get.return_value = test_dataset_bytes
            
            # Load it using the tool - async tools use .coroutine attribute
            command = await load_dataset_tool.coroutine(dataset_id, mock_runtime)
        
        # Check result
        assert command is not None
        tool_message = command.update["messages"][0]
        result = json.loads(tool_message.content)
        
        print(f"📊 Load result: {result}")
        
        # Should have dataset metadata
        assert result["dataset_id"] == dataset_id
        assert "path" in result
        assert "shape" in result
        assert result["shape"] == [4, 3]  # 4 rows, 3 columns
        assert "columns" in result
        assert set(result["columns"]) == {"city", "population", "region"}


class TestExportDatasetTool:
    """Tests for export_dataset_tool - actually exports datasets to S3 output/ prefix."""
    
    @pytest.mark.skipif(not _have_full_config(), reason="Modal and S3 not fully configured")
    def test_export_nonexistent_dataset(self, test_session_id, mock_runtime):
        """Test exporting a dataset that doesn't exist."""
        print(f"\n📤 Testing export nonexistent dataset in session {test_session_id[:8]}...")
        
        command = export_dataset_tool.func("datasets/nonexistent-file-xyz.csv", mock_runtime)
        
        tool_message = command.update["messages"][0]
        result = json.loads(tool_message.content)
        
        print(f"📊 Export result: {result}")
        
        # Should have error
        assert "error" in result
        assert "not found" in result["error"].lower()
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not _have_full_config(), reason="Modal and S3 not fully configured")
    async def test_export_dataset_to_s3(self, test_session_id, mock_runtime, test_dataset_bytes):
        """Test creating a dataset in sandbox and exporting to S3 output/."""
        print(f"\n📤 Testing real export to S3 output/ in session {test_session_id[:8]}...")
        
        from unittest.mock import patch, AsyncMock
        
        dataset_id = f"export-test-{test_session_id[:8]}"
        
        # 1. Load dataset into sandbox (mock API)
        print("  Step 1: Loading dataset into sandbox...")
        with patch("backend.graph.sandbox_tools.is_dataset_too_heavy", new_callable=AsyncMock) as mock_heavy, \
             patch("backend.graph.sandbox_tools.get_dataset_bytes", new_callable=AsyncMock) as mock_get:
            
            mock_heavy.return_value = False
            mock_get.return_value = test_dataset_bytes
            
            load_cmd = await load_dataset_tool.coroutine(dataset_id, mock_runtime)
            load_result = json.loads(load_cmd.update["messages"][0].content)
            
        print(f"  ✅ Loaded: {load_result}")
        print(f"  Path: {load_result['path']}")
        print(f"  Rel path: {load_result.get('rel_path')}")
        dataset_path = load_result["rel_path"]  # Relative path
        
        # 2. Export to S3
        print(f"  Step 2: Exporting to S3 with path: {dataset_path}...")
        import time
        time.sleep(5)  # Brief pause for Modal volume to sync
        export_cmd = export_dataset_tool.func(dataset_path, mock_runtime)
        export_result = json.loads(export_cmd.update["messages"][0].content)
        
        print(f"  📊 Export result: {export_result}")
        
        # 3. Verify export succeeded
        assert "error" not in export_result
        assert "s3_key" in export_result
        assert export_result["s3_key"].startswith("output/datasets/")
        assert "sha256" in export_result
        
        # 4. Verify file exists in S3
        s3 = boto3.client("s3")
        bucket = os.getenv("S3_BUCKET")
        s3_key = export_result["s3_key"]
        
        try:
            response = s3.head_object(Bucket=bucket, Key=s3_key)
            print(f"  ✅ Verified in S3: s3://{bucket}/{s3_key}")
            assert response["ContentLength"] == export_result["size"]
        finally:
            # Cleanup: delete from S3 (may fail due to IAM permissions)
            try:
                s3.delete_object(Bucket=bucket, Key=s3_key)
                print(f"  🗑️  Cleaned up: {s3_key}")
            except Exception as e:
                print(f"  ⚠️  Cleanup failed (expected if no DeleteObject permission): {e.__class__.__name__}")


class TestExecutorCacheManagement:
    """Tests for executor cache lifecycle."""
    
    @pytest.mark.skipif(not _have_modal_tokens(), reason="Modal tokens not configured")
    def test_executor_reused_within_session(self, test_session_id, mock_runtime):
        """Test that the same executor is reused for multiple calls in a session."""
        print(f"\n♻️  Testing executor reuse in session {test_session_id[:8]}...")
        
        # First execution
        execute_code_tool.func("x = 1", mock_runtime)
        
        # Check cache has our session
        assert test_session_id in _executor_cache
        executor1 = _executor_cache[test_session_id]
        print(f"📦 Executor created: {id(executor1)}")
        
        # Second execution
        execute_code_tool.func("y = 2", mock_runtime)
        
        # Should be same executor instance
        executor2 = _executor_cache[test_session_id]
        print(f"📦 Executor reused: {id(executor2)}")
        assert executor1 is executor2, "Executor should be reused for same session"
    
    @pytest.mark.skipif(not _have_modal_tokens(), reason="Modal tokens not configured")
    def test_terminate_session_executor(self, test_session_id, mock_runtime):
        """Test that terminate_session_executor properly cleans up."""
        # Save original thread_id
        original_thread_id = get_thread_id()
        
        # Create a new session
        session_id = str(uuid.uuid4())
        set_thread_id(uuid.UUID(session_id))
        
        print(f"\n🗑️  Testing executor termination for session {session_id[:8]}...")
        
        try:
            # Execute code to create executor
            execute_code_tool.func("x = 1", mock_runtime)
            
            # Verify executor exists
            assert session_id in _executor_cache
            print(f"✅ Executor in cache")
            
            # Terminate
            terminate_session_executor(session_id)
            print(f"🛑 Executor terminated")
            
            # Verify executor removed
            assert session_id not in _executor_cache
            print(f"✅ Executor removed from cache")
            
        finally:
            # Cleanup in case of test failure
            _executor_cache.pop(session_id, None)
            # Restore original thread_id
            set_thread_id(original_thread_id)


@pytest.mark.asyncio
class TestIntegrationFlow:
    """End-to-end integration tests using real Modal and S3."""
    
    @pytest.mark.skipif(not _have_full_config(), reason="Modal and S3 not fully configured")
    async def test_full_workflow_load_and_analyze(self, test_session_id, mock_runtime, test_dataset_bytes):
        """Test a full workflow: load dataset (mock API), analyze with code, export to S3."""
        print(f"\n🔄 Testing full workflow in session {test_session_id[:8]}...")
        import time
        time.sleep(5)  # Brief pause for Modal volume to sync
        
        from unittest.mock import patch, AsyncMock
        
        # 1. Load dataset into Modal (mock API)
        dataset_id = f"integration-test-{test_session_id[:8]}"
        print(f"📥 Step 1: Loading dataset {dataset_id}...")
        
        with patch("backend.graph.sandbox_tools.is_dataset_too_heavy", new_callable=AsyncMock) as mock_heavy, \
             patch("backend.graph.sandbox_tools.get_dataset_bytes", new_callable=AsyncMock) as mock_get:
            
            mock_heavy.return_value = False
            mock_get.return_value = test_dataset_bytes
            
            load_cmd = await load_dataset_tool.coroutine(dataset_id, mock_runtime)
            load_result = json.loads(load_cmd.update["messages"][0].content)
        
        print(f"✅ Loaded: {load_result}")
        assert load_result["dataset_id"] == dataset_id
        assert load_result["shape"] == [4, 3]
        time.sleep(5)  # Brief pause for Modal volume to sync
        
        dataset_path = load_result["rel_path"]  # Relative path for export
        dataset_abs_path = load_result["path"]  # Full path for code execution
        
        # 2. List datasets
        print(f"📋 Step 2: Listing datasets...")
        time.sleep(5)  # Brief pause for Modal volume to sync
        list_cmd = list_datasets_tool.func(mock_runtime)
        list_result = json.loads(list_cmd.update["messages"][0].content)
        
        print(f"✅ Found {len(list_result)} datasets: {[d['path'] for d in list_result]}")
        # Should have our dataset
        assert len(list_result) > 0
        assert any(dataset_id in d["path"] for d in list_result)
        
        # 3. Analyze with code in sandbox
        print(f"🐍 Step 3: Analyzing dataset with Python...")
        code = f"""
import pandas as pd

# Load the dataset
df = pd.read_parquet('{dataset_abs_path}')

# Analyze
total_pop = df['population'].sum()
avg_pop = df['population'].mean()
max_city = df.loc[df['population'].idxmax(), 'city']

print(f'Total population: {{total_pop:,}}')
print(f'Average population: {{avg_pop:,.0f}}')
print(f'Largest city: {{max_city}}')
print(f'Cities: {{", ".join(df["city"].tolist())}}')
"""
        
        exec_cmd = execute_code_tool.func(code, mock_runtime)
        exec_result = json.loads(exec_cmd.update["messages"][0].content)
        
        print(f"✅ Analysis result:\n{exec_result['stdout']}")
        if exec_result.get('stderr'):
            print(f"⚠️  Stderr:\n{exec_result['stderr']}")
        
        # Verify calculations
        assert exec_result['stdout'], f"Expected stdout but got empty. stderr: {exec_result.get('stderr')}"
        assert "Total population: 6,059,000" in exec_result["stdout"]
        assert "Largest city: Roma" in exec_result["stdout"]
        assert "Milano" in exec_result["stdout"]

        time.sleep(5)  # Brief pause for Modal volume to sync
        
        # 4. Export dataset to S3 and verify
        print(f"📤 Step 4: Exporting dataset to S3...")
        export_cmd = export_dataset_tool.func(dataset_path, mock_runtime)
        export_result = json.loads(export_cmd.update["messages"][0].content)
        
        print(f"✅ Exported: {export_result.get('s3_url')}")
        assert "error" not in export_result
        assert "s3_key" in export_result
        
        # Cleanup: delete exported file (may fail due to IAM permissions)
        try:
            s3 = boto3.client("s3")
            bucket = os.getenv("S3_BUCKET")
            s3.delete_object(Bucket=bucket, Key=export_result["s3_key"])
            print(f"🗑️  Cleaned up exported file: {export_result['s3_key']}")
        except Exception as e:
            print(f"⚠️  Export cleanup failed (expected if no DeleteObject permission): {e.__class__.__name__}")
        
        print(f"🎉 Full workflow completed successfully!")

