"""
End-to-end tests for the complete LG-Urban workflow.
Tests the full flow from user message to artifact display.
"""
import pytest
import json
from httpx import AsyncClient
from unittest.mock import patch

from backend.main import app


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.fixture
    async def client(self):
        """Create test client."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.fixture
    async def test_thread_id(self, client):
        """Create a test thread and return its ID."""
        response = await client.post("/api/threads", json={"user_id": "test-user"})
        assert response.status_code == 200
        thread_data = response.json()
        return thread_data["id"]
    
    @pytest.mark.asyncio
    async def test_create_thread_and_message(self, client):
        """Test creating a thread and sending a message."""
        # Create thread
        response = await client.post("/api/threads", json={"user_id": "test-user"})
        assert response.status_code == 200
        thread_data = response.json()
        thread_id = thread_data["id"]
        
        # Send message
        message_data = {
            "message_id": "test-msg-1",
            "content": {"text": "Hello, can you help me create a simple plot?"},
            "role": "user"
        }
        
        response = await client.post(
            f"/api/threads/{thread_id}/messages",
            json=message_data
        )
        assert response.status_code == 200
        
        # Verify message was created
        response = await client.get(f"/api/threads/{thread_id}/messages")
        assert response.status_code == 200
        messages = response.json()
        assert len(messages) >= 1  # At least the user message
    
    @pytest.mark.asyncio
    @patch('backend.graph.tools.get_session_manager')
    async def test_code_execution_with_artifacts(self, mock_session_manager, client, test_thread_id):
        """Test code execution that creates artifacts."""
        # Mock session manager to avoid Docker dependency
        mock_manager = AsyncMock()
        mock_manager.exec.return_value = {
            "ok": True,
            "stdout": "Plot created successfully!",
            "stderr": "",
            "artifacts": [{
                "name": "test_plot.png",
                "mime": "image/png",
                "url": "/api/artifacts/123",
                "size": 12345
            }]
        }
        mock_session_manager.return_value = mock_manager
        
        # Send message requesting code execution
        message_data = {
            "message_id": "test-msg-2",
            "content": {"text": "Create a simple sine wave plot and save it as plot.png"},
            "role": "user"
        }
        
        response = await client.post(
            f"/api/threads/{test_thread_id}/messages",
            json=message_data
        )
        assert response.status_code == 200
        
        # Verify tool was called
        mock_manager.exec.assert_called_once()
        call_args = mock_manager.exec.call_args
        assert "sin" in call_args[1]["code"].lower() or "plot" in call_args[1]["code"].lower()
    
    @pytest.mark.asyncio
    async def test_artifact_download(self, client, test_thread_id):
        """Test artifact download functionality."""
        # This would require setting up actual artifacts in the database
        # For now, just test the endpoint exists
        response = await client.get("/api/artifacts/nonexistent")
        assert response.status_code == 404  # Expected for non-existent artifact
    
    @pytest.mark.asyncio
    async def test_thread_deletion_cleanup(self, client):
        """Test that deleting a thread cleans up associated data."""
        # Create thread
        response = await client.post("/api/threads", json={"user_id": "test-user"})
        assert response.status_code == 200
        thread_data = response.json()
        thread_id = thread_data["id"]
        
        # Send a message
        message_data = {
            "message_id": "test-msg-3",
            "content": {"text": "Hello"},
            "role": "user"
        }
        response = await client.post(
            f"/api/threads/{thread_id}/messages",
            json=message_data
        )
        assert response.status_code == 200
        
        # Delete thread
        response = await client.delete(f"/api/threads/{thread_id}")
        assert response.status_code == 200
        
        # Verify thread is gone
        response = await client.get(f"/api/threads/{thread_id}")
        assert response.status_code == 404
        
        # Verify messages are gone
        response = await client.get(f"/api/threads/{thread_id}/messages")
        assert response.status_code == 404


class TestAPIEndpoints:
    """Test individual API endpoints."""
    
    @pytest.fixture
    async def client(self):
        """Create test client."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check endpoint."""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_thread_listing(self, client):
        """Test thread listing."""
        # Create a thread
        response = await client.post("/api/threads", json={"user_id": "test-user"})
        assert response.status_code == 200
        
        # List threads
        response = await client.get("/api/threads?user_id=test-user")
        assert response.status_code == 200
        threads = response.json()
        assert len(threads) >= 1
    
    @pytest.mark.asyncio
    async def test_message_streaming(self, client):
        """Test message streaming endpoint."""
        # Create thread
        response = await client.post("/api/threads", json={"user_id": "test-user"})
        assert response.status_code == 200
        thread_data = response.json()
        thread_id = thread_data["id"]
        
        # Send message (this should stream)
        message_data = {
            "message_id": "test-msg-stream",
            "content": {"text": "Hello"},
            "role": "user"
        }
        
        response = await client.post(
            f"/api/threads/{thread_id}/messages",
            json=message_data
        )
        assert response.status_code == 200
        
        # Verify response is streaming (SSE format)
        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type or "text/plain" in content_type


class TestErrorHandling:
    """Test error handling across the system."""
    
    @pytest.fixture
    async def client(self):
        """Create test client."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_invalid_thread_id(self, client):
        """Test handling of invalid thread IDs."""
        invalid_id = "00000000-0000-0000-0000-000000000000"
        
        # Try to get messages for non-existent thread
        response = await client.get(f"/api/threads/{invalid_id}/messages")
        assert response.status_code == 404
        
        # Try to send message to non-existent thread
        message_data = {
            "message_id": "test-msg",
            "content": {"text": "Hello"},
            "role": "user"
        }
        response = await client.post(
            f"/api/threads/{invalid_id}/messages",
            json=message_data
        )
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_duplicate_message_id(self, client):
        """Test handling of duplicate message IDs."""
        # Create thread
        response = await client.post("/api/threads", json={"user_id": "test-user"})
        assert response.status_code == 200
        thread_data = response.json()
        thread_id = thread_data["id"]
        
        message_data = {
            "message_id": "duplicate-test",
            "content": {"text": "First message"},
            "role": "user"
        }
        
        # Send first message
        response = await client.post(
            f"/api/threads/{thread_id}/messages",
            json=message_data
        )
        assert response.status_code == 200
        
        # Try to send duplicate message ID
        message_data["content"] = {"text": "Second message"}
        response = await client.post(
            f"/api/threads/{thread_id}/messages",
            json=message_data
        )
        assert response.status_code == 409  # Conflict
