"""
Tests for the AsyncPredicatoClient.
"""

import pytest
from pytest_httpx import HTTPXMock

from predicato import AsyncPredicatoClient
from predicato.exceptions import ConfigurationError, ValidationError


class TestAsyncClientInitialization:
    """Tests for async client initialization."""

    def test_init_with_url(self):
        """Test initializing with explicit URL."""
        client = AsyncPredicatoClient(base_url="http://localhost:8080")
        assert client._base_url == "http://localhost:8080"

    def test_init_with_env_var(self, monkeypatch):
        """Test initializing with environment variable."""
        monkeypatch.setenv("PREDICATO_URL", "http://env.example.com")
        client = AsyncPredicatoClient()
        assert client._base_url == "http://env.example.com"

    def test_init_without_url_raises(self):
        """Test that missing URL raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            AsyncPredicatoClient()
        assert "base_url" in str(exc_info.value)

    def test_init_with_group_id(self):
        """Test initializing with group_id."""
        client = AsyncPredicatoClient(
            base_url="http://localhost:8080",
            group_id="my-group",
        )
        assert client._group_id == "my-group"


class TestAsyncAddMessages:
    """Tests for async add_messages method."""

    @pytest.mark.asyncio
    async def test_add_messages_success(self, httpx_mock: HTTPXMock):
        """Test adding messages successfully."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8080/api/v1/ingest/messages",
            json={
                "success": True,
                "message": "Queued 2 messages for processing",
                "process_id": "proc_abc123",
            },
            status_code=202,
        )

        async with AsyncPredicatoClient(base_url="http://localhost:8080") as client:
            result = await client.add_messages(
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                group_id="test-group",
            )

        assert result.success is True
        assert result.process_id == "proc_abc123"
        assert result.queued_count == 2

    @pytest.mark.asyncio
    async def test_add_messages_requires_group_id(self, httpx_mock: HTTPXMock):
        """Test that group_id is required."""
        async with AsyncPredicatoClient(base_url="http://localhost:8080") as client:
            with pytest.raises(ValidationError) as exc_info:
                await client.add_messages(
                    messages=[{"role": "user", "content": "Hello"}],
                )
            assert "group_id" in str(exc_info.value)


class TestAsyncAddEpisode:
    """Tests for async add_episode method."""

    @pytest.mark.asyncio
    async def test_add_episode_success(self, httpx_mock: HTTPXMock):
        """Test adding an episode successfully."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8080/api/v1/ingest/messages",
            json={"success": True, "message": "OK", "process_id": "proc_123"},
            status_code=202,
        )

        async with AsyncPredicatoClient(base_url="http://localhost:8080") as client:
            result = await client.add_episode(
                name="Test Episode",
                content="This is test content.",
                group_id="test-group",
            )

        assert result is not None

    @pytest.mark.asyncio
    async def test_add_episode_validates_content_size(self):
        """Test that content size is validated."""
        async with AsyncPredicatoClient(base_url="http://localhost:8080") as client:
            with pytest.raises(ValidationError) as exc_info:
                await client.add_episode(
                    name="Test",
                    content="x" * (1_048_576 + 1),
                    group_id="test-group",
                )
            assert "1MB" in str(exc_info.value)


class TestAsyncAddEntity:
    """Tests for async add_entity method."""

    @pytest.mark.asyncio
    async def test_add_entity_success(self, httpx_mock: HTTPXMock):
        """Test adding an entity successfully."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8080/api/v1/ingest/entity",
            json={"success": True, "message": "Entity created"},
            status_code=201,
        )

        async with AsyncPredicatoClient(base_url="http://localhost:8080") as client:
            result = await client.add_entity(
                name="Alice",
                entity_type="Person",
                group_id="test-group",
                attributes={"role": "Engineer"},
            )

        assert result is not None


class TestAsyncSearch:
    """Tests for async search method."""

    @pytest.mark.asyncio
    async def test_search_success(self, httpx_mock: HTTPXMock):
        """Test searching successfully."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8080/api/v1/search",
            json={"facts": [], "total": 0},
            status_code=200,
        )

        async with AsyncPredicatoClient(base_url="http://localhost:8080") as client:
            result = await client.search(query="test query")

        assert result is not None
        assert result.nodes == []


class TestAsyncClearGraph:
    """Tests for async clear_graph method."""

    @pytest.mark.asyncio
    async def test_clear_graph_success(self, httpx_mock: HTTPXMock):
        """Test clearing graph successfully."""
        httpx_mock.add_response(
            method="DELETE",
            url="http://localhost:8080/api/v1/ingest/clear",
            json={"success": True, "message": "Cleared"},
            status_code=200,
        )

        async with AsyncPredicatoClient(base_url="http://localhost:8080") as client:
            result = await client.clear_graph(group_ids=["test-group"])

        assert result.success is True
        assert "test-group" in result.cleared_groups

    @pytest.mark.asyncio
    async def test_clear_graph_requires_group_ids(self):
        """Test that group_ids are required."""
        async with AsyncPredicatoClient(base_url="http://localhost:8080") as client:
            with pytest.raises(ValidationError) as exc_info:
                await client.clear_graph(group_ids=[])
            assert "group_ids" in str(exc_info.value)


class TestAsyncContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, httpx_mock: HTTPXMock):
        """Test using client as async context manager."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8080/api/v1/search",
            json={"facts": [], "total": 0},
            status_code=200,
        )

        async with AsyncPredicatoClient(base_url="http://localhost:8080") as client:
            result = await client.search(query="test")
            assert result is not None
