"""
Tests for the PredicatoClient.
"""

import pytest
from pytest_httpx import HTTPXMock

from predicato import PredicatoClient
from predicato.exceptions import ConfigurationError, ValidationError


class TestClientInitialization:
    """Tests for client initialization."""

    def test_init_with_url(self):
        """Test initializing with explicit URL."""
        client = PredicatoClient(base_url="http://localhost:8080")
        assert client._base_url == "http://localhost:8080"
        client.close()

    def test_init_with_env_var(self, monkeypatch):
        """Test initializing with environment variable."""
        monkeypatch.setenv("PREDICATO_URL", "http://env.example.com")
        client = PredicatoClient()
        assert client._base_url == "http://env.example.com"
        client.close()

    def test_init_without_url_raises(self):
        """Test that missing URL raises ConfigurationError."""
        with pytest.raises(ConfigurationError) as exc_info:
            PredicatoClient()
        assert "base_url" in str(exc_info.value)
        assert "PREDICATO_URL" in str(exc_info.value)

    def test_init_with_group_id(self):
        """Test initializing with group_id."""
        client = PredicatoClient(
            base_url="http://localhost:8080",
            group_id="my-group",
        )
        assert client._group_id == "my-group"
        client.close()

    def test_init_with_group_id_env_var(self, monkeypatch):
        """Test initializing with group_id from env."""
        monkeypatch.setenv("PREDICATO_GROUP_ID", "env-group")
        client = PredicatoClient(base_url="http://localhost:8080")
        assert client._group_id == "env-group"
        client.close()

    def test_init_explicit_overrides_env(self, monkeypatch):
        """Test that explicit parameters override environment."""
        monkeypatch.setenv("PREDICATO_URL", "http://env.example.com")
        monkeypatch.setenv("PREDICATO_GROUP_ID", "env-group")
        client = PredicatoClient(
            base_url="http://explicit.example.com",
            group_id="explicit-group",
        )
        assert client._base_url == "http://explicit.example.com"
        assert client._group_id == "explicit-group"
        client.close()

    def test_context_manager(self):
        """Test using client as context manager."""
        with PredicatoClient(base_url="http://localhost:8080") as client:
            assert client._base_url == "http://localhost:8080"
        # Client should be closed after exiting context


class TestAddMessages:
    """Tests for add_messages method."""

    def test_add_messages_success(self, httpx_mock: HTTPXMock):
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

        with PredicatoClient(base_url="http://localhost:8080") as client:
            result = client.add_messages(
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                group_id="test-group",
            )

        assert result.success is True
        assert result.process_id == "proc_abc123"
        assert result.queued_count == 2

    def test_add_messages_requires_group_id(self, httpx_mock: HTTPXMock):
        """Test that group_id is required."""
        with PredicatoClient(base_url="http://localhost:8080") as client:
            with pytest.raises(ValidationError) as exc_info:
                client.add_messages(
                    messages=[{"role": "user", "content": "Hello"}],
                )
            assert "group_id" in str(exc_info.value)

    def test_add_messages_uses_default_group(self, httpx_mock: HTTPXMock):
        """Test that default group_id is used."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8080/api/v1/ingest/messages",
            json={"success": True, "message": "OK"},
            status_code=202,
        )

        with PredicatoClient(
            base_url="http://localhost:8080", group_id="default-group"
        ) as client:
            result = client.add_messages(
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert result.success is True


class TestAddEpisode:
    """Tests for add_episode method."""

    def test_add_episode_success(self, httpx_mock: HTTPXMock):
        """Test adding an episode successfully."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8080/api/v1/ingest/messages",
            json={"success": True, "message": "OK", "process_id": "proc_123"},
            status_code=202,
        )

        with PredicatoClient(base_url="http://localhost:8080") as client:
            result = client.add_episode(
                name="Test Episode",
                content="This is test content.",
                group_id="test-group",
            )

        assert result is not None

    def test_add_episode_validates_content_size(self):
        """Test that content size is validated."""
        with PredicatoClient(base_url="http://localhost:8080") as client:
            with pytest.raises(ValidationError) as exc_info:
                client.add_episode(
                    name="Test",
                    content="x" * (1_048_576 + 1),
                    group_id="test-group",
                )
            assert "1MB" in str(exc_info.value)


class TestAddEntity:
    """Tests for add_entity method."""

    def test_add_entity_success(self, httpx_mock: HTTPXMock):
        """Test adding an entity successfully."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8080/api/v1/ingest/entity",
            json={"success": True, "message": "Entity created"},
            status_code=201,
        )

        with PredicatoClient(base_url="http://localhost:8080") as client:
            result = client.add_entity(
                name="Alice",
                entity_type="Person",
                group_id="test-group",
                attributes={"role": "Engineer"},
            )

        assert result is not None


class TestSearch:
    """Tests for search method."""

    def test_search_success(self, httpx_mock: HTTPXMock):
        """Test searching successfully."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8080/api/v1/search",
            json={"facts": [], "total": 0},
            status_code=200,
        )

        with PredicatoClient(base_url="http://localhost:8080") as client:
            result = client.search(query="test query")

        assert result is not None
        assert result.nodes == []


class TestClearGraph:
    """Tests for clear_graph method."""

    def test_clear_graph_success(self, httpx_mock: HTTPXMock):
        """Test clearing graph successfully."""
        httpx_mock.add_response(
            method="DELETE",
            url="http://localhost:8080/api/v1/ingest/clear",
            json={"success": True, "message": "Cleared"},
            status_code=200,
        )

        with PredicatoClient(base_url="http://localhost:8080") as client:
            result = client.clear_graph(group_ids=["test-group"])

        assert result.success is True
        assert "test-group" in result.cleared_groups

    def test_clear_graph_requires_group_ids(self):
        """Test that group_ids are required."""
        with PredicatoClient(base_url="http://localhost:8080") as client:
            with pytest.raises(ValidationError) as exc_info:
                client.clear_graph(group_ids=[])
            assert "group_ids" in str(exc_info.value)
