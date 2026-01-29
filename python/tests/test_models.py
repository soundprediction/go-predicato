"""
Tests for Pydantic models.
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError as PydanticValidationError

from predicato.models import (
    AddEntityNodeRequest,
    AddMessagesRequest,
    ClearDataRequest,
    Edge,
    Episode,
    Message,
    Node,
    SearchRequest,
)


class TestEpisode:
    """Tests for the Episode model."""

    def test_create_episode_minimal(self):
        """Test creating an episode with minimal fields."""
        episode = Episode(
            name="Test",
            content="Test content",
            group_id="test-group",
        )
        assert episode.name == "Test"
        assert episode.content == "Test content"
        assert episode.group_id == "test-group"
        assert episode.source == ""
        assert episode.metadata is None

    def test_create_episode_full(self, sample_datetime):
        """Test creating an episode with all fields."""
        episode = Episode(
            id="ep-123",
            name="Full Episode",
            content="Full content",
            source="document",
            reference=sample_datetime,
            created_at=sample_datetime,
            group_id="my-group",
            metadata={"key": "value"},
        )
        assert episode.id == "ep-123"
        assert episode.name == "Full Episode"
        assert episode.source == "document"
        assert episode.reference == sample_datetime
        assert episode.metadata == {"key": "value"}

    def test_episode_name_max_length(self):
        """Test that name field enforces max length."""
        with pytest.raises(PydanticValidationError) as exc_info:
            Episode(
                name="x" * 1025,
                content="test",
                group_id="test",
            )
        assert "max_length" in str(exc_info.value).lower() or "string_too_long" in str(
            exc_info.value
        ).lower()

    def test_episode_content_max_length(self):
        """Test that content field enforces max length (1MB)."""
        with pytest.raises(PydanticValidationError):
            Episode(
                name="test",
                content="x" * (1_048_576 + 1),
                group_id="test",
            )

    def test_episode_serialization(self, sample_datetime):
        """Test that episode serializes correctly."""
        episode = Episode(
            name="Test",
            content="Content",
            group_id="group",
            reference=sample_datetime,
        )
        data = episode.model_dump(mode="json")
        assert data["name"] == "Test"
        assert data["content"] == "Content"
        assert data["group_id"] == "group"
        assert "reference" in data


class TestMessage:
    """Tests for the Message model."""

    def test_create_message(self):
        """Test creating a message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is None

    def test_message_with_timestamp(self, sample_datetime):
        """Test creating a message with timestamp."""
        msg = Message(role="assistant", content="Hi", timestamp=sample_datetime)
        assert msg.timestamp == sample_datetime

    def test_message_valid_roles(self):
        """Test that only valid roles are accepted."""
        for role in ["user", "assistant", "system"]:
            msg = Message(role=role, content="test")
            assert msg.role == role

    def test_message_invalid_role(self):
        """Test that invalid roles are rejected."""
        with pytest.raises(PydanticValidationError):
            Message(role="unknown", content="test")

    def test_message_content_max_length(self):
        """Test that content enforces max length."""
        with pytest.raises(PydanticValidationError):
            Message(role="user", content="x" * (1_048_576 + 1))


class TestAddMessagesRequest:
    """Tests for the AddMessagesRequest model."""

    def test_create_request(self):
        """Test creating a request."""
        messages = [
            Message(role="user", content="Hi"),
            Message(role="assistant", content="Hello"),
        ]
        request = AddMessagesRequest(group_id="test", messages=messages)
        assert request.group_id == "test"
        assert len(request.messages) == 2

    def test_request_requires_messages(self):
        """Test that messages are required."""
        with pytest.raises(PydanticValidationError):
            AddMessagesRequest(group_id="test", messages=[])

    def test_request_max_messages(self):
        """Test that max 1000 messages are allowed."""
        messages = [Message(role="user", content=f"msg {i}") for i in range(1001)]
        with pytest.raises(PydanticValidationError):
            AddMessagesRequest(group_id="test", messages=messages)


class TestAddEntityNodeRequest:
    """Tests for the AddEntityNodeRequest model."""

    def test_create_request(self):
        """Test creating a request."""
        request = AddEntityNodeRequest(
            group_id="test",
            name="Alice",
            entity_type="Person",
        )
        assert request.name == "Alice"
        assert request.entity_type == "Person"
        assert request.attributes == {}

    def test_request_with_attributes(self):
        """Test creating a request with attributes."""
        request = AddEntityNodeRequest(
            group_id="test",
            name="Alice",
            entity_type="Person",
            attributes={"role": "Engineer"},
        )
        assert request.attributes == {"role": "Engineer"}

    def test_request_max_attributes(self):
        """Test that max 100 attributes are allowed."""
        attrs = {f"key_{i}": f"value_{i}" for i in range(101)}
        with pytest.raises(ValueError, match="more than 100"):
            AddEntityNodeRequest(
                group_id="test",
                name="Test",
                entity_type="Test",
                attributes=attrs,
            )


class TestClearDataRequest:
    """Tests for the ClearDataRequest model."""

    def test_create_request(self):
        """Test creating a request."""
        request = ClearDataRequest(group_ids=["group1", "group2"])
        assert request.group_ids == ["group1", "group2"]

    def test_request_requires_group_ids(self):
        """Test that at least one group_id is required."""
        with pytest.raises(PydanticValidationError):
            ClearDataRequest(group_ids=[])


class TestSearchRequest:
    """Tests for the SearchRequest model."""

    def test_create_request_minimal(self):
        """Test creating a request with defaults."""
        request = SearchRequest(query="test query")
        assert request.query == "test query"
        assert request.limit == 10
        assert request.include_edges is False
        assert request.min_score == 0.0

    def test_create_request_full(self):
        """Test creating a request with all options."""
        request = SearchRequest(
            query="test",
            group_id="my-group",
            limit=50,
            include_edges=True,
            min_score=0.5,
        )
        assert request.group_id == "my-group"
        assert request.limit == 50
        assert request.include_edges is True
        assert request.min_score == 0.5

    def test_request_query_required(self):
        """Test that query is required."""
        with pytest.raises(PydanticValidationError):
            SearchRequest(query="")

    def test_request_limit_bounds(self):
        """Test that limit is bounded 1-100."""
        with pytest.raises(PydanticValidationError):
            SearchRequest(query="test", limit=0)
        with pytest.raises(PydanticValidationError):
            SearchRequest(query="test", limit=101)

    def test_request_min_score_bounds(self):
        """Test that min_score is bounded 0-1."""
        with pytest.raises(PydanticValidationError):
            SearchRequest(query="test", min_score=-0.1)
        with pytest.raises(PydanticValidationError):
            SearchRequest(query="test", min_score=1.1)


class TestNode:
    """Tests for the Node model."""

    def test_create_node(self, sample_datetime):
        """Test creating a node."""
        node = Node(
            uuid="node-123",
            group_id="test",
            name="Test Node",
            type="entity",
            created_at=sample_datetime,
        )
        assert node.uuid == "node-123"
        assert node.name == "Test Node"
        assert node.type == "entity"

    def test_node_optional_fields(self, sample_datetime):
        """Test node with optional fields."""
        node = Node(
            uuid="node-123",
            group_id="test",
            name="Test",
            type="entity",
            entity_type="Person",
            summary="A test node",
            created_at=sample_datetime,
            updated_at=sample_datetime,
            source_ids=["ep-1", "ep-2"],
            metadata={"key": "value"},
        )
        assert node.entity_type == "Person"
        assert node.summary == "A test node"
        assert node.source_ids == ["ep-1", "ep-2"]


class TestEdge:
    """Tests for the Edge model."""

    def test_create_edge(self, sample_datetime):
        """Test creating an edge."""
        edge = Edge(
            uuid="edge-123",
            group_id="test",
            source_node_uuid="node-1",
            target_node_uuid="node-2",
            type="KNOWS",
            created_at=sample_datetime,
        )
        assert edge.uuid == "edge-123"
        assert edge.source_node_uuid == "node-1"
        assert edge.target_node_uuid == "node-2"
        assert edge.type == "KNOWS"

    def test_edge_optional_fields(self, sample_datetime):
        """Test edge with optional fields."""
        edge = Edge(
            uuid="edge-123",
            group_id="test",
            source_node_uuid="node-1",
            target_node_uuid="node-2",
            type="WORKS_WITH",
            name="Collaboration",
            fact="Alice works with Bob on the project",
            created_at=sample_datetime,
            episodes=["ep-1"],
        )
        assert edge.name == "Collaboration"
        assert edge.fact == "Alice works with Bob on the project"
        assert edge.episodes == ["ep-1"]
