"""
Data models for the Predicato Python client.

This module contains Pydantic models that match the Go server's DTOs exactly,
providing type-safe request and response handling.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Request Models
# =============================================================================


class Episode(BaseModel):
    """
    An episode represents a content unit to be ingested into the knowledge graph.

    Maps to: pkg/types/types.go:104-115

    Args:
        id: Optional episode identifier.
        name: Name/title of the episode (max 1024 chars).
        content: The content to be processed (max 1MB).
        source: Source identifier for the content.
        reference: Reference timestamp for the content.
        created_at: When the episode was created.
        group_id: Group identifier for data isolation.
        metadata: Additional metadata for the episode.

    Example:
        >>> episode = Episode(
        ...     name="Meeting Notes",
        ...     content="Discussed project timeline...",
        ...     group_id="project-alpha",
        ...     source="meeting"
        ... )
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str | None = None
    name: str = Field(..., max_length=1024)
    content: str = Field(..., max_length=1_048_576)  # 1MB
    source: str = Field(default="", max_length=256)
    reference: datetime | None = None
    created_at: datetime | None = None
    group_id: str = Field(..., max_length=256)
    metadata: dict[str, Any] | None = None


class Message(BaseModel):
    """
    A message in a conversation format.

    Maps to: pkg/server/dto/ingest.go

    Args:
        role: The role of the message sender (user, assistant, or system).
        content: The message content (max 1MB).
        timestamp: When the message was sent.

    Example:
        >>> message = Message(
        ...     role="user",
        ...     content="What is the project status?"
        ... )
    """

    model_config = ConfigDict(populate_by_name=True)

    role: Literal["user", "assistant", "system"]
    content: str = Field(..., max_length=1_048_576)  # 1MB
    timestamp: datetime | None = None


class AddMessagesRequest(BaseModel):
    """
    Request to add multiple messages to the knowledge graph.

    Maps to: pkg/server/dto/ingest.go

    Args:
        group_id: Group identifier for data isolation.
        messages: List of messages to add (max 1000).
        reference: Reference timestamp for the messages.

    Example:
        >>> request = AddMessagesRequest(
        ...     group_id="conversation-001",
        ...     messages=[
        ...         Message(role="user", content="Hello"),
        ...         Message(role="assistant", content="Hi there!")
        ...     ]
        ... )
    """

    model_config = ConfigDict(populate_by_name=True)

    group_id: str = Field(..., max_length=256)
    messages: list[Message] = Field(..., min_length=1, max_length=1000)
    reference: datetime | None = None


class AddEntityNodeRequest(BaseModel):
    """
    Request to create a standalone entity node.

    Maps to: pkg/server/dto/ingest.go

    Args:
        group_id: Group identifier for data isolation.
        name: Name of the entity (max 1024 chars).
        entity_type: Type of the entity (max 256 chars).
        attributes: Additional attributes for the entity (max 100 keys).

    Example:
        >>> request = AddEntityNodeRequest(
        ...     group_id="project-alpha",
        ...     name="Alice Smith",
        ...     entity_type="Person",
        ...     attributes={"role": "Engineer"}
        ... )
    """

    model_config = ConfigDict(populate_by_name=True)

    group_id: str = Field(..., max_length=256)
    name: str = Field(..., max_length=1024)
    entity_type: str = Field(..., max_length=256)
    attributes: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Validate attribute count after initialization."""
        if len(self.attributes) > 100:
            raise ValueError("attributes cannot have more than 100 keys")


class ClearDataRequest(BaseModel):
    """
    Request to clear data for specified groups.

    Args:
        group_ids: List of group IDs to clear (at least one required).

    Example:
        >>> request = ClearDataRequest(group_ids=["project-alpha"])
    """

    model_config = ConfigDict(populate_by_name=True)

    group_ids: list[str] = Field(..., min_length=1)


class SearchRequest(BaseModel):
    """
    Request to search the knowledge graph.

    Args:
        query: The search query (min 1 char).
        group_id: Optional group to filter results.
        limit: Maximum results to return (1-100, default 10).
        include_edges: Whether to include edges in results.
        min_score: Minimum relevance score (0.0-1.0).

    Example:
        >>> request = SearchRequest(
        ...     query="project timeline",
        ...     group_id="project-alpha",
        ...     limit=20,
        ...     include_edges=True
        ... )
    """

    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(..., min_length=1)
    group_id: str | None = None
    limit: int = Field(default=10, ge=1, le=100)
    include_edges: bool = False
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


# =============================================================================
# Response Models
# =============================================================================


class Node(BaseModel):
    """
    A node in the knowledge graph.

    Maps to: pkg/types/types.go

    Args:
        uuid: Unique identifier for the node.
        group_id: Group the node belongs to.
        name: Name of the node.
        type: Node type (entity, episodic, community).
        entity_type: Specific entity type if applicable.
        summary: Summary of the node content.
        created_at: When the node was created.
        updated_at: When the node was last updated.
        valid_from: Start of validity period.
        valid_to: End of validity period.
        source_ids: IDs of source episodes.
        metadata: Additional metadata.
    """

    model_config = ConfigDict(populate_by_name=True)

    uuid: str
    group_id: str
    name: str
    type: str  # "entity", "episodic", "community"
    entity_type: str | None = None
    summary: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    source_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class Edge(BaseModel):
    """
    An edge (relationship) in the knowledge graph.

    Maps to: pkg/types/types.go

    Args:
        uuid: Unique identifier for the edge.
        group_id: Group the edge belongs to.
        source_node_uuid: UUID of the source node.
        target_node_uuid: UUID of the target node.
        type: Relationship type.
        name: Optional name for the edge.
        fact: Fact represented by the edge.
        created_at: When the edge was created.
        expired_at: When the edge expired.
        valid_at: Start of validity period.
        invalid_at: End of validity period.
        episodes: Episode IDs associated with this edge.
        metadata: Additional metadata.
    """

    model_config = ConfigDict(populate_by_name=True)

    uuid: str
    group_id: str
    source_node_uuid: str
    target_node_uuid: str
    type: str
    name: str | None = None
    fact: str | None = None
    created_at: datetime
    expired_at: datetime | None = None
    valid_at: datetime | None = None
    invalid_at: datetime | None = None
    episodes: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class AddEpisodeResults(BaseModel):
    """
    Results from adding a single episode.

    Maps to: pkg/types/types.go:255-268

    Args:
        episode: The created episode node.
        episodic_edges: Edges from the episode to entities.
        nodes: Entity nodes extracted from the episode.
        edges: Relationship edges extracted.
        communities: Community nodes affected.
        community_edges: Community relationship edges.
    """

    model_config = ConfigDict(populate_by_name=True)

    episode: Node | None = None
    episodic_edges: list[Edge] = Field(default_factory=list)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    communities: list[Node] = Field(default_factory=list)
    community_edges: list[Edge] = Field(default_factory=list)


class AddBulkEpisodeResults(BaseModel):
    """
    Results from adding multiple episodes in bulk.

    Maps to: pkg/types/types.go:271-284

    Args:
        episodes: Created episode nodes.
        episodic_edges: Edges from episodes to entities.
        nodes: Entity nodes extracted.
        edges: Relationship edges extracted.
        communities: Community nodes affected.
        community_edges: Community relationship edges.
    """

    model_config = ConfigDict(populate_by_name=True)

    episodes: list[Node] = Field(default_factory=list)
    episodic_edges: list[Edge] = Field(default_factory=list)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    communities: list[Node] = Field(default_factory=list)
    community_edges: list[Edge] = Field(default_factory=list)


class AddMessagesResponse(BaseModel):
    """
    Response from adding messages.

    Args:
        success: Whether the operation succeeded.
        message: Status message.
        process_id: ID for tracking async processing.
        queued_count: Number of messages queued.
    """

    model_config = ConfigDict(populate_by_name=True)

    success: bool
    message: str
    process_id: str | None = None
    queued_count: int = 0


class AddEntityResults(BaseModel):
    """
    Results from adding an entity node.

    Args:
        node: The created entity node.
    """

    model_config = ConfigDict(populate_by_name=True)

    node: Node | None = None


class SearchResults(BaseModel):
    """
    Results from a search query.

    Args:
        nodes: Matching nodes.
        edges: Related edges (if include_edges was True).
        node_scores: Relevance scores for nodes.
        edge_scores: Relevance scores for edges.
    """

    model_config = ConfigDict(populate_by_name=True)

    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    node_scores: dict[str, float] = Field(default_factory=dict)
    edge_scores: dict[str, float] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """
    Error response from the server.

    Args:
        error: Error type/code.
        message: Error message.
        detail: Additional error details.
    """

    model_config = ConfigDict(populate_by_name=True)

    error: str
    message: str
    detail: str | None = None


class ClearDataResponse(BaseModel):
    """
    Response from clearing data.

    Args:
        success: Whether the operation succeeded.
        cleared_groups: Groups that were cleared.
        failed_groups: Groups that failed to clear.
        errors: Error messages for failed groups.
    """

    model_config = ConfigDict(populate_by_name=True)

    success: bool
    cleared_groups: list[str] = Field(default_factory=list)
    failed_groups: list[str] = Field(default_factory=list)
    errors: dict[str, str] = Field(default_factory=dict)
