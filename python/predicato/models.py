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


# =============================================================================
# Two-Stage Ingestion Models
# =============================================================================


class ExtractEpisodeRequest(BaseModel):
    """
    Request to extract entities from an episode without promoting to graph.

    This is the first stage of two-stage ingestion.

    Args:
        id: Optional episode identifier.
        name: Name/title of the episode (max 1024 chars).
        content: The content to process (max 1MB).
        source: Source identifier for the content.
        group_id: Group identifier for data isolation.
        reference: Reference timestamp for the content.
        metadata: Additional metadata for the episode.
        entity_types: Custom entity type definitions.
        excluded_entity_types: Entity types to exclude from extraction.
        edge_types: Custom edge type definitions.
        edge_type_map: Mapping of entity pairs to edge types.
        previous_episode_uuids: UUIDs of previous episodes for context.
        max_characters: Maximum characters per chunk.
        use_yaml: Whether to use YAML for LLM interchange.

    Example:
        >>> request = ExtractEpisodeRequest(
        ...     name="Meeting Notes",
        ...     content="Discussed project timeline...",
        ...     group_id="project-alpha"
        ... )
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str | None = None
    name: str = Field(..., max_length=1024)
    content: str = Field(..., max_length=1_048_576)
    source: str = Field(default="", max_length=256)
    group_id: str = Field(..., max_length=256)
    reference: datetime | None = None
    metadata: dict[str, Any] | None = None
    entity_types: dict[str, Any] | None = None
    excluded_entity_types: list[str] | None = None
    edge_types: dict[str, Any] | None = None
    edge_type_map: dict[str, dict[str, Any]] | None = None
    previous_episode_uuids: list[str] | None = None
    max_characters: int | None = None
    use_yaml: bool = False


class ExtractedNode(BaseModel):
    """
    A raw entity extracted from content, before graph promotion.

    Args:
        id: Unique identifier for the extracted node.
        source_id: ID of the source episode.
        group_id: Group the node belongs to.
        name: Name of the entity.
        type: Entity type.
        description: Description/summary of the entity.
        chunk_index: Index of the chunk this was extracted from.
        created_at: When the extraction occurred.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str
    source_id: str
    group_id: str
    name: str
    type: str
    description: str = ""
    chunk_index: int = 0
    created_at: datetime | None = None


class ExtractedEdge(BaseModel):
    """
    A raw relationship extracted from content, before graph promotion.

    Args:
        id: Unique identifier for the extracted edge.
        source_id: ID of the source episode.
        group_id: Group the edge belongs to.
        source_node_name: Name of the source entity.
        target_node_name: Name of the target entity.
        relation: Relationship type/name.
        description: Description of the relationship.
        weight: Relationship strength/weight.
        chunk_index: Index of the chunk this was extracted from.
        created_at: When the extraction occurred.
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str
    source_id: str
    group_id: str
    source_node_name: str
    target_node_name: str
    relation: str
    description: str = ""
    weight: float = 1.0
    chunk_index: int = 0
    created_at: datetime | None = None


class ExtractionMetadata(BaseModel):
    """
    Metadata about the extraction process.

    Args:
        model_used: NLP model used for extraction.
        embedding_model: Model used for embeddings.
        embedding_dimensions: Dimension of embeddings.
        total_tokens: Estimated token count processed.
    """

    model_config = ConfigDict(populate_by_name=True)

    model_used: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = None
    total_tokens: int | None = None


class ExtractionResults(BaseModel):
    """
    Results from extracting knowledge from an episode.

    This is returned by extract_to_facts and contains raw extractions
    before graph modeling.

    Args:
        source_id: ID of the source episode in the fact store.
        extracted_nodes: Raw entities before resolution/deduplication.
        extracted_edges: Raw relationships before resolution.
        chunk_count: Number of chunks the episode was split into.
        extraction_time: How long the extraction took.
        metadata: Additional extraction information.

    Example:
        >>> results = client.extract_to_facts(episode)
        >>> print(f"Extracted {len(results.extracted_nodes)} entities")
        >>> print(f"Source ID for promotion: {results.source_id}")
    """

    model_config = ConfigDict(populate_by_name=True)

    source_id: str
    extracted_nodes: list[ExtractedNode] = Field(default_factory=list)
    extracted_edges: list[ExtractedEdge] = Field(default_factory=list)
    chunk_count: int = 0
    extraction_time: str | None = None
    metadata: ExtractionMetadata | None = None


class PromoteToGraphRequest(BaseModel):
    """
    Request to promote extracted knowledge to the graph.

    This is the second stage of two-stage ingestion.

    Args:
        source_id: ID of the source to promote (from extraction).
        entity_types: Custom entity type definitions.
        edge_types: Custom edge type definitions.
        edge_type_map: Mapping of entity pairs to edge types.
        previous_episode_uuids: UUIDs of previous episodes for context.
        skip_reflexion: Skip reflexion step for faster processing.
        skip_resolution: Skip entity resolution.
        skip_attributes: Skip attribute extraction.
        skip_edge_resolution: Skip edge resolution.
        use_yaml: Whether to use YAML for LLM interchange.

    Example:
        >>> request = PromoteToGraphRequest(
        ...     source_id="episode-123",
        ...     skip_resolution=True  # Faster but less accurate
        ... )
    """

    model_config = ConfigDict(populate_by_name=True)

    source_id: str
    entity_types: dict[str, Any] | None = None
    edge_types: dict[str, Any] | None = None
    edge_type_map: dict[str, dict[str, Any]] | None = None
    previous_episode_uuids: list[str] | None = None
    skip_reflexion: bool = False
    skip_resolution: bool = False
    skip_attributes: bool = False
    skip_edge_resolution: bool = False
    use_yaml: bool = False


class PromoteToGraphResults(BaseModel):
    """
    Results from promoting extracted knowledge to the graph.

    This is returned by promote_to_graph and contains the final
    resolved entities and relationships.

    Args:
        episode: The created episode node.
        episodic_edges: Edges from the episode to entities.
        nodes: Resolved entity nodes.
        edges: Resolved relationship edges.
        communities: Community nodes created/updated.
        community_edges: Community relationship edges.

    Example:
        >>> results = client.promote_to_graph(source_id)
        >>> print(f"Created {len(results.nodes)} entities")
        >>> print(f"Created {len(results.edges)} relationships")
    """

    model_config = ConfigDict(populate_by_name=True)

    episode: Node | None = None
    episodic_edges: list[Edge] = Field(default_factory=list)
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    communities: list[Node] = Field(default_factory=list)
    community_edges: list[Edge] = Field(default_factory=list)


# =============================================================================
# Fact Store Search Models
# =============================================================================


class SearchFactsRequest(BaseModel):
    """
    Request to search the fact store using vector similarity and/or keyword search.

    The fact store contains raw extracted entities and relationships before
    they are promoted to the knowledge graph. This is useful for RAG applications
    that don't need full graph traversal.

    Args:
        query: Search query text.
        group_id: Group to search within.
        node_types: Filter to specific entity types.
        limit: Maximum results to return.
        min_score: Minimum similarity score threshold (0.0-1.0).
        search_methods: Search methods to use ("vector", "keyword", or both).

    Example:
        >>> request = SearchFactsRequest(
        ...     query="hypertension treatment",
        ...     group_id="medical-kb",
        ...     limit=10,
        ...     min_score=0.7
        ... )
    """

    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(..., min_length=1)
    group_id: str | None = None
    node_types: list[str] | None = None
    limit: int = Field(default=10, ge=1, le=1000)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    search_methods: list[Literal["vector", "keyword"]] | None = None


class SearchFactsResults(BaseModel):
    """
    Results from searching the fact store.

    Contains extracted nodes and edges with their similarity scores.

    Args:
        success: Whether the search succeeded.
        nodes: Matching extracted nodes.
        edges: Matching extracted edges.
        node_scores: Similarity scores for each node (parallel to nodes list).
        edge_scores: Similarity scores for each edge (parallel to edges list).
        query: The original search query.
        total: Total number of results.

    Example:
        >>> results = client.search_facts("diabetes treatment", group_id="medical")
        >>> for node, score in zip(results.nodes, results.node_scores):
        ...     print(f"{node.name} ({node.type}): {score:.2f}")
    """

    model_config = ConfigDict(populate_by_name=True)

    success: bool = True
    nodes: list[ExtractedNode] = Field(default_factory=list)
    edges: list[ExtractedEdge] = Field(default_factory=list)
    node_scores: list[float] = Field(default_factory=list)
    edge_scores: list[float] = Field(default_factory=list)
    query: str = ""
    total: int = 0


# =============================================================================
# Server Configuration Models
# =============================================================================


class NLPModelConfig(BaseModel):
    """
    Configuration for a single NLP model.

    Args:
        provider: Model provider (e.g., "openai", "anthropic").
        model: Model name (e.g., "gpt-4o-mini").
        base_url: API base URL for the provider.
        temperature: Sampling temperature (0.0-2.0).
        max_tokens: Maximum tokens in response.

    Example:
        >>> config = client.get_models()
        >>> default_model = config.nlp.models.get("default")
        >>> print(f"Using {default_model.model} at {default_model.base_url}")
    """

    model_config = ConfigDict(populate_by_name=True)

    provider: str = ""
    model: str = ""
    base_url: str = Field(default="", alias="base_url")
    temperature: float = 0.0
    max_tokens: int = 0


class RouterRule(BaseModel):
    """
    A rule for routing NLP requests to different providers.

    Args:
        usage: Tag to match (e.g., "hipaa", "coding").
        provider: Provider ID to use.
        fallback: Fallback provider ID.
    """

    model_config = ConfigDict(populate_by_name=True)

    usage: str
    provider: str
    fallback: str = ""


class NLPModelsConfig(BaseModel):
    """
    NLP model configuration from the server.

    Args:
        models: Map of model name to configuration.
        router_rules: Rules for routing requests.

    Example:
        >>> config = client.get_models()
        >>> for name, model in config.nlp.models.items():
        ...     print(f"{name}: {model.provider}/{model.model}")
    """

    model_config = ConfigDict(populate_by_name=True)

    models: dict[str, NLPModelConfig] = Field(default_factory=dict)
    router_rules: list[RouterRule] = Field(default_factory=list)


class EmbeddingModelConfig(BaseModel):
    """
    Configuration for the embedding model.

    Args:
        provider: Embedding provider (e.g., "openai", "voyage").
        model: Model name (e.g., "text-embedding-3-small").
        base_url: API base URL for the provider.

    Example:
        >>> config = client.get_models()
        >>> print(f"Embeddings: {config.embedding.model}")
    """

    model_config = ConfigDict(populate_by_name=True)

    provider: str = ""
    model: str = ""
    base_url: str = Field(default="", alias="base_url")


class ModelsConfig(BaseModel):
    """
    Complete model configuration from the server.

    Contains both NLP (language model) and embedding model configurations.
    Use this to understand what models the server is using and to configure
    clients that need to interact with the same models.

    Args:
        nlp: NLP model configuration.
        embedding: Embedding model configuration.

    Example:
        >>> config = client.get_models()
        >>> print(f"NLP: {config.nlp.models}")
        >>> print(f"Embedding: {config.embedding.model}")
    """

    model_config = ConfigDict(populate_by_name=True)

    nlp: NLPModelsConfig = Field(default_factory=NLPModelsConfig)
    embedding: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)
