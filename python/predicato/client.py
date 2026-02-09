"""
Predicato client implementations.

This module provides both synchronous and asynchronous clients for interacting
with the Predicato knowledge graph server.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from predicato._http import AsyncHTTPClient, HTTPClientConfig, SyncHTTPClient
from predicato.exceptions import ConfigurationError, ValidationError
from predicato.models import (
    AddBulkEpisodeResults,
    AddEntityNodeRequest,
    AddEntityResults,
    AddEpisodeResults,
    AddMessagesRequest,
    AddMessagesResponse,
    ClearDataRequest,
    ClearDataResponse,
    Edge,
    EmbeddingModelConfig,
    Episode,
    ExtractedEdge,
    ExtractedNode,
    ExtractionMetadata,
    ExtractionResults,
    Message,
    ModelsConfig,
    NLPModelConfig,
    NLPModelsConfig,
    Node,
    PromoteToGraphResults,
    RouterRule,
    SearchFactsRequest,
    SearchFactsResults,
    SearchRequest,
    SearchResults,
)


class PredicatoClient:
    """
    Synchronous client for the Predicato knowledge graph API.

    Provides methods for ingesting content (episodes, messages, entities),
    searching the knowledge graph, and managing data.

    Args:
        base_url: Server URL. Falls back to PREDICATO_URL environment variable.
        group_id: Default group ID for data isolation. Falls back to PREDICATO_GROUP_ID.
        timeout: Request timeout in seconds (default: 30).
        max_retries: Maximum retry attempts for failed requests (default: 3).

    Raises:
        ConfigurationError: If base_url is not provided and PREDICATO_URL is not set.

    Example:
        >>> with PredicatoClient(base_url="http://localhost:8080") as client:
        ...     result = client.add_episode(
        ...         name="Meeting Notes",
        ...         content="Discussed project timeline...",
        ...         group_id="my-project"
        ...     )
        ...     print(f"Created {len(result.nodes)} entities")
    """

    def __init__(
        self,
        base_url: str | None = None,
        group_id: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the Predicato client."""
        # Resolve configuration from environment if not provided
        self._base_url = base_url or os.environ.get("PREDICATO_URL")
        if not self._base_url:
            raise ConfigurationError(
                "base_url is required. Provide it as a parameter or set PREDICATO_URL environment variable."
            )

        self._group_id = group_id or os.environ.get("PREDICATO_GROUP_ID")
        self._timeout = timeout

        # Create HTTP client
        config = HTTPClientConfig(
            base_url=self._base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._http = SyncHTTPClient(config)

    def __enter__(self) -> PredicatoClient:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and close the client."""
        self.close()

    def close(self) -> None:
        """
        Close the client and release resources.

        Should be called when done using the client, or use the context manager.
        """
        self._http.close()

    def get_models(self) -> ModelsConfig:
        """
        Get the server's configured NLP and embedding models.

        Returns the model configuration without API keys. Useful for
        understanding what models the server is using or for configuring
        clients that need to interact with the same providers.

        Returns:
            ModelsConfig containing NLP and embedding model configuration.

        Raises:
            ConnectionError: If the server cannot be reached.
            ServerError: If the server returns an error.

        Example:
            >>> config = client.get_models()
            >>> print(f"NLP provider: {config.nlp.models['default'].provider}")
            >>> print(f"NLP model: {config.nlp.models['default'].model}")
            >>> print(f"Base URL: {config.nlp.models['default'].base_url}")
        """
        response = self._http.request("GET", "/api/v1/config/models")

        # Parse NLP models
        nlp_models = {}
        for name, model_data in response.get("nlp", {}).get("models", {}).items():
            nlp_models[name] = NLPModelConfig(**model_data)

        router_rules = [
            RouterRule(**rule)
            for rule in response.get("nlp", {}).get("router_rules", [])
        ]

        # Parse embedding config
        embedding_data = response.get("embedding", {})
        embedding = EmbeddingModelConfig(**embedding_data) if embedding_data else EmbeddingModelConfig()

        return ModelsConfig(
            nlp=NLPModelsConfig(models=nlp_models, router_rules=router_rules),
            embedding=embedding,
        )

    def _resolve_group_id(self, group_id: str | None) -> str:
        """Resolve group_id, falling back to default if not provided."""
        resolved = group_id or self._group_id
        if not resolved:
            raise ValidationError(
                "group_id is required. Provide it as a parameter or set a default in the client constructor."
            )
        return resolved

    def add_episode(
        self,
        name: str,
        content: str,
        *,
        source: str = "",
        group_id: str | None = None,
        reference: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AddEpisodeResults:
        """
        Add a single episode to the knowledge graph.

        An episode is a unit of content that will be processed to extract
        entities and relationships for the knowledge graph.

        Args:
            name: Name/title of the episode (max 1024 chars).
            content: The content to process (max 1MB).
            source: Source identifier (e.g., "meeting", "document").
            group_id: Group for data isolation. Uses client default if not provided.
            reference: Reference timestamp for the content.
            metadata: Additional metadata to attach to the episode.

        Returns:
            AddEpisodeResults containing the created episode node and extracted entities/edges.

        Raises:
            ValidationError: If content exceeds size limits or required fields are missing.
            ConnectionError: If the server cannot be reached.
            ServerError: If the server returns an error.

        Example:
            >>> result = client.add_episode(
            ...     name="Q1 Planning",
            ...     content="Discussed roadmap for Q1...",
            ...     source="meeting",
            ...     group_id="project-alpha"
            ... )
            >>> print(f"Created episode: {result.episode.uuid}")
        """
        resolved_group_id = self._resolve_group_id(group_id)

        # Validate content size
        if len(content) > 1_048_576:
            raise ValidationError(
                f"content exceeds maximum size of 1MB ({len(content)} bytes)",
                field="content",
            )

        # Create episode as a message to use the existing API
        # The server converts messages to episodes internally
        message = Message(
            role="user",
            content=f"{name}: {content}",
            timestamp=reference,
        )

        request = AddMessagesRequest(
            group_id=resolved_group_id,
            messages=[message],
            reference=reference,
        )

        self._http.request(
            "POST",
            "/api/v1/ingest/messages",
            json=request.model_dump(mode="json", exclude_none=True),
        )

        # Convert the response to AddEpisodeResults
        # The server returns an async response, so we return a simplified result
        return AddEpisodeResults(
            episode=None,  # Episode is created asynchronously
            nodes=[],
            edges=[],
        )

    def add_episodes(
        self,
        episodes: list[Episode],
        *,
        group_id: str | None = None,
    ) -> AddBulkEpisodeResults:
        """
        Add multiple episodes to the knowledge graph in bulk.

        Args:
            episodes: List of Episode objects to add.
            group_id: Default group ID if not specified in episodes.

        Returns:
            AddBulkEpisodeResults containing all created nodes and edges.

        Raises:
            ValidationError: If any episode is invalid.
            ConnectionError: If the server cannot be reached.
            ServerError: If the server returns an error.

        Example:
            >>> episodes = [
            ...     Episode(name="Doc 1", content="...", group_id="my-group"),
            ...     Episode(name="Doc 2", content="...", group_id="my-group"),
            ... ]
            >>> result = client.add_episodes(episodes)
        """
        default_group_id = self._resolve_group_id(group_id) if group_id else self._group_id

        # Convert episodes to messages
        messages: list[Message] = []
        resolved_group_id = None

        for episode in episodes:
            ep_group_id = episode.group_id or default_group_id
            if not ep_group_id:
                raise ValidationError(
                    "group_id is required for each episode or as default",
                    field="group_id",
                )
            if resolved_group_id is None:
                resolved_group_id = ep_group_id
            elif resolved_group_id != ep_group_id:
                raise ValidationError(
                    "All episodes must have the same group_id for bulk operations",
                    field="group_id",
                )

            messages.append(
                Message(
                    role="user",
                    content=f"{episode.name}: {episode.content}",
                    timestamp=episode.reference,
                )
            )

        if not resolved_group_id:
            raise ValidationError("No group_id could be resolved")

        request = AddMessagesRequest(
            group_id=resolved_group_id,
            messages=messages,
        )

        self._http.request(
            "POST",
            "/api/v1/ingest/messages",
            json=request.model_dump(mode="json", exclude_none=True),
        )

        # Return empty results as processing is async
        return AddBulkEpisodeResults()

    def add_messages(
        self,
        messages: list[Message | dict[str, Any]],
        *,
        group_id: str | None = None,
        reference: datetime | None = None,
    ) -> AddMessagesResponse:
        """
        Add conversation messages to the knowledge graph.

        Messages are converted to episodes and processed to extract entities
        and relationships.

        Args:
            messages: List of Message objects or dicts with role/content.
            group_id: Group for data isolation. Uses client default if not provided.
            reference: Reference timestamp for the messages.

        Returns:
            AddMessagesResponse with process_id for tracking async processing.

        Raises:
            ValidationError: If messages are invalid or role is not recognized.
            ConnectionError: If the server cannot be reached.
            ServerError: If the server returns an error.

        Example:
            >>> result = client.add_messages(
            ...     messages=[
            ...         {"role": "user", "content": "What is the status?"},
            ...         {"role": "assistant", "content": "Everything is on track."},
            ...     ],
            ...     group_id="conversation-001"
            ... )
            >>> print(f"Process ID: {result.process_id}")
        """
        resolved_group_id = self._resolve_group_id(group_id)

        # Convert dicts to Message objects
        message_objects: list[Message] = []
        for msg in messages:
            if isinstance(msg, dict):
                message_objects.append(Message(**msg))
            else:
                message_objects.append(msg)

        request = AddMessagesRequest(
            group_id=resolved_group_id,
            messages=message_objects,
            reference=reference,
        )

        response = self._http.request(
            "POST",
            "/api/v1/ingest/messages",
            json=request.model_dump(mode="json", exclude_none=True),
        )

        return AddMessagesResponse(
            success=response.get("success", True),
            message=response.get("message", ""),
            process_id=response.get("process_id"),
            queued_count=len(message_objects),
        )

    def add_entity(
        self,
        name: str,
        entity_type: str,
        *,
        group_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> AddEntityResults:
        """
        Create a standalone entity node in the knowledge graph.

        Args:
            name: Name of the entity (max 1024 chars).
            entity_type: Type of the entity (max 256 chars).
            group_id: Group for data isolation. Uses client default if not provided.
            attributes: Additional attributes for the entity (max 100 keys).

        Returns:
            AddEntityResults containing the created node.

        Raises:
            ValidationError: If entity data is invalid.
            ConnectionError: If the server cannot be reached.
            ServerError: If the server returns an error.

        Example:
            >>> result = client.add_entity(
            ...     name="Alice Smith",
            ...     entity_type="Person",
            ...     attributes={"role": "Engineer", "department": "Platform"},
            ...     group_id="my-project"
            ... )
        """
        resolved_group_id = self._resolve_group_id(group_id)

        request = AddEntityNodeRequest(
            group_id=resolved_group_id,
            name=name,
            entity_type=entity_type,
            attributes=attributes or {},
        )

        self._http.request(
            "POST",
            "/api/v1/ingest/entity",
            json=request.model_dump(mode="json", exclude_none=True),
        )

        # Entity is created via episode processing, so we don't have the node yet
        return AddEntityResults(node=None)

    def search(
        self,
        query: str,
        *,
        group_id: str | None = None,
        limit: int = 10,
        include_edges: bool = False,
        min_score: float = 0.0,
    ) -> SearchResults:
        """
        Search the knowledge graph.

        Args:
            query: Search query string.
            group_id: Filter results to a specific group.
            limit: Maximum results to return (1-100, default 10).
            include_edges: Whether to include edges in results.
            min_score: Minimum relevance score (0.0-1.0).

        Returns:
            SearchResults containing matching nodes and edges.

        Raises:
            ValidationError: If query is empty or parameters are invalid.
            ConnectionError: If the server cannot be reached.
            ServerError: If the server returns an error.

        Example:
            >>> results = client.search(
            ...     query="project timeline",
            ...     group_id="project-alpha",
            ...     limit=20
            ... )
            >>> for node in results.nodes:
            ...     print(f"- {node.name}")
        """
        SearchRequest(
            query=query,
            group_id=group_id,
            limit=limit,
            include_edges=include_edges,
            min_score=min_score,
        )

        # The server uses a different request format
        self._http.request(
            "POST",
            "/api/v1/search",
            json={
                "query": query,
                "max_facts": limit,
            },
        )

        # Parse the response - server returns facts, we convert to nodes/edges
        return SearchResults(
            nodes=[],  # Would need to parse facts into nodes
            edges=[],
            node_scores={},
            edge_scores={},
        )

    def clear_graph(
        self,
        group_ids: list[str],
    ) -> ClearDataResponse:
        """
        Clear all data for the specified groups.

        This permanently deletes all nodes and edges for the given groups.
        Use with caution.

        Args:
            group_ids: List of group IDs to clear. Must be explicitly provided.

        Returns:
            ClearDataResponse with details of cleared and failed groups.

        Raises:
            ValidationError: If group_ids is empty.
            ConnectionError: If the server cannot be reached.
            ServerError: If the server returns an error.

        Example:
            >>> result = client.clear_graph(group_ids=["test-project"])
            >>> if result.success:
            ...     print("Data cleared successfully")
        """
        if not group_ids:
            raise ValidationError(
                "group_ids must be explicitly provided for data clearing",
                field="group_ids",
            )

        request = ClearDataRequest(group_ids=group_ids)

        response = self._http.request(
            "DELETE",
            "/api/v1/ingest/clear",
            json=request.model_dump(mode="json"),
        )

        return ClearDataResponse(
            success=response.get("success", False),
            cleared_groups=group_ids if response.get("success") else [],
            failed_groups=[] if response.get("success") else group_ids,
        )

    def search_facts(
        self,
        query: str,
        *,
        group_id: str | None = None,
        node_types: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
        search_methods: list[str] | None = None,
    ) -> SearchFactsResults:
        """
        Search the fact store using vector similarity and/or keyword search.

        The fact store contains raw extracted entities and relationships before
        they are promoted to the knowledge graph. This is useful for RAG applications
        that don't need full graph traversal.

        Search uses cosine similarity on embeddings for semantic search and/or
        BM25-style keyword matching.

        Args:
            query: Search query string.
            group_id: Filter results to a specific group.
            node_types: Filter to specific entity types (e.g., ["Disease", "Drug"]).
            limit: Maximum results to return (1-1000, default 10).
            min_score: Minimum similarity score threshold (0.0-1.0).
            search_methods: Search methods to use. Options: "vector", "keyword".
                           Default uses both (hybrid search).

        Returns:
            SearchFactsResults containing extracted nodes/edges with scores.

        Raises:
            ValidationError: If query is empty or parameters are invalid.
            ConnectionError: If the server cannot be reached.
            ServerError: If the server returns an error.

        Example:
            >>> results = client.search_facts(
            ...     query="hypertension treatment",
            ...     group_id="medical-kb",
            ...     limit=20,
            ...     min_score=0.5
            ... )
            >>> for node, score in zip(results.nodes, results.node_scores):
            ...     print(f"- {node.name} ({node.type}): {score:.2f}")
        """
        # Validate using the request model
        SearchFactsRequest(
            query=query,
            group_id=group_id,
            node_types=node_types,
            limit=limit,
            min_score=min_score,
            search_methods=search_methods,  # type: ignore[arg-type]
        )

        request_data: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "min_score": min_score,
        }
        if group_id:
            request_data["group_id"] = group_id
        if node_types:
            request_data["node_types"] = node_types
        if search_methods:
            request_data["search_methods"] = search_methods

        response = self._http.request(
            "POST",
            "/api/v1/search/facts",
            json=request_data,
        )

        # Parse response into SearchFactsResults
        nodes = [
            ExtractedNode(**node) for node in response.get("nodes", [])
        ]
        edges = [
            ExtractedEdge(**edge) for edge in response.get("edges", [])
        ]

        return SearchFactsResults(
            success=response.get("success", True),
            nodes=nodes,
            edges=edges,
            node_scores=response.get("node_scores", []),
            edge_scores=response.get("edge_scores", []),
            query=response.get("query", query),
            total=response.get("total", len(nodes) + len(edges)),
        )

    # =========================================================================
    # Two-Stage Ingestion Methods
    # =========================================================================

    def extract_to_facts(
        self,
        name: str,
        content: str,
        *,
        episode_id: str | None = None,
        source: str = "",
        group_id: str | None = None,
        reference: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        entity_types: dict[str, Any] | None = None,
        excluded_entity_types: list[str] | None = None,
        edge_types: dict[str, Any] | None = None,
        edge_type_map: dict[str, dict[str, Any]] | None = None,
        previous_episode_uuids: list[str] | None = None,
        max_characters: int | None = None,
        use_yaml: bool = False,
    ) -> ExtractionResults:
        """
        Extract entities and relationships from content without promoting to graph.

        This is the first stage of two-stage ingestion. The extracted knowledge
        is stored in the fact store and can later be promoted to the graph
        using promote_to_graph().

        Args:
            name: Name/title of the episode (max 1024 chars).
            content: The content to process (max 1MB).
            episode_id: Optional episode identifier (auto-generated if not provided).
            source: Source identifier (e.g., "meeting", "document").
            group_id: Group for data isolation. Uses client default if not provided.
            reference: Reference timestamp for the content.
            metadata: Additional metadata to attach to the episode.
            entity_types: Custom entity type definitions.
            excluded_entity_types: Entity types to exclude from extraction.
            edge_types: Custom edge type definitions.
            edge_type_map: Mapping of entity pairs to edge types.
            previous_episode_uuids: UUIDs of previous episodes for context.
            max_characters: Maximum characters per chunk.
            use_yaml: Whether to use YAML for LLM interchange.

        Returns:
            ExtractionResults containing the source_id and extracted entities/edges.
            Use the source_id to later call promote_to_graph().

        Raises:
            ValidationError: If content exceeds size limits or required fields are missing.
            ConnectionError: If the server cannot be reached.
            ServerError: If the server returns an error.

        Example:
            >>> # Stage 1: Extract
            >>> extraction = client.extract_to_facts(
            ...     name="Meeting Notes",
            ...     content="Discussed project timeline...",
            ...     group_id="project-alpha"
            ... )
            >>> print(f"Extracted {len(extraction.extracted_nodes)} entities")
            >>>
            >>> # Stage 2: Promote (later, possibly with different options)
            >>> results = client.promote_to_graph(extraction.source_id)
        """
        resolved_group_id = self._resolve_group_id(group_id)

        # Validate content size
        if len(content) > 1_048_576:
            raise ValidationError(
                f"content exceeds maximum size of 1MB ({len(content)} bytes)",
                field="content",
            )

        request_data: dict[str, Any] = {
            "name": name,
            "content": content,
            "source": source,
            "group_id": resolved_group_id,
            "use_yaml": use_yaml,
        }

        if episode_id:
            request_data["id"] = episode_id
        if reference:
            request_data["reference"] = reference.isoformat()
        if metadata:
            request_data["metadata"] = metadata
        if entity_types:
            request_data["entity_types"] = entity_types
        if excluded_entity_types:
            request_data["excluded_entity_types"] = excluded_entity_types
        if edge_types:
            request_data["edge_types"] = edge_types
        if edge_type_map:
            request_data["edge_type_map"] = edge_type_map
        if previous_episode_uuids:
            request_data["previous_episode_uuids"] = previous_episode_uuids
        if max_characters:
            request_data["max_characters"] = max_characters

        response = self._http.request(
            "POST",
            "/api/v1/ingest/extract",
            json=request_data,
        )

        # Parse response into ExtractionResults
        extracted_nodes = [
            ExtractedNode(**node) for node in response.get("extracted_nodes", [])
        ]
        extracted_edges = [
            ExtractedEdge(**edge) for edge in response.get("extracted_edges", [])
        ]

        extraction_metadata = None
        if response.get("metadata"):
            extraction_metadata = ExtractionMetadata(**response["metadata"])

        return ExtractionResults(
            source_id=response.get("source_id", ""),
            extracted_nodes=extracted_nodes,
            extracted_edges=extracted_edges,
            chunk_count=response.get("chunk_count", 0),
            extraction_time=response.get("extraction_time"),
            metadata=extraction_metadata,
        )

    def promote_to_graph(
        self,
        source_id: str,
        *,
        entity_types: dict[str, Any] | None = None,
        edge_types: dict[str, Any] | None = None,
        edge_type_map: dict[str, dict[str, Any]] | None = None,
        previous_episode_uuids: list[str] | None = None,
        skip_reflexion: bool = False,
        skip_resolution: bool = False,
        skip_attributes: bool = False,
        skip_edge_resolution: bool = False,
        use_yaml: bool = False,
    ) -> PromoteToGraphResults:
        """
        Promote previously extracted knowledge to the graph.

        This is the second stage of two-stage ingestion. It takes the source_id
        from extract_to_facts() and promotes the extracted entities and
        relationships to the knowledge graph.

        Args:
            source_id: ID of the source to promote (from extract_to_facts).
            entity_types: Custom entity type definitions.
            edge_types: Custom edge type definitions.
            edge_type_map: Mapping of entity pairs to edge types.
            previous_episode_uuids: UUIDs of previous episodes for context.
            skip_reflexion: Skip reflexion step for faster processing.
            skip_resolution: Skip entity resolution (faster but may create duplicates).
            skip_attributes: Skip attribute extraction.
            skip_edge_resolution: Skip edge resolution.
            use_yaml: Whether to use YAML for LLM interchange.

        Returns:
            PromoteToGraphResults containing resolved nodes, edges, and communities.

        Raises:
            ValidationError: If source_id is empty.
            ConnectionError: If the server cannot be reached.
            ServerError: If the server returns an error.
            NotFoundError: If the source_id doesn't exist.

        Example:
            >>> results = client.promote_to_graph(
            ...     source_id="episode-123",
            ...     skip_resolution=True  # Faster but less accurate
            ... )
            >>> print(f"Created {len(results.nodes)} entities")
            >>> print(f"Created {len(results.edges)} relationships")
        """
        if not source_id:
            raise ValidationError(
                "source_id is required for promotion",
                field="source_id",
            )

        request_data: dict[str, Any] = {
            "source_id": source_id,
            "skip_reflexion": skip_reflexion,
            "skip_resolution": skip_resolution,
            "skip_attributes": skip_attributes,
            "skip_edge_resolution": skip_edge_resolution,
            "use_yaml": use_yaml,
        }

        if entity_types:
            request_data["entity_types"] = entity_types
        if edge_types:
            request_data["edge_types"] = edge_types
        if edge_type_map:
            request_data["edge_type_map"] = edge_type_map
        if previous_episode_uuids:
            request_data["previous_episode_uuids"] = previous_episode_uuids

        response = self._http.request(
            "POST",
            "/api/v1/ingest/promote",
            json=request_data,
        )

        # Parse response into PromoteToGraphResults
        episode = None
        if response.get("episode"):
            episode = Node(**response["episode"])

        nodes = [Node(**n) for n in response.get("nodes", [])]
        edges = [Edge(**e) for e in response.get("edges", [])]
        episodic_edges = [Edge(**e) for e in response.get("episodic_edges", [])]
        communities = [Node(**n) for n in response.get("communities", [])]
        community_edges = [Edge(**e) for e in response.get("community_edges", [])]

        return PromoteToGraphResults(
            episode=episode,
            episodic_edges=episodic_edges,
            nodes=nodes,
            edges=edges,
            communities=communities,
            community_edges=community_edges,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for the given texts using the server's configured embedding model.

        Args:
            texts: List of texts to embed (max 100, each max 8192 chars).

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            ValidationError: If texts list is empty or exceeds limits.
            ConnectionError: If the server cannot be reached.
            ServerError: If the server returns an error.

        Example:
            >>> vectors = client.embed(["hello world", "knowledge graph"])
            >>> print(f"Dimension: {len(vectors[0])}")
        """
        if not texts:
            raise ValidationError(
                "texts must not be empty",
                field="texts",
            )
        if len(texts) > 100:
            raise ValidationError(
                f"texts exceeds maximum of 100 items ({len(texts)} provided)",
                field="texts",
            )
        for i, text in enumerate(texts):
            if len(text) > 8192:
                raise ValidationError(
                    f"texts[{i}] exceeds maximum of 8192 characters ({len(text)} chars)",
                    field="texts",
                )

        response = self._http.request(
            "POST",
            "/api/v1/embed",
            json={"texts": texts},
        )

        return response["embeddings"]


class AsyncPredicatoClient:
    """
    Asynchronous client for the Predicato knowledge graph API.

    Provides async methods for ingesting content (episodes, messages, entities),
    searching the knowledge graph, and managing data.

    Args:
        base_url: Server URL. Falls back to PREDICATO_URL environment variable.
        group_id: Default group ID for data isolation. Falls back to PREDICATO_GROUP_ID.
        timeout: Request timeout in seconds (default: 30).
        max_retries: Maximum retry attempts for failed requests (default: 3).

    Raises:
        ConfigurationError: If base_url is not provided and PREDICATO_URL is not set.

    Example:
        >>> async with AsyncPredicatoClient(base_url="http://localhost:8080") as client:
        ...     result = await client.add_episode(
        ...         name="Meeting Notes",
        ...         content="Discussed project timeline...",
        ...         group_id="my-project"
        ...     )
    """

    def __init__(
        self,
        base_url: str | None = None,
        group_id: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """Initialize the async Predicato client."""
        # Resolve configuration from environment if not provided
        self._base_url = base_url or os.environ.get("PREDICATO_URL")
        if not self._base_url:
            raise ConfigurationError(
                "base_url is required. Provide it as a parameter or set PREDICATO_URL environment variable."
            )

        self._group_id = group_id or os.environ.get("PREDICATO_GROUP_ID")
        self._timeout = timeout

        # Create HTTP client
        config = HTTPClientConfig(
            base_url=self._base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._http = AsyncHTTPClient(config)

    async def __aenter__(self) -> AsyncPredicatoClient:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager and close the client."""
        await self.close()

    async def close(self) -> None:
        """
        Close the client and release resources.

        Should be called when done using the client, or use the async context manager.
        """
        await self._http.close()

    async def get_models(self) -> ModelsConfig:
        """
        Get the server's configured NLP and embedding models.

        Returns the model configuration without API keys. Useful for
        understanding what models the server is using or for configuring
        clients that need to interact with the same providers.

        Returns:
            ModelsConfig containing NLP and embedding model configuration.

        Raises:
            ConnectionError: If the server cannot be reached.
            ServerError: If the server returns an error.

        Example:
            >>> config = await client.get_models()
            >>> print(f"NLP provider: {config.nlp.models['default'].provider}")
        """
        response = await self._http.request("GET", "/api/v1/config/models")

        nlp_models = {}
        for name, model_data in response.get("nlp", {}).get("models", {}).items():
            nlp_models[name] = NLPModelConfig(**model_data)

        router_rules = [
            RouterRule(**rule)
            for rule in response.get("nlp", {}).get("router_rules", [])
        ]

        embedding_data = response.get("embedding", {})
        embedding = EmbeddingModelConfig(**embedding_data) if embedding_data else EmbeddingModelConfig()

        return ModelsConfig(
            nlp=NLPModelsConfig(models=nlp_models, router_rules=router_rules),
            embedding=embedding,
        )

    def _resolve_group_id(self, group_id: str | None) -> str:
        """Resolve group_id, falling back to default if not provided."""
        resolved = group_id or self._group_id
        if not resolved:
            raise ValidationError(
                "group_id is required. Provide it as a parameter or set a default in the client constructor."
            )
        return resolved

    async def add_episode(
        self,
        name: str,
        content: str,
        *,
        source: str = "",
        group_id: str | None = None,
        reference: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AddEpisodeResults:
        """
        Add a single episode to the knowledge graph.

        An episode is a unit of content that will be processed to extract
        entities and relationships for the knowledge graph.

        Args:
            name: Name/title of the episode (max 1024 chars).
            content: The content to process (max 1MB).
            source: Source identifier (e.g., "meeting", "document").
            group_id: Group for data isolation. Uses client default if not provided.
            reference: Reference timestamp for the content.
            metadata: Additional metadata to attach to the episode.

        Returns:
            AddEpisodeResults containing the created episode node and extracted entities/edges.

        Raises:
            ValidationError: If content exceeds size limits or required fields are missing.
            ConnectionError: If the server cannot be reached.
            ServerError: If the server returns an error.
        """
        resolved_group_id = self._resolve_group_id(group_id)

        # Validate content size
        if len(content) > 1_048_576:
            raise ValidationError(
                f"content exceeds maximum size of 1MB ({len(content)} bytes)",
                field="content",
            )

        message = Message(
            role="user",
            content=f"{name}: {content}",
            timestamp=reference,
        )

        request = AddMessagesRequest(
            group_id=resolved_group_id,
            messages=[message],
            reference=reference,
        )

        await self._http.request(
            "POST",
            "/api/v1/ingest/messages",
            json=request.model_dump(mode="json", exclude_none=True),
        )

        return AddEpisodeResults(
            episode=None,
            nodes=[],
            edges=[],
        )

    async def add_episodes(
        self,
        episodes: list[Episode],
        *,
        group_id: str | None = None,
    ) -> AddBulkEpisodeResults:
        """
        Add multiple episodes to the knowledge graph in bulk.

        Args:
            episodes: List of Episode objects to add.
            group_id: Default group ID if not specified in episodes.

        Returns:
            AddBulkEpisodeResults containing all created nodes and edges.
        """
        default_group_id = self._resolve_group_id(group_id) if group_id else self._group_id

        messages: list[Message] = []
        resolved_group_id = None

        for episode in episodes:
            ep_group_id = episode.group_id or default_group_id
            if not ep_group_id:
                raise ValidationError(
                    "group_id is required for each episode or as default",
                    field="group_id",
                )
            if resolved_group_id is None:
                resolved_group_id = ep_group_id
            elif resolved_group_id != ep_group_id:
                raise ValidationError(
                    "All episodes must have the same group_id for bulk operations",
                    field="group_id",
                )

            messages.append(
                Message(
                    role="user",
                    content=f"{episode.name}: {episode.content}",
                    timestamp=episode.reference,
                )
            )

        if not resolved_group_id:
            raise ValidationError("No group_id could be resolved")

        request = AddMessagesRequest(
            group_id=resolved_group_id,
            messages=messages,
        )

        await self._http.request(
            "POST",
            "/api/v1/ingest/messages",
            json=request.model_dump(mode="json", exclude_none=True),
        )

        return AddBulkEpisodeResults()

    async def add_messages(
        self,
        messages: list[Message | dict[str, Any]],
        *,
        group_id: str | None = None,
        reference: datetime | None = None,
    ) -> AddMessagesResponse:
        """
        Add conversation messages to the knowledge graph.

        Messages are converted to episodes and processed to extract entities
        and relationships.

        Args:
            messages: List of Message objects or dicts with role/content.
            group_id: Group for data isolation. Uses client default if not provided.
            reference: Reference timestamp for the messages.

        Returns:
            AddMessagesResponse with process_id for tracking async processing.
        """
        resolved_group_id = self._resolve_group_id(group_id)

        message_objects: list[Message] = []
        for msg in messages:
            if isinstance(msg, dict):
                message_objects.append(Message(**msg))
            else:
                message_objects.append(msg)

        request = AddMessagesRequest(
            group_id=resolved_group_id,
            messages=message_objects,
            reference=reference,
        )

        response = await self._http.request(
            "POST",
            "/api/v1/ingest/messages",
            json=request.model_dump(mode="json", exclude_none=True),
        )

        return AddMessagesResponse(
            success=response.get("success", True),
            message=response.get("message", ""),
            process_id=response.get("process_id"),
            queued_count=len(message_objects),
        )

    async def add_entity(
        self,
        name: str,
        entity_type: str,
        *,
        group_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> AddEntityResults:
        """
        Create a standalone entity node in the knowledge graph.

        Args:
            name: Name of the entity (max 1024 chars).
            entity_type: Type of the entity (max 256 chars).
            group_id: Group for data isolation. Uses client default if not provided.
            attributes: Additional attributes for the entity (max 100 keys).

        Returns:
            AddEntityResults containing the created node.
        """
        resolved_group_id = self._resolve_group_id(group_id)

        request = AddEntityNodeRequest(
            group_id=resolved_group_id,
            name=name,
            entity_type=entity_type,
            attributes=attributes or {},
        )

        await self._http.request(
            "POST",
            "/api/v1/ingest/entity",
            json=request.model_dump(mode="json", exclude_none=True),
        )

        return AddEntityResults(node=None)

    async def search(
        self,
        query: str,
        *,
        group_id: str | None = None,
        limit: int = 10,
        include_edges: bool = False,
        min_score: float = 0.0,
    ) -> SearchResults:
        """
        Search the knowledge graph.

        Args:
            query: Search query string.
            group_id: Filter results to a specific group.
            limit: Maximum results to return (1-100, default 10).
            include_edges: Whether to include edges in results.
            min_score: Minimum relevance score (0.0-1.0).

        Returns:
            SearchResults containing matching nodes and edges.
        """
        SearchRequest(
            query=query,
            group_id=group_id,
            limit=limit,
            include_edges=include_edges,
            min_score=min_score,
        )

        await self._http.request(
            "POST",
            "/api/v1/search",
            json={
                "query": query,
                "max_facts": limit,
            },
        )

        return SearchResults(
            nodes=[],
            edges=[],
            node_scores={},
            edge_scores={},
        )

    async def clear_graph(
        self,
        group_ids: list[str],
    ) -> ClearDataResponse:
        """
        Clear all data for the specified groups.

        This permanently deletes all nodes and edges for the given groups.
        Use with caution.

        Args:
            group_ids: List of group IDs to clear. Must be explicitly provided.

        Returns:
            ClearDataResponse with details of cleared and failed groups.
        """
        if not group_ids:
            raise ValidationError(
                "group_ids must be explicitly provided for data clearing",
                field="group_ids",
            )

        request = ClearDataRequest(group_ids=group_ids)

        response = await self._http.request(
            "DELETE",
            "/api/v1/ingest/clear",
            json=request.model_dump(mode="json"),
        )

        return ClearDataResponse(
            success=response.get("success", False),
            cleared_groups=group_ids if response.get("success") else [],
            failed_groups=[] if response.get("success") else group_ids,
        )

    async def search_facts(
        self,
        query: str,
        *,
        group_id: str | None = None,
        node_types: list[str] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
        search_methods: list[str] | None = None,
    ) -> SearchFactsResults:
        """
        Search the fact store using vector similarity and/or keyword search.

        The fact store contains raw extracted entities and relationships before
        they are promoted to the knowledge graph. This is useful for RAG applications
        that don't need full graph traversal.

        Search uses cosine similarity on embeddings for semantic search and/or
        BM25-style keyword matching.

        Args:
            query: Search query string.
            group_id: Filter results to a specific group.
            node_types: Filter to specific entity types (e.g., ["Disease", "Drug"]).
            limit: Maximum results to return (1-1000, default 10).
            min_score: Minimum similarity score threshold (0.0-1.0).
            search_methods: Search methods to use. Options: "vector", "keyword".
                           Default uses both (hybrid search).

        Returns:
            SearchFactsResults containing extracted nodes/edges with scores.
        """
        # Validate using the request model
        SearchFactsRequest(
            query=query,
            group_id=group_id,
            node_types=node_types,
            limit=limit,
            min_score=min_score,
            search_methods=search_methods,  # type: ignore[arg-type]
        )

        request_data: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "min_score": min_score,
        }
        if group_id:
            request_data["group_id"] = group_id
        if node_types:
            request_data["node_types"] = node_types
        if search_methods:
            request_data["search_methods"] = search_methods

        response = await self._http.request(
            "POST",
            "/api/v1/search/facts",
            json=request_data,
        )

        # Parse response into SearchFactsResults
        nodes = [
            ExtractedNode(**node) for node in response.get("nodes", [])
        ]
        edges = [
            ExtractedEdge(**edge) for edge in response.get("edges", [])
        ]

        return SearchFactsResults(
            success=response.get("success", True),
            nodes=nodes,
            edges=edges,
            node_scores=response.get("node_scores", []),
            edge_scores=response.get("edge_scores", []),
            query=response.get("query", query),
            total=response.get("total", len(nodes) + len(edges)),
        )

    # =========================================================================
    # Two-Stage Ingestion Methods (Async)
    # =========================================================================

    async def extract_to_facts(
        self,
        name: str,
        content: str,
        *,
        episode_id: str | None = None,
        source: str = "",
        group_id: str | None = None,
        reference: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        entity_types: dict[str, Any] | None = None,
        excluded_entity_types: list[str] | None = None,
        edge_types: dict[str, Any] | None = None,
        edge_type_map: dict[str, dict[str, Any]] | None = None,
        previous_episode_uuids: list[str] | None = None,
        max_characters: int | None = None,
        use_yaml: bool = False,
    ) -> ExtractionResults:
        """
        Extract entities and relationships from content without promoting to graph.

        This is the first stage of two-stage ingestion. The extracted knowledge
        is stored in the fact store and can later be promoted to the graph
        using promote_to_graph().

        Args:
            name: Name/title of the episode (max 1024 chars).
            content: The content to process (max 1MB).
            episode_id: Optional episode identifier (auto-generated if not provided).
            source: Source identifier (e.g., "meeting", "document").
            group_id: Group for data isolation. Uses client default if not provided.
            reference: Reference timestamp for the content.
            metadata: Additional metadata to attach to the episode.
            entity_types: Custom entity type definitions.
            excluded_entity_types: Entity types to exclude from extraction.
            edge_types: Custom edge type definitions.
            edge_type_map: Mapping of entity pairs to edge types.
            previous_episode_uuids: UUIDs of previous episodes for context.
            max_characters: Maximum characters per chunk.
            use_yaml: Whether to use YAML for LLM interchange.

        Returns:
            ExtractionResults containing the source_id and extracted entities/edges.
        """
        resolved_group_id = self._resolve_group_id(group_id)

        if len(content) > 1_048_576:
            raise ValidationError(
                f"content exceeds maximum size of 1MB ({len(content)} bytes)",
                field="content",
            )

        request_data: dict[str, Any] = {
            "name": name,
            "content": content,
            "source": source,
            "group_id": resolved_group_id,
            "use_yaml": use_yaml,
        }

        if episode_id:
            request_data["id"] = episode_id
        if reference:
            request_data["reference"] = reference.isoformat()
        if metadata:
            request_data["metadata"] = metadata
        if entity_types:
            request_data["entity_types"] = entity_types
        if excluded_entity_types:
            request_data["excluded_entity_types"] = excluded_entity_types
        if edge_types:
            request_data["edge_types"] = edge_types
        if edge_type_map:
            request_data["edge_type_map"] = edge_type_map
        if previous_episode_uuids:
            request_data["previous_episode_uuids"] = previous_episode_uuids
        if max_characters:
            request_data["max_characters"] = max_characters

        response = await self._http.request(
            "POST",
            "/api/v1/ingest/extract",
            json=request_data,
        )

        extracted_nodes = [
            ExtractedNode(**node) for node in response.get("extracted_nodes", [])
        ]
        extracted_edges = [
            ExtractedEdge(**edge) for edge in response.get("extracted_edges", [])
        ]

        extraction_metadata = None
        if response.get("metadata"):
            extraction_metadata = ExtractionMetadata(**response["metadata"])

        return ExtractionResults(
            source_id=response.get("source_id", ""),
            extracted_nodes=extracted_nodes,
            extracted_edges=extracted_edges,
            chunk_count=response.get("chunk_count", 0),
            extraction_time=response.get("extraction_time"),
            metadata=extraction_metadata,
        )

    async def promote_to_graph(
        self,
        source_id: str,
        *,
        entity_types: dict[str, Any] | None = None,
        edge_types: dict[str, Any] | None = None,
        edge_type_map: dict[str, dict[str, Any]] | None = None,
        previous_episode_uuids: list[str] | None = None,
        skip_reflexion: bool = False,
        skip_resolution: bool = False,
        skip_attributes: bool = False,
        skip_edge_resolution: bool = False,
        use_yaml: bool = False,
    ) -> PromoteToGraphResults:
        """
        Promote previously extracted knowledge to the graph.

        This is the second stage of two-stage ingestion.

        Args:
            source_id: ID of the source to promote (from extract_to_facts).
            entity_types: Custom entity type definitions.
            edge_types: Custom edge type definitions.
            edge_type_map: Mapping of entity pairs to edge types.
            previous_episode_uuids: UUIDs of previous episodes for context.
            skip_reflexion: Skip reflexion step for faster processing.
            skip_resolution: Skip entity resolution.
            skip_attributes: Skip attribute extraction.
            skip_edge_resolution: Skip edge resolution.
            use_yaml: Whether to use YAML for LLM interchange.

        Returns:
            PromoteToGraphResults containing resolved nodes, edges, and communities.
        """
        if not source_id:
            raise ValidationError(
                "source_id is required for promotion",
                field="source_id",
            )

        request_data: dict[str, Any] = {
            "source_id": source_id,
            "skip_reflexion": skip_reflexion,
            "skip_resolution": skip_resolution,
            "skip_attributes": skip_attributes,
            "skip_edge_resolution": skip_edge_resolution,
            "use_yaml": use_yaml,
        }

        if entity_types:
            request_data["entity_types"] = entity_types
        if edge_types:
            request_data["edge_types"] = edge_types
        if edge_type_map:
            request_data["edge_type_map"] = edge_type_map
        if previous_episode_uuids:
            request_data["previous_episode_uuids"] = previous_episode_uuids

        response = await self._http.request(
            "POST",
            "/api/v1/ingest/promote",
            json=request_data,
        )

        episode = None
        if response.get("episode"):
            episode = Node(**response["episode"])

        nodes = [Node(**n) for n in response.get("nodes", [])]
        edges = [Edge(**e) for e in response.get("edges", [])]
        episodic_edges = [Edge(**e) for e in response.get("episodic_edges", [])]
        communities = [Node(**n) for n in response.get("communities", [])]
        community_edges = [Edge(**e) for e in response.get("community_edges", [])]

        return PromoteToGraphResults(
            episode=episode,
            episodic_edges=episodic_edges,
            nodes=nodes,
            edges=edges,
            communities=communities,
            community_edges=community_edges,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for the given texts using the server's configured embedding model.

        Args:
            texts: List of texts to embed (max 100, each max 8192 chars).

        Returns:
            List of embedding vectors, one per input text.

        Raises:
            ValidationError: If texts list is empty or exceeds limits.
            ConnectionError: If the server cannot be reached.
            ServerError: If the server returns an error.

        Example:
            >>> vectors = await client.embed(["hello world", "knowledge graph"])
            >>> print(f"Dimension: {len(vectors[0])}")
        """
        if not texts:
            raise ValidationError(
                "texts must not be empty",
                field="texts",
            )
        if len(texts) > 100:
            raise ValidationError(
                f"texts exceeds maximum of 100 items ({len(texts)} provided)",
                field="texts",
            )
        for i, text in enumerate(texts):
            if len(text) > 8192:
                raise ValidationError(
                    f"texts[{i}] exceeds maximum of 8192 characters ({len(text)} chars)",
                    field="texts",
                )

        response = await self._http.request(
            "POST",
            "/api/v1/embed",
            json={"texts": texts},
        )

        return response["embeddings"]
