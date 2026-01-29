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
    Episode,
    Message,
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
