"""
Predicato Python Client.

A Python client for the Predicato knowledge graph framework, providing
a Pythonic interface for ingesting content and querying the knowledge graph.

Example:
    >>> from predicato import PredicatoClient
    >>> with PredicatoClient(base_url="http://localhost:8080") as client:
    ...     result = client.add_episode(
    ...         name="Meeting Notes",
    ...         content="Discussed project timeline...",
    ...         group_id="my-project"
    ...     )
"""

__version__ = "0.1.0"

from predicato.client import AsyncPredicatoClient, PredicatoClient
from predicato.exceptions import (
    ConfigurationError,
    ConnectionError,
    NotFoundError,
    PredicatoError,
    RateLimitError,
    ServerError,
    ValidationError,
)
from predicato.models import (
    AddBulkEpisodeResults,
    AddEntityNodeRequest,
    AddEpisodeResults,
    AddMessagesRequest,
    AddMessagesResponse,
    ClearDataRequest,
    ClearDataResponse,
    Edge,
    Episode,
    ErrorResponse,
    Message,
    Node,
    SearchRequest,
    SearchResults,
)

__all__ = [
    # Version
    "__version__",
    # Clients
    "PredicatoClient",
    "AsyncPredicatoClient",
    # Exceptions
    "PredicatoError",
    "ConfigurationError",
    "ValidationError",
    "ConnectionError",
    "ServerError",
    "RateLimitError",
    "NotFoundError",
    # Request Models
    "Episode",
    "Message",
    "AddMessagesRequest",
    "AddEntityNodeRequest",
    "ClearDataRequest",
    "SearchRequest",
    # Response Models
    "Node",
    "Edge",
    "AddEpisodeResults",
    "AddBulkEpisodeResults",
    "AddMessagesResponse",
    "SearchResults",
    "ErrorResponse",
    "ClearDataResponse",
]
