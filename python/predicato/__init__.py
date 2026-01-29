"""
Predicato Python Client.

A Python client for the Predicato knowledge graph framework, providing
a Pythonic interface for ingesting content and querying the knowledge graph.

Basic Example:
    >>> from predicato import PredicatoClient
    >>> with PredicatoClient(base_url="http://localhost:8080") as client:
    ...     result = client.add_episode(
    ...         name="Meeting Notes",
    ...         content="Discussed project timeline...",
    ...         group_id="my-project"
    ...     )

Crawler Example (requires `pip install predicato[crawler]`):
    >>> from predicato import PredicatoClient
    >>> from predicato.crawler import crawl_and_ingest, CrawlConfig
    >>> with PredicatoClient(base_url="http://localhost:8080") as client:
    ...     stats = crawl_and_ingest(
    ...         client,
    ...         seed_urls=["https://example.com/health"],
    ...         group_id="health-kb",
    ...         crawl_config=CrawlConfig(max_depth=2)
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
    EmbeddingModelConfig,
    Episode,
    ErrorResponse,
    ExtractedEdge,
    ExtractedNode,
    ExtractEpisodeRequest,
    ExtractionMetadata,
    ExtractionResults,
    Message,
    ModelsConfig,
    NLPModelConfig,
    NLPModelsConfig,
    Node,
    PromoteToGraphRequest,
    PromoteToGraphResults,
    RouterRule,
    SearchFactsRequest,
    SearchFactsResults,
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
    # Two-Stage Ingestion Models
    "ExtractEpisodeRequest",
    "ExtractedNode",
    "ExtractedEdge",
    "ExtractionMetadata",
    "ExtractionResults",
    "PromoteToGraphRequest",
    "PromoteToGraphResults",
    # Fact Store Search Models
    "SearchFactsRequest",
    "SearchFactsResults",
    # Server Configuration Models
    "ModelsConfig",
    "NLPModelsConfig",
    "NLPModelConfig",
    "EmbeddingModelConfig",
    "RouterRule",
]
