"""
Predicato Web Crawler.

This module provides web crawling capabilities for discovering and extracting
content from web pages, with optional LLM-based relevance analysis and
source metadata extraction.

The crawler integrates seamlessly with the Predicato client for direct
ingestion into the knowledge graph. When LLM-based analysis is enabled,
NLP operations are performed by the predicato server's configured models.

Example:
    Basic crawling with keyword-based filtering::

        from predicato.crawler import WebCrawler, CrawlConfig

        config = CrawlConfig(max_depth=2, use_llm=False)
        with WebCrawler(config=config) as crawler:
            summary = crawler.crawl(["https://example.com/health"])
            for result in summary.results:
                print(f"{result.url}: {result.title}")

    Crawl and ingest into Predicato::

        from predicato import PredicatoClient
        from predicato.crawler import crawl_and_ingest, CrawlConfig

        with PredicatoClient(base_url="http://localhost:8080") as client:
            stats = crawl_and_ingest(
                client,
                seed_urls=["https://example.com/health"],
                group_id="health-kb",
                crawl_config=CrawlConfig(max_depth=2)
            )
            print(f"Ingested {stats['ingested']} pages")

    Using server's LLM for relevance analysis::

        from predicato import PredicatoClient
        from predicato.crawler import WebCrawler, CrawlConfig

        # LLM operations use the predicato server's configured models
        config = CrawlConfig(use_llm=True, extract_source=True)

        with PredicatoClient(base_url="http://localhost:8080") as client:
            with WebCrawler(config=config, predicato_client=client) as crawler:
                for result in crawler.crawl_iter(["https://example.com"]):
                    if result.is_relevant and result.source:
                        print(f"{result.url} from {result.source.name}")
"""

from predicato.crawler.filters import (
    ADMIN_KEYWORDS,
    DEFAULT_KEYWORDS,
    DIABETES_KEYWORDS,
    NUTRITION_KEYWORDS,
    PREGNANCY_KEYWORDS,
    KeywordFilter,
    URLFilter,
    remove_boilerplate,
)
from predicato.crawler.ingest import (
    HEALTH_ENTITY_TYPES,
    crawl_and_ingest,
    crawl_and_ingest_async,
    ingest_crawl_results,
    read_urls_from_file,
)
from predicato.crawler.models import (
    CrawlConfig,
    CrawlResult,
    CrawlSummary,
    SourceInfo,
    SourceType,
)
from predicato.crawler.web import WebCrawler

__all__ = [
    # Core crawler
    "WebCrawler",
    # Models
    "CrawlConfig",
    "CrawlResult",
    "CrawlSummary",
    "SourceInfo",
    "SourceType",
    # Filters
    "KeywordFilter",
    "URLFilter",
    "remove_boilerplate",
    # Keywords
    "DEFAULT_KEYWORDS",
    "PREGNANCY_KEYWORDS",
    "DIABETES_KEYWORDS",
    "NUTRITION_KEYWORDS",
    "ADMIN_KEYWORDS",
    # Ingestion
    "crawl_and_ingest",
    "crawl_and_ingest_async",
    "ingest_crawl_results",
    "read_urls_from_file",
    "HEALTH_ENTITY_TYPES",
]
