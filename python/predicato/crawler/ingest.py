"""
Ingestion utilities for crawled content.

This module provides utilities for ingesting crawled content into
the Predicato knowledge graph, mirroring the Go implementation's
crawl-and-index workflow.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from predicato.crawler.models import CrawlConfig, CrawlResult, SourceType
from predicato.crawler.web import WebCrawler

if TYPE_CHECKING:
    from predicato.client import AsyncPredicatoClient, PredicatoClient

logger = logging.getLogger(__name__)


# Default entity types for health content
HEALTH_ENTITY_TYPES = {
    "Disease": {
        "description": "Medical conditions, disorders, or diseases",
        "examples": ["diabetes", "hypertension", "gestational diabetes"],
    },
    "Symptom": {
        "description": "Signs and symptoms of medical conditions",
        "examples": ["fatigue", "nausea", "headache"],
    },
    "Treatment": {
        "description": "Medical treatments, therapies, or interventions",
        "examples": ["insulin therapy", "dietary management", "medication"],
    },
    "Drug": {
        "description": "Medications and pharmaceutical products",
        "examples": ["metformin", "insulin", "prenatal vitamins"],
    },
    "Procedure": {
        "description": "Medical procedures or diagnostic tests",
        "examples": ["blood glucose test", "ultrasound", "amniocentesis"],
    },
    "RiskFactor": {
        "description": "Factors that increase risk of conditions",
        "examples": ["obesity", "family history", "advanced maternal age"],
    },
    "Nutrient": {
        "description": "Vitamins, minerals, and nutritional components",
        "examples": ["folic acid", "iron", "calcium"],
    },
    "FoodItem": {
        "description": "Foods and dietary items",
        "examples": ["leafy greens", "whole grains", "lean protein"],
    },
    "LifestyleRecommendation": {
        "description": "Health and lifestyle recommendations",
        "examples": ["regular exercise", "stress management", "adequate sleep"],
    },
    "Organization": {
        "description": "Medical organizations, institutions, or agencies",
        "examples": ["CDC", "WHO", "Mayo Clinic"],
    },
}


def read_urls_from_file(file_path: str | Path) -> list[str]:
    """
    Read URLs from a CSV or text file.

    Supports two formats:
    1. CSV with URL in first column (header row skipped)
    2. Plain text with one URL per line

    Args:
        file_path: Path to the file containing URLs.

    Returns:
        List of URLs.

    Example:
        >>> urls = read_urls_from_file("urls.csv")
        >>> print(f"Found {len(urls)} URLs")
    """
    path = Path(file_path)
    urls = []

    with open(path, "r", encoding="utf-8") as f:
        # Try to detect if it's a CSV
        first_line = f.readline().strip()
        f.seek(0)

        if "," in first_line or first_line.lower().startswith("url"):
            # CSV format
            reader = csv.reader(f)
            next(reader, None)  # Skip header

            for row in reader:
                if row and row[0].strip():
                    urls.append(row[0].strip())
        else:
            # Plain text format
            for line in f:
                line = line.strip()
                if line and line.startswith("http"):
                    urls.append(line)

    return urls


def crawl_and_ingest(
    client: PredicatoClient,
    seed_urls: list[str],
    *,
    group_id: str,
    crawl_config: CrawlConfig | None = None,
    entity_types: dict[str, Any] | None = None,
    use_two_stage: bool = True,
    skip_resolution: bool = False,
) -> dict[str, Any]:
    """
    Crawl URLs and ingest content into Predicato.

    This function mirrors the Go `crawl-and-index` command, crawling web pages
    and indexing their content directly into the Predicato knowledge graph.

    When crawl_config.use_llm is True, the predicato server's configured LLM
    is used for relevance analysis and source extraction. Otherwise, keyword-based
    filtering is used.

    Args:
        client: Predicato client for ingestion and NLP operations.
        seed_urls: URLs to start crawling from.
        group_id: Group ID for data isolation in Predicato.
        crawl_config: Crawler configuration.
        entity_types: Custom entity types for extraction.
        use_two_stage: Use two-stage ingestion (extract then promote).
        skip_resolution: Skip entity resolution during promotion.

    Returns:
        Summary dict with crawl and ingestion statistics.

    Example:
        >>> with PredicatoClient(base_url="http://localhost:8080") as client:
        ...     result = crawl_and_ingest(
        ...         client,
        ...         ["https://example.com/health"],
        ...         group_id="health-kb",
        ...         crawl_config=CrawlConfig(max_depth=2, use_llm=True)
        ...     )
        ...     print(f"Ingested {result['ingested']} pages")
    """
    config = crawl_config or CrawlConfig()
    types = entity_types or HEALTH_ENTITY_TYPES

    stats = {
        "seed_urls": len(seed_urls),
        "crawled": 0,
        "relevant": 0,
        "ingested": 0,
        "errors": 0,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }

    # Pass the predicato client for server-side NLP operations
    with WebCrawler(config=config, predicato_client=client) as crawler:
        for result in crawler.crawl_iter(seed_urls):
            stats["crawled"] += 1

            if result.error:
                stats["errors"] += 1
                logger.warning(f"Crawl error for {result.url}: {result.error}")
                continue

            if not result.is_relevant:
                continue

            stats["relevant"] += 1

            # Prepare content for ingestion
            content = result.markdown or result.content
            if not content:
                continue

            # Build metadata
            metadata = {
                "url": result.url,
                "crawl_depth": result.depth,
                "base_domain": result.base_domain,
                "crawled_at": result.crawled_at.isoformat(),
            }

            if result.source:
                metadata["source_name"] = result.source.name
                metadata["source_type"] = result.source.source_type.value

            if result.metadata.get("content_hash"):
                metadata["content_hash"] = result.metadata["content_hash"]

            try:
                if use_two_stage:
                    # Two-stage ingestion: extract then promote
                    extraction = client.extract_to_facts(
                        name=result.title or result.url,
                        content=content,
                        source=result.url,
                        group_id=group_id,
                        metadata=metadata,
                        entity_types=types,
                    )

                    if extraction.source_id:
                        client.promote_to_graph(
                            source_id=extraction.source_id,
                            entity_types=types,
                            skip_resolution=skip_resolution,
                        )

                    logger.info(
                        f"Ingested {result.url}: {len(extraction.extracted_nodes)} entities"
                    )
                else:
                    # Single-stage ingestion
                    client.add_episode(
                        name=result.title or result.url,
                        content=content,
                        source=result.url,
                        group_id=group_id,
                        metadata=metadata,
                    )
                    logger.info(f"Ingested {result.url}")

                stats["ingested"] += 1

            except Exception as e:
                stats["errors"] += 1
                logger.error(f"Ingestion error for {result.url}: {e}")

    stats["completed_at"] = datetime.utcnow().isoformat()
    return stats


async def crawl_and_ingest_async(
    client: AsyncPredicatoClient,
    seed_urls: list[str],
    *,
    group_id: str,
    crawl_config: CrawlConfig | None = None,
    entity_types: dict[str, Any] | None = None,
    use_two_stage: bool = True,
    skip_resolution: bool = False,
) -> dict[str, Any]:
    """
    Async version of crawl_and_ingest.

    Crawl URLs and ingest content into Predicato using the async client.

    When crawl_config.use_llm is True, the predicato server's configured LLM
    is used for relevance analysis and source extraction. Otherwise, keyword-based
    filtering is used.

    Note: The crawling itself is synchronous (uses httpx sync client), but
    ingestion uses the async predicato client.

    Args:
        client: Async Predicato client for ingestion and NLP operations.
        seed_urls: URLs to start crawling from.
        group_id: Group ID for data isolation in Predicato.
        crawl_config: Crawler configuration.
        entity_types: Custom entity types for extraction.
        use_two_stage: Use two-stage ingestion (extract then promote).
        skip_resolution: Skip entity resolution during promotion.

    Returns:
        Summary dict with crawl and ingestion statistics.

    Example:
        >>> async with AsyncPredicatoClient(base_url="http://localhost:8080") as client:
        ...     result = await crawl_and_ingest_async(
        ...         client,
        ...         ["https://example.com/health"],
        ...         group_id="health-kb",
        ...         crawl_config=CrawlConfig(use_llm=True)
        ...     )
    """
    config = crawl_config or CrawlConfig()
    types = entity_types or HEALTH_ENTITY_TYPES

    stats = {
        "seed_urls": len(seed_urls),
        "crawled": 0,
        "relevant": 0,
        "ingested": 0,
        "errors": 0,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }

    # Note: WebCrawler uses sync httpx, but we pass client for NLP operations
    # For async, the crawling is sync but ingestion is async
    with WebCrawler(config=config) as crawler:
        for result in crawler.crawl_iter(seed_urls):
            stats["crawled"] += 1

            if result.error:
                stats["errors"] += 1
                continue

            if not result.is_relevant:
                continue

            stats["relevant"] += 1

            content = result.markdown or result.content
            if not content:
                continue

            metadata = {
                "url": result.url,
                "crawl_depth": result.depth,
                "base_domain": result.base_domain,
                "crawled_at": result.crawled_at.isoformat(),
            }

            if result.source:
                metadata["source_name"] = result.source.name
                metadata["source_type"] = result.source.source_type.value

            if result.metadata.get("content_hash"):
                metadata["content_hash"] = result.metadata["content_hash"]

            try:
                if use_two_stage:
                    extraction = await client.extract_to_facts(
                        name=result.title or result.url,
                        content=content,
                        source=result.url,
                        group_id=group_id,
                        metadata=metadata,
                        entity_types=types,
                    )

                    if extraction.source_id:
                        await client.promote_to_graph(
                            source_id=extraction.source_id,
                            entity_types=types,
                            skip_resolution=skip_resolution,
                        )
                else:
                    await client.add_episode(
                        name=result.title or result.url,
                        content=content,
                        source=result.url,
                        group_id=group_id,
                        metadata=metadata,
                    )

                stats["ingested"] += 1

            except Exception as e:
                stats["errors"] += 1
                logger.error(f"Ingestion error for {result.url}: {e}")

    stats["completed_at"] = datetime.utcnow().isoformat()
    return stats


def ingest_crawl_results(
    client: PredicatoClient,
    results: Iterator[CrawlResult] | list[CrawlResult],
    *,
    group_id: str,
    entity_types: dict[str, Any] | None = None,
    use_two_stage: bool = True,
    skip_resolution: bool = False,
) -> dict[str, Any]:
    """
    Ingest pre-crawled results into Predicato.

    Useful when you have already crawled content and want to ingest
    it separately, or when re-ingesting saved results.

    Args:
        client: Predicato client for ingestion.
        results: Iterator or list of CrawlResult objects.
        group_id: Group ID for data isolation.
        entity_types: Custom entity types for extraction.
        use_two_stage: Use two-stage ingestion.
        skip_resolution: Skip entity resolution.

    Returns:
        Summary dict with ingestion statistics.
    """
    types = entity_types or HEALTH_ENTITY_TYPES

    stats = {
        "processed": 0,
        "ingested": 0,
        "errors": 0,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }

    for result in results:
        stats["processed"] += 1

        if not result.is_relevant:
            continue

        content = result.markdown or result.content
        if not content:
            continue

        metadata = {
            "url": result.url,
            "crawl_depth": result.depth,
            "base_domain": result.base_domain,
            "crawled_at": result.crawled_at.isoformat(),
        }

        if result.source:
            metadata["source_name"] = result.source.name
            metadata["source_type"] = result.source.source_type.value

        try:
            if use_two_stage:
                extraction = client.extract_to_facts(
                    name=result.title or result.url,
                    content=content,
                    source=result.url,
                    group_id=group_id,
                    metadata=metadata,
                    entity_types=types,
                )

                if extraction.source_id:
                    client.promote_to_graph(
                        source_id=extraction.source_id,
                        entity_types=types,
                        skip_resolution=skip_resolution,
                    )
            else:
                client.add_episode(
                    name=result.title or result.url,
                    content=content,
                    source=result.url,
                    group_id=group_id,
                    metadata=metadata,
                )

            stats["ingested"] += 1

        except Exception as e:
            stats["errors"] += 1
            logger.error(f"Ingestion error for {result.url}: {e}")

    stats["completed_at"] = datetime.utcnow().isoformat()
    return stats
