"""
Data models for the web crawler.

This module contains Pydantic models for crawler configuration,
crawl results, and content extraction.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceType(str, Enum):
    """Classification of content sources."""

    GOVERNMENT_HEALTH_AGENCY = "government_health_agency"
    MEDICAL_INSTITUTION = "medical_institution"
    ACADEMIC = "academic"
    HEALTH_INFORMATION_WEBSITE = "health_information_website"
    NEWS = "news"
    OTHER = "other"
    WEB = "web"


class CrawlConfig(BaseModel):
    """
    Configuration for web crawling operations.

    Args:
        max_depth: Maximum crawl depth from seed URLs (default: 3).
        request_timeout: HTTP request timeout in seconds (default: 15).
        request_delay: Delay between requests in seconds (default: 1.0).
        max_content_chars: Maximum characters to analyze (default: 15000).
        use_llm: Whether to use LLM for relevance analysis.
        extract_source: Whether to extract source metadata with LLM.
        stay_on_domain: Whether to restrict crawling to seed domains.
        filter_admin_urls: Whether to filter administrative URLs.
        filter_toplevel_urls: Whether to filter top-level domain URLs.

    Example:
        >>> config = CrawlConfig(
        ...     max_depth=2,
        ...     use_llm=True,
        ...     extract_source=True
        ... )
    """

    model_config = ConfigDict(populate_by_name=True)

    max_depth: int = Field(default=3, ge=1, le=10)
    request_timeout: float = Field(default=15.0, ge=1.0)
    request_delay: float = Field(default=1.0, ge=0.0)
    max_content_chars: int = Field(default=15000, ge=1000)
    use_llm: bool = False
    extract_source: bool = False
    stay_on_domain: bool = True
    filter_admin_urls: bool = True
    filter_toplevel_urls: bool = True


class LLMConfig(BaseModel):
    """
    Configuration for LLM API access.

    Args:
        api_key: API key for the LLM service.
        base_url: Base URL for the LLM API.
        model: Model name to use.
        temperature: Sampling temperature (default: 0.0 for deterministic).
        max_tokens: Maximum tokens in response.
        timeout: Request timeout in seconds.

    Example:
        >>> config = LLMConfig(
        ...     api_key="sk-...",
        ...     base_url="https://api.openai.com/v1",
        ...     model="gpt-4o-mini"
        ... )
    """

    model_config = ConfigDict(populate_by_name=True)

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=100, ge=1)
    timeout: float = Field(default=30.0, ge=1.0)


class SourceInfo(BaseModel):
    """
    Source metadata extracted from a web page.

    Args:
        name: Name of the organization/publisher.
        source_type: Classification of the source.

    Example:
        >>> source = SourceInfo(
        ...     name="Mayo Clinic",
        ...     source_type=SourceType.MEDICAL_INSTITUTION
        ... )
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str = "Unknown"
    source_type: SourceType = SourceType.WEB


class CrawlResult(BaseModel):
    """
    Result from crawling a single URL.

    Args:
        url: The crawled URL.
        title: Page title if available.
        content: Extracted text content.
        markdown: Content converted to markdown.
        is_relevant: Whether the content was deemed relevant.
        source: Extracted source information.
        depth: Crawl depth from seed URL.
        base_domain: The seed domain this URL belongs to.
        crawled_at: When the URL was crawled.
        error: Error message if crawl failed.

    Example:
        >>> result = CrawlResult(
        ...     url="https://example.com/article",
        ...     title="Health Article",
        ...     content="Article content...",
        ...     is_relevant=True,
        ...     depth=1
        ... )
    """

    model_config = ConfigDict(populate_by_name=True)

    url: str
    title: str = ""
    content: str = ""
    markdown: str = ""
    is_relevant: bool = False
    source: SourceInfo | None = None
    depth: int = 0
    base_domain: str = ""
    crawled_at: datetime = Field(default_factory=datetime.utcnow)
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CrawlSummary(BaseModel):
    """
    Summary of a complete crawl operation.

    Args:
        seed_urls: Initial URLs that started the crawl.
        total_crawled: Total number of URLs processed.
        relevant_count: Number of relevant URLs found.
        error_count: Number of URLs that failed to crawl.
        results: List of successful crawl results.
        started_at: When the crawl started.
        completed_at: When the crawl finished.

    Example:
        >>> summary = CrawlSummary(
        ...     seed_urls=["https://example.com"],
        ...     total_crawled=50,
        ...     relevant_count=25,
        ...     error_count=5,
        ...     results=[...]
        ... )
    """

    model_config = ConfigDict(populate_by_name=True)

    seed_urls: list[str] = Field(default_factory=list)
    total_crawled: int = 0
    relevant_count: int = 0
    error_count: int = 0
    results: list[CrawlResult] = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
