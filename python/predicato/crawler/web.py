"""
Web crawler implementation.

This module provides a web crawler that discovers and extracts content
from web pages, with optional LLM-based relevance analysis and source extraction.

All NLP operations (relevance analysis, source extraction) are performed
through the predicato server's configured LLM, not directly by this client.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Iterator
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from predicato.crawler.filters import (
    KeywordFilter,
    URLFilter,
    remove_boilerplate,
)
from predicato.crawler.models import (
    CrawlConfig,
    CrawlResult,
    CrawlSummary,
    SourceInfo,
    SourceType,
)

if TYPE_CHECKING:
    from predicato.client import PredicatoClient

logger = logging.getLogger(__name__)


class WebCrawler:
    """
    Web crawler for discovering and extracting content from web pages.

    Supports breadth-first crawling with configurable depth, domain restrictions,
    and content filtering. LLM-based relevance analysis and source extraction
    use the predicato server's configured models.

    Args:
        config: Crawler configuration.
        predicato_client: Predicato client for NLP operations via server.
        keyword_filter: Optional custom keyword filter for relevance checking.
        url_filter: Optional custom URL filter.
        on_result: Optional callback invoked for each crawl result.

    Example:
        Using predicato server for NLP operations::

            >>> from predicato import PredicatoClient
            >>> from predicato.crawler import WebCrawler, CrawlConfig
            >>>
            >>> with PredicatoClient(base_url="http://localhost:8080") as client:
            ...     config = CrawlConfig(max_depth=2, use_llm=True)
            ...     with WebCrawler(config, predicato_client=client) as crawler:
            ...         summary = crawler.crawl(["https://example.com/health"])

        Keyword-based filtering (no server required)::

            >>> config = CrawlConfig(max_depth=2, use_llm=False)
            >>> with WebCrawler(config) as crawler:
            ...     summary = crawler.crawl(["https://example.com/health"])
    """

    def __init__(
        self,
        config: CrawlConfig | None = None,
        predicato_client: PredicatoClient | None = None,
        keyword_filter: KeywordFilter | None = None,
        url_filter: URLFilter | None = None,
        on_result: Callable[[CrawlResult], None] | None = None,
    ) -> None:
        self.config = config or CrawlConfig()
        self._predicato_client = predicato_client
        self.keyword_filter = keyword_filter or KeywordFilter()
        self.url_filter = url_filter or URLFilter(
            filter_admin=self.config.filter_admin_urls,
            filter_toplevel=self.config.filter_toplevel_urls,
        )
        self.on_result = on_result

        # HTTP client for fetching pages
        self._client = httpx.Client(
            timeout=self.config.request_timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "PredicatoCrawler/1.0 (+https://github.com/soundprediction/predicato)"
            },
        )

        # Content hash cache for deduplication
        self._content_hashes: set[str] = set()

    def __enter__(self) -> WebCrawler:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the crawler and release resources."""
        self._client.close()

    def _content_hash(self, content: str) -> str:
        """Generate SHA256 hash of content for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _fetch_page(self, url: str) -> tuple[str, str, list[str]]:
        """
        Fetch a web page and extract content.

        Returns:
            Tuple of (title, body_text, links).
        """
        response = self._client.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
        elif soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)

        # Extract body content
        body = soup.find("body")
        body_text = body.get_text() if body else soup.get_text()

        # Extract links
        links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            absolute_url = urljoin(url, href)
            links.append(absolute_url)

        return title, body_text, links

    def _analyze_relevance_server(self, content: str) -> bool:
        """
        Use predicato server's LLM to determine if content is relevant.

        Args:
            content: Text content to analyze.

        Returns:
            True if content is relevant.
        """
        if self._predicato_client is None:
            return self.keyword_filter.is_relevant(content)

        try:
            # Call the server's analyze-relevance endpoint
            response = self._predicato_client._http.request(
                "POST",
                "/api/v1/nlp/analyze-relevance",
                json={
                    "content": content[:self.config.max_content_chars],
                },
            )
            return response.get("is_relevant", False)

        except Exception as e:
            logger.warning(f"Server relevance check failed: {e}, falling back to keywords")
            return self.keyword_filter.is_relevant(content)

    def _extract_source_server(self, content: str, url: str) -> SourceInfo | None:
        """
        Use predicato server's LLM to extract source information.

        Args:
            content: Page content.
            url: Page URL.

        Returns:
            SourceInfo if extraction succeeded, None otherwise.
        """
        if self._predicato_client is None:
            return None

        try:
            # Call the server's extract-source endpoint
            response = self._predicato_client._http.request(
                "POST",
                "/api/v1/nlp/extract-source",
                json={
                    "content": content[:5000],
                    "url": url,
                },
            )

            source_name = response.get("source_name", "Unknown")
            source_type_str = response.get("source_type", "other").lower()

            try:
                source_type = SourceType(source_type_str)
            except ValueError:
                source_type = SourceType.OTHER

            return SourceInfo(name=source_name, source_type=source_type)

        except Exception as e:
            logger.warning(f"Server source extraction failed: {e}")
            return None

    def _analyze_content(
        self, content: str, url: str
    ) -> tuple[bool, SourceInfo | None]:
        """
        Analyze content for relevance and optionally extract source info.

        Args:
            content: Page content.
            url: Page URL.

        Returns:
            Tuple of (is_relevant, source_info).
        """
        # Check relevance
        if self.config.use_llm and self._predicato_client:
            is_relevant = self._analyze_relevance_server(content)
        else:
            is_relevant = self.keyword_filter.is_relevant(content)

        # Extract source if relevant and configured
        source_info = None
        if is_relevant and self.config.extract_source and self._predicato_client:
            source_info = self._extract_source_server(content, url)

        return is_relevant, source_info

    def crawl_url(
        self,
        url: str,
        depth: int = 0,
        base_domain: str | None = None,
    ) -> CrawlResult:
        """
        Crawl a single URL and return the result.

        Args:
            url: URL to crawl.
            depth: Current crawl depth.
            base_domain: Base domain for this crawl.

        Returns:
            CrawlResult with extracted content and metadata.
        """
        cleaned_url = self.url_filter.clean(url)
        domain = base_domain or self.url_filter.get_domain(cleaned_url)

        result = CrawlResult(
            url=cleaned_url,
            depth=depth,
            base_domain=domain,
        )

        try:
            title, body_text, _ = self._fetch_page(cleaned_url)
            result.title = title
            result.content = body_text

            # Check for duplicate content
            content_hash = self._content_hash(body_text)
            if content_hash in self._content_hashes:
                result.metadata["duplicate"] = True
                return result

            self._content_hashes.add(content_hash)
            result.metadata["content_hash"] = content_hash

            # Truncate for analysis
            analysis_content = body_text[: self.config.max_content_chars]

            # Analyze content
            is_relevant, source_info = self._analyze_content(analysis_content, cleaned_url)
            result.is_relevant = is_relevant
            result.source = source_info

            # Clean content if relevant
            if is_relevant:
                result.markdown = remove_boilerplate(body_text)

        except httpx.HTTPStatusError as e:
            result.error = f"HTTP {e.response.status_code}: {e.response.reason_phrase}"
            logger.warning(f"HTTP error crawling {cleaned_url}: {result.error}")
        except httpx.TimeoutException:
            result.error = "Request timeout"
            logger.warning(f"Timeout crawling {cleaned_url}")
        except Exception as e:
            result.error = str(e)
            logger.error(f"Error crawling {cleaned_url}: {e}")

        return result

    def crawl(
        self,
        seed_urls: list[str],
        max_results: int | None = None,
    ) -> CrawlSummary:
        """
        Crawl starting from seed URLs using breadth-first search.

        Args:
            seed_urls: Initial URLs to start crawling from.
            max_results: Optional maximum number of relevant results to collect.

        Returns:
            CrawlSummary with all results and statistics.

        Example:
            >>> summary = crawler.crawl(["https://example.com/health"])
            >>> print(f"Found {summary.relevant_count} relevant pages")
        """
        summary = CrawlSummary(
            seed_urls=seed_urls,
            started_at=datetime.utcnow(),
        )

        visited: set[str] = set()
        queue: deque[tuple[str, int, str]] = deque()

        # Initialize queue with seed URLs
        for url in seed_urls:
            cleaned_url = self.url_filter.clean(url)
            domain = self.url_filter.get_domain(cleaned_url)

            if not domain:
                logger.warning(f"Skipping malformed URL: {url}")
                continue

            # Filter seed URLs
            if self.config.filter_admin_urls and self.url_filter.is_administrative(cleaned_url):
                logger.info(f"Skipping administrative seed URL: {cleaned_url}")
                continue

            if self.config.filter_toplevel_urls and self.url_filter.is_toplevel(cleaned_url):
                logger.info(f"Skipping top-level seed URL: {cleaned_url}")
                continue

            queue.append((cleaned_url, 0, domain))
            visited.add(cleaned_url)

        # BFS crawl
        while queue:
            if max_results and summary.relevant_count >= max_results:
                break

            url, depth, base_domain = queue.popleft()

            if depth > self.config.max_depth:
                continue

            logger.info(f"Crawling: {url} (depth={depth}, domain={base_domain})")

            # Crawl the URL
            result = self.crawl_url(url, depth, base_domain)
            summary.total_crawled += 1

            if result.error:
                summary.error_count += 1
            elif result.is_relevant:
                summary.relevant_count += 1
                summary.results.append(result)

                # Notify callback
                if self.on_result:
                    self.on_result(result)

            # Queue child URLs if relevant and within depth
            if result.is_relevant and depth < self.config.max_depth:
                try:
                    _, _, links = self._fetch_page(url)

                    for link in links:
                        cleaned_link = self.url_filter.clean(link)

                        if cleaned_link in visited:
                            continue

                        if not self.url_filter.should_crawl(
                            cleaned_link,
                            base_domain if self.config.stay_on_domain else None,
                        ):
                            continue

                        visited.add(cleaned_link)
                        queue.append((cleaned_link, depth + 1, base_domain))

                except Exception as e:
                    logger.warning(f"Error extracting links from {url}: {e}")

            # Rate limiting
            if self.config.request_delay > 0:
                time.sleep(self.config.request_delay)

        summary.completed_at = datetime.utcnow()
        return summary

    def crawl_iter(
        self,
        seed_urls: list[str],
    ) -> Iterator[CrawlResult]:
        """
        Crawl and yield results as they are discovered.

        This is a generator version of crawl() that yields results
        incrementally, useful for streaming or early termination.

        Args:
            seed_urls: Initial URLs to start crawling from.

        Yields:
            CrawlResult for each crawled URL.

        Example:
            >>> for result in crawler.crawl_iter(["https://example.com"]):
            ...     if result.is_relevant:
            ...         print(f"Found: {result.url}")
        """
        visited: set[str] = set()
        queue: deque[tuple[str, int, str]] = deque()

        # Initialize queue
        for url in seed_urls:
            cleaned_url = self.url_filter.clean(url)
            domain = self.url_filter.get_domain(cleaned_url)

            if not domain:
                continue

            if self.config.filter_admin_urls and self.url_filter.is_administrative(cleaned_url):
                continue

            if self.config.filter_toplevel_urls and self.url_filter.is_toplevel(cleaned_url):
                continue

            queue.append((cleaned_url, 0, domain))
            visited.add(cleaned_url)

        while queue:
            url, depth, base_domain = queue.popleft()

            if depth > self.config.max_depth:
                continue

            result = self.crawl_url(url, depth, base_domain)
            yield result

            # Queue child URLs
            if result.is_relevant and depth < self.config.max_depth:
                try:
                    _, _, links = self._fetch_page(url)

                    for link in links:
                        cleaned_link = self.url_filter.clean(link)

                        if cleaned_link in visited:
                            continue

                        if not self.url_filter.should_crawl(
                            cleaned_link,
                            base_domain if self.config.stay_on_domain else None,
                        ):
                            continue

                        visited.add(cleaned_link)
                        queue.append((cleaned_link, depth + 1, base_domain))

                except Exception:
                    pass

            if self.config.request_delay > 0:
                time.sleep(self.config.request_delay)
