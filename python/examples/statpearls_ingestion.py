#!/usr/bin/env python3
"""
StatPearls Ingestion Example

This example demonstrates how to ingest medical articles from the StatPearls
dataset into Predicato's knowledge graph.

StatPearls is a peer-reviewed medical education resource available from NCBI.
This example supports two data sources:
1. NCBI Entrez API (official programmatic access)
2. Local files (downloaded tarball from NCBI FTP)

NCBI API Documentation:
    https://www.ncbi.nlm.nih.gov/home/develop/api/
    https://www.ncbi.nlm.nih.gov/books/NBK25500/

Requirements:
    pip install predicato

Usage:
    # Set the Predicato server URL
    export PREDICATO_URL=http://localhost:8080

    # Use NCBI Entrez API (recommended)
    python statpearls_ingestion.py --source entrez --max-articles 10

    # Optionally provide your email for NCBI API (recommended by NCBI)
    python statpearls_ingestion.py --source entrez --email your@email.com

    # Use local tarball (download from NCBI FTP first)
    # wget https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz
    python statpearls_ingestion.py --source local --data-path ./statpearls_NBK430685.tar.gz

    # Additional options
    python statpearls_ingestion.py --source entrez --max-articles 20 --mode two-stage
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree

from predicato import PredicatoClient
from predicato.exceptions import PredicatoError
from predicato.statpearls import (
    _extract_text_from_element,
    _get_element_text,
    iter_articles_from_directory,
    iter_articles_from_tarball,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

# NCBI Entrez E-utilities base URL
ENTREZ_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI FTP URL for StatPearls tarball (for local mode)
NCBI_FTP_URL = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"

# StatPearls book ID in NCBI Books database
STATPEARLS_BOOK_ID = "NBK430685"


# =============================================================================
# Entrez API functions (different XML schema from local NXML files)
# =============================================================================


def search_statpearls_chapters(
    email: str | None = None,
    max_results: int = 100,
) -> list[str]:
    """
    Search for StatPearls chapter IDs using NCBI Entrez ESearch.

    Args:
        email: Email address for NCBI API (recommended by NCBI)
        max_results: Maximum number of chapter IDs to retrieve

    Returns:
        List of chapter IDs (NBK IDs)
    """
    params = {
        "db": "books",
        "term": "statpearls[book]",
        "retmax": str(max_results),
        "retmode": "xml",
    }
    if email:
        params["email"] = email

    url = f"{ENTREZ_BASE_URL}/esearch.fcgi?" + urllib.parse.urlencode(params)

    print("Searching NCBI Books for StatPearls chapters...")
    print(f"(Requesting up to {max_results} results)")

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            xml_content = response.read().decode("utf-8")

        root = ElementTree.fromstring(xml_content)

        ids = []
        id_list = root.find(".//IdList")
        if id_list is not None:
            for id_elem in id_list.findall("Id"):
                if id_elem.text:
                    ids.append(id_elem.text)

        count_elem = root.find(".//Count")
        total_count = int(count_elem.text) if count_elem is not None and count_elem.text else 0

        print(f"Found {total_count} total chapters, retrieved {len(ids)} IDs")
        return ids

    except Exception as e:
        print(f"Error searching NCBI: {e}")
        print("\nFallback: You can download the tarball manually:")
        print(f"  wget {NCBI_FTP_URL}")
        print("  python statpearls_ingestion.py --source local --data-path ./statpearls_NBK430685.tar.gz")
        sys.exit(1)


def fetch_chapter_content(
    chapter_id: str,
    email: str | None = None,
) -> dict[str, str] | None:
    """
    Fetch a single StatPearls chapter using NCBI Entrez EFetch.

    Args:
        chapter_id: NCBI Books chapter ID
        email: Email address for NCBI API

    Returns:
        Article dictionary with id, title, and content, or None if failed
    """
    params = {
        "db": "books",
        "id": chapter_id,
        "rettype": "xml",
        "retmode": "xml",
    }
    if email:
        params["email"] = email

    url = f"{ENTREZ_BASE_URL}/efetch.fcgi?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            xml_content = response.read().decode("utf-8")

        return parse_entrez_chapter_xml(xml_content, chapter_id)

    except Exception as e:
        print(f"Warning: Failed to fetch chapter {chapter_id}: {e}")
        return None


def parse_entrez_chapter_xml(xml_content: str, chapter_id: str) -> dict[str, str] | None:
    """
    Parse NCBI Books XML response and extract article content.

    Note: The Entrez API returns a different XML schema than local NXML files,
    so this function uses its own parsing logic rather than the shared module.

    Args:
        xml_content: Raw XML string from EFetch
        chapter_id: Chapter ID for fallback title

    Returns:
        Article dictionary or None if parsing fails
    """
    try:
        root = ElementTree.fromstring(xml_content)

        title = None
        for title_path in [
            ".//book-part-meta/title-group/title",
            ".//title-group/title",
            ".//book-title",
            ".//article-title",
            ".//title",
        ]:
            title_elem = root.find(title_path)
            if title_elem is not None:
                title = _get_element_text(title_elem)
                if title:
                    break

        if not title:
            title = f"StatPearls Chapter {chapter_id}"

        content_parts: list[str] = []

        body = root.find(".//body")
        if body is not None:
            content_parts.extend(_extract_text_from_element(body))

        abstract = root.find(".//abstract")
        if abstract is not None:
            abstract_text = _extract_text_from_element(abstract)
            if abstract_text:
                content_parts = ["Abstract:", *abstract_text, "", *content_parts]

        content = "\n\n".join(content_parts)

        if not content:
            return None

        return {
            "id": chapter_id,
            "title": title,
            "content": content,
        }

    except ElementTree.ParseError as e:
        print(f"XML parse error for chapter {chapter_id}: {e}")
        return None


def iter_articles_from_entrez(
    email: str | None = None,
    max_articles: int | None = None,
    delay: float = 0.4,
) -> Iterator[dict[str, str]]:
    """
    Iterate through StatPearls articles using NCBI Entrez API.

    NCBI recommends no more than 3 requests per second without an API key,
    so we add a delay between requests.

    Args:
        email: Email address for NCBI API (recommended)
        max_articles: Maximum number of articles to yield
        delay: Delay between API requests (NCBI recommends >= 0.34s)

    Yields:
        Article dictionaries with id, title, and content
    """
    search_limit = max_articles if max_articles else 1000
    chapter_ids = search_statpearls_chapters(email, search_limit)

    if not chapter_ids:
        print("No chapters found.")
        return

    print("\nFetching chapter content...")
    count = 0

    for i, chapter_id in enumerate(chapter_ids):
        if max_articles and count >= max_articles:
            break

        print(f"  [{i + 1}/{len(chapter_ids)}] Fetching {chapter_id}...", end=" ", flush=True)

        article = fetch_chapter_content(chapter_id, email)
        if article and article.get("content"):
            count += 1
            print(f"OK ({article['title'][:40]}...)")
            yield article
        else:
            print("Skipped (no content)")

        if delay > 0:
            time.sleep(delay)


# =============================================================================
# Loading and ingestion orchestration
# =============================================================================


def load_articles(
    source: str,
    data_path: str | None,
    max_articles: int | None,
    email: str | None,
) -> list[dict[str, str]]:
    """
    Load articles based on source type.

    Args:
        source: "entrez" for NCBI API, "local" for local files
        data_path: Path to local tarball or directory
        max_articles: Maximum articles to load
        email: Email for NCBI API

    Returns:
        List of article dictionaries
    """
    if source == "entrez":
        print("Source: NCBI Entrez E-utilities API")
        if email:
            print(f"Email: {email}")
        else:
            print("Note: Consider providing --email for NCBI API (recommended by NCBI)")
        articles = list(iter_articles_from_entrez(email, max_articles))

    else:  # local
        if not data_path:
            print("Error: --data-path is required when using --source local")
            print("\nTo download the tarball:")
            print(f"  wget {NCBI_FTP_URL}")
            sys.exit(1)

        path = Path(data_path)
        if not path.exists():
            print(f"Error: Path not found: {data_path}")
            print("\nTo download the tarball:")
            print(f"  wget {NCBI_FTP_URL}")
            sys.exit(1)

        print(f"Source: Local ({data_path})")

        if path.is_file() and path.suffix in (".gz", ".tar"):
            print("Loading from tarball...")
            articles = list(iter_articles_from_tarball(str(path), max_articles))
        elif path.is_dir():
            print("Loading from directory...")
            articles = list(iter_articles_from_directory(str(path), max_articles))
        else:
            print(f"Error: Unsupported file type: {data_path}")
            print("Expected a .tar.gz file or a directory")
            sys.exit(1)

    print(f"Loaded {len(articles)} articles")
    return articles


def ingest_simple(
    client: PredicatoClient,
    articles: list[dict[str, str]],
    group_id: str,
    delay: float = 0.5,
) -> None:
    """
    Ingest articles using the simple single-stage approach.
    """
    print("\n" + "=" * 60)
    print("Simple Ingestion (Single-Stage)")
    print("=" * 60)

    success_count = 0
    error_count = 0

    for i, article in enumerate(articles, 1):
        title = article["title"][:60] + "..." if len(article["title"]) > 60 else article["title"]
        print(f"\n[{i}/{len(articles)}] {title}")

        content = article.get("content", "")
        if not content:
            print("    Skipping: No content")
            continue

        try:
            client.add_episode(
                name=article["title"],
                content=content,
                source="statpearls",
                group_id=group_id,
                metadata={
                    "article_id": article["id"],
                    "source": "NCBI/StatPearls",
                    "content_type": "medical_article",
                },
            )
            print("    Queued for processing")
            success_count += 1

        except PredicatoError as e:
            print(f"    Error: {e}")
            error_count += 1

        if delay > 0:
            time.sleep(delay)

    print(f"\nCompleted: {success_count} succeeded, {error_count} failed")


def ingest_two_stage(
    client: PredicatoClient,
    articles: list[dict[str, str]],
    group_id: str,
    delay: float = 0.5,
) -> None:
    """
    Ingest articles using the two-stage approach.

    Stage 1: Extract entities and relationships (stored in fact store)
    Stage 2: Promote to graph with entity resolution
    """
    print("\n" + "=" * 60)
    print("Two-Stage Ingestion")
    print("=" * 60)

    medical_entity_types = {
        "Disease": {
            "description": "A medical condition, disorder, or disease",
            "examples": ["diabetes mellitus", "hypertension", "pneumonia"],
        },
        "Drug": {
            "description": "A medication, pharmaceutical, or therapeutic agent",
            "examples": ["metformin", "lisinopril", "aspirin"],
        },
        "Symptom": {
            "description": "A symptom, sign, or clinical finding",
            "examples": ["fever", "chest pain", "dyspnea"],
        },
        "Procedure": {
            "description": "A medical procedure, surgery, or diagnostic test",
            "examples": ["MRI", "colonoscopy", "appendectomy"],
        },
        "AnatomicalStructure": {
            "description": "A body part, organ, or anatomical structure",
            "examples": ["heart", "liver", "femur"],
        },
        "Pathogen": {
            "description": "A bacteria, virus, fungus, or pathogenic organism",
            "examples": ["Staphylococcus aureus", "SARS-CoV-2", "Candida albicans"],
        },
        "LabTest": {
            "description": "A laboratory test or biomarker",
            "examples": ["HbA1c", "troponin", "complete blood count"],
        },
    }

    success_count = 0
    error_count = 0
    total_entities = 0
    total_relationships = 0

    for i, article in enumerate(articles, 1):
        title = article["title"][:60] + "..." if len(article["title"]) > 60 else article["title"]
        print(f"\n[{i}/{len(articles)}] {title}")

        content = article.get("content", "")
        if not content:
            print("    Skipping: No content")
            continue

        try:
            print("    Stage 1: Extracting entities...")
            extraction = client.extract_to_facts(
                name=article["title"],
                content=content,
                source="statpearls",
                group_id=group_id,
                metadata={
                    "article_id": article["id"],
                    "source": "NCBI/StatPearls",
                },
                entity_types=medical_entity_types,
            )

            num_nodes = len(extraction.extracted_nodes)
            num_edges = len(extraction.extracted_edges)
            print(f"    Extracted: {num_nodes} entities, {num_edges} relationships")

            if extraction.extracted_nodes:
                print("    Sample entities:")
                for node in extraction.extracted_nodes[:3]:
                    print(f"      - {node.name} ({node.type})")
                if num_nodes > 3:
                    print(f"      ... and {num_nodes - 3} more")

            print("    Stage 2: Promoting to graph...")
            result = client.promote_to_graph(source_id=extraction.source_id)

            resolved_nodes = len(result.nodes)
            resolved_edges = len(result.edges)
            print(f"    Resolved: {resolved_nodes} entities, {resolved_edges} relationships")

            if result.communities:
                print(f"    Communities updated: {len(result.communities)}")

            success_count += 1
            total_entities += resolved_nodes
            total_relationships += resolved_edges

        except PredicatoError as e:
            print(f"    Error: {e}")
            error_count += 1

        if delay > 0:
            time.sleep(delay)

    print(f"\nCompleted: {success_count} succeeded, {error_count} failed")
    print(f"Total: {total_entities} entities, {total_relationships} relationships")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest StatPearls medical articles into Predicato",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Data Sources:
  --source entrez   Use NCBI Entrez E-utilities API (recommended)
                    API Docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/

  --source local    Use locally downloaded tarball
                    Download: wget {NCBI_FTP_URL}
                    Requires: --data-path <path>

Examples:
  # Use Entrez API
  python statpearls_ingestion.py --source entrez --max-articles 10

  # With email (recommended by NCBI)
  python statpearls_ingestion.py --source entrez --email you@example.com

  # Use local tarball
  python statpearls_ingestion.py --source local --data-path ./statpearls_NBK430685.tar.gz
""",
    )
    parser.add_argument(
        "--source",
        choices=["entrez", "local"],
        default="entrez",
        help="Data source: 'entrez' for NCBI API, 'local' for local files (default: 'entrez')",
    )
    parser.add_argument(
        "--data-path",
        help="Path to local tarball or directory (required for --source local)",
    )
    parser.add_argument(
        "--email",
        help="Email address for NCBI API (recommended by NCBI)",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=5,
        help="Maximum number of articles to ingest (default: 5)",
    )
    parser.add_argument(
        "--group-id",
        default="statpearls",
        help="Group ID for data isolation (default: 'statpearls')",
    )
    parser.add_argument(
        "--mode",
        choices=["simple", "two-stage"],
        default="two-stage",
        help="Ingestion mode (default: 'two-stage')",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Predicato server URL (default: from PREDICATO_URL env var)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list articles, don't ingest",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("StatPearls Ingestion Example")
    print("=" * 60)

    # Load articles
    articles = load_articles(args.source, args.data_path, args.max_articles, args.email)

    if not articles:
        print("No articles loaded.")
        sys.exit(1)

    # List articles if requested
    if args.list_only:
        print("\nArticles:")
        for i, article in enumerate(articles, 1):
            print(f"  {i}. {article['title']}")
        sys.exit(0)

    # Initialize Predicato client
    print("\nConnecting to Predicato server...")
    try:
        client = PredicatoClient(base_url=args.url, group_id=args.group_id)
    except Exception as e:
        print(f"\nError: Could not initialize client: {e}")
        print("\nMake sure:")
        print("  1. The Predicato server is running")
        print("  2. PREDICATO_URL environment variable is set")
        print("     export PREDICATO_URL=http://localhost:8080")
        sys.exit(1)

    try:
        if args.mode == "simple":
            ingest_simple(client, articles, args.group_id, args.delay)
        elif args.mode == "two-stage":
            ingest_two_stage(client, articles, args.group_id, args.delay)

        print("\n" + "=" * 60)
        print("Ingestion complete!")
        print("=" * 60)

        print("\nExample: Search the knowledge graph")
        print("-" * 40)
        print(f"""
from predicato import PredicatoClient

with PredicatoClient() as client:
    results = client.search(
        query="hypertension treatment",
        group_id="{args.group_id}",
        limit=10
    )
    for node in results.nodes:
        print(f"- {{node.name}} ({{node.type}})")
""")

    finally:
        client.close()


if __name__ == "__main__":
    main()
