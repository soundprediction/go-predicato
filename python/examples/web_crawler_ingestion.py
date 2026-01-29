#!/usr/bin/env python3
"""
Web Crawler Ingestion Example.

This example demonstrates how to use the Predicato web crawler to discover
and ingest web content into the knowledge graph.

This mirrors the Go `crawl-and-index` command from the humn microservice.

Requirements:
    pip install predicato[crawler]

Usage:
    # Basic crawl (keyword-based relevance)
    python web_crawler_ingestion.py --urls https://example.com/health

    # With LLM-based relevance analysis (uses predicato server's configured models)
    python web_crawler_ingestion.py --urls https://example.com/health --use-llm

    # From a URL file
    python web_crawler_ingestion.py --file urls.csv --group-id health-kb

Environment Variables:
    PREDICATO_URL: Predicato server URL (default: http://localhost:8080)
    PREDICATO_GROUP_ID: Default group ID for ingestion

Note:
    When using --use-llm, the predicato server's configured NLP models are used
    for relevance analysis and source extraction. No additional API configuration
    is needed on the client side.
"""

import argparse
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Crawl URLs and ingest content into Predicato knowledge graph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # URL input options
    url_group = parser.add_mutually_exclusive_group(required=True)
    url_group.add_argument(
        "--urls",
        nargs="+",
        help="Starting URLs to crawl",
    )
    url_group.add_argument(
        "--file",
        help="CSV or text file containing URLs (one per line)",
    )

    # Predicato options
    parser.add_argument(
        "--predicato-url",
        default=os.environ.get("PREDICATO_URL", "http://localhost:8080"),
        help="Predicato server URL",
    )
    parser.add_argument(
        "--group-id",
        default=os.environ.get("PREDICATO_GROUP_ID", "web-crawl"),
        help="Group ID for data isolation in Predicato",
    )

    # Crawl options
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum crawl depth from seed URLs (default: 3)",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--no-filter-admin",
        action="store_true",
        help="Don't filter administrative URLs",
    )
    parser.add_argument(
        "--no-filter-toplevel",
        action="store_true",
        help="Don't filter top-level domain URLs",
    )

    # LLM options (uses predicato server's configured models)
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for relevance analysis (via predicato server)",
    )
    parser.add_argument(
        "--extract-source",
        action="store_true",
        help="Extract source metadata using LLM (implies --use-llm)",
    )

    # Ingestion options
    parser.add_argument(
        "--single-stage",
        action="store_true",
        help="Use single-stage ingestion instead of two-stage",
    )
    parser.add_argument(
        "--skip-resolution",
        action="store_true",
        help="Skip entity resolution during promotion (faster but may create duplicates)",
    )

    # Output options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Crawl only, don't ingest into Predicato",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Import predicato modules
    try:
        from predicato import PredicatoClient
        from predicato.crawler import (
            CrawlConfig,
            WebCrawler,
            crawl_and_ingest,
            read_urls_from_file,
        )
    except ImportError as e:
        logger.error(
            f"Failed to import predicato modules: {e}\n"
            "Make sure predicato is installed with crawler extras:\n"
            "  pip install predicato[crawler]"
        )
        sys.exit(1)

    # Get URLs
    if args.urls:
        seed_urls = args.urls
    else:
        logger.info(f"Reading URLs from: {args.file}")
        seed_urls = read_urls_from_file(args.file)

    if not seed_urls:
        logger.error("No URLs provided")
        sys.exit(1)

    logger.info(f"Starting crawl with {len(seed_urls)} seed URL(s)")

    # Build configurations
    crawl_config = CrawlConfig(
        max_depth=args.max_depth,
        request_delay=args.request_delay,
        filter_admin_urls=not args.no_filter_admin,
        filter_toplevel_urls=not args.no_filter_toplevel,
        use_llm=args.use_llm or args.extract_source,
        extract_source=args.extract_source,
    )

    if crawl_config.use_llm:
        logger.info("LLM mode enabled - will use predicato server for NLP operations")

    # Dry run mode - just crawl without ingestion
    if args.dry_run:
        logger.info("Dry run mode - crawling only, not ingesting")

        # For dry run without LLM, we use keyword-based filtering only
        with WebCrawler(config=crawl_config) as crawler:
            summary = crawler.crawl(seed_urls)

        print("\n" + "=" * 60)
        print("CRAWL SUMMARY (Dry Run)")
        print("=" * 60)
        print(f"Seed URLs:        {len(seed_urls)}")
        print(f"Total crawled:    {summary.total_crawled}")
        print(f"Relevant pages:   {summary.relevant_count}")
        print(f"Errors:           {summary.error_count}")
        print(f"Duration:         {summary.completed_at - summary.started_at}")
        print()
        print("Relevant URLs found:")
        for result in summary.results:
            source_info = ""
            if result.source:
                source_info = f" [{result.source.name}]"
            print(f"  - {result.url}{source_info}")

        return

    # Full crawl and ingest
    logger.info(f"Connecting to Predicato at: {args.predicato_url}")
    logger.info(f"Using group ID: {args.group_id}")

    try:
        with PredicatoClient(base_url=args.predicato_url) as client:
            stats = crawl_and_ingest(
                client,
                seed_urls,
                group_id=args.group_id,
                crawl_config=crawl_config,
                use_two_stage=not args.single_stage,
                skip_resolution=args.skip_resolution,
            )

        print("\n" + "=" * 60)
        print("CRAWL AND INGEST SUMMARY")
        print("=" * 60)
        print(f"Seed URLs:        {stats['seed_urls']}")
        print(f"Total crawled:    {stats['crawled']}")
        print(f"Relevant pages:   {stats['relevant']}")
        print(f"Ingested:         {stats['ingested']}")
        print(f"Errors:           {stats['errors']}")
        print(f"Started:          {stats['started_at']}")
        print(f"Completed:        {stats['completed_at']}")
        print()
        print(f"Content indexed to group: {args.group_id}")

    except Exception as e:
        logger.error(f"Failed to connect to Predicato: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
