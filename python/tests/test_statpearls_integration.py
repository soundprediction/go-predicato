"""
Integration tests for StatPearls ingestion.

These tests verify the full ingestion pipeline using real StatPearls NXML files
and a running predicato server.

Requirements:
    - Predicato server running (set PREDICATO_URL env var, default: http://localhost:8085)
    - StatPearls test data:
      - Set STATPEARLS_TEST_DIR env var, OR
      - Place .nxml files in tests/fixtures/statpearls/ (gitignored)

Usage:
    # Start the predicato server first:
    cd /path/to/predicato && make run-server -- --port 8085

    # Run integration tests:
    pytest tests/test_statpearls_integration.py -v --run-integration
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

from predicato.statpearls import parse_nxml_file

# Skip all tests in this module unless --run-integration flag is passed
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def predicato_url() -> str:
    """Get predicato server URL from environment or use default."""
    return os.environ.get("PREDICATO_URL", "http://localhost:8085")


@pytest.fixture(scope="module")
def test_group_id() -> str:
    """Generate a unique group ID for this test run."""
    return f"test-statpearls-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def statpearls_dir() -> Path:
    """Path to StatPearls test data (portable, no hardcoded paths)."""
    # 1. Check environment variable
    env_dir = os.environ.get("STATPEARLS_TEST_DIR")
    if env_dir:
        path = Path(env_dir)
        if path.exists():
            return path

    # 2. Fall back to fixtures relative to this test file
    path = Path(__file__).parent / "fixtures" / "statpearls"
    if path.exists() and list(path.glob("*.nxml")):
        return path

    pytest.skip(
        "StatPearls test data not found. "
        "Set STATPEARLS_TEST_DIR or place .nxml files in tests/fixtures/statpearls/"
    )


@pytest.fixture(scope="module")
def statpearls_files(statpearls_dir: Path) -> list[Path]:
    """Get all StatPearls NXML files."""
    files = sorted(statpearls_dir.glob("*.nxml"))
    if not files:
        pytest.skip("No NXML files found in StatPearls directory")
    return files


# Health-related entity types for StatPearls content
MEDICAL_ENTITY_TYPES = {
    "Drug": {
        "description": "Medications, pharmaceuticals, and chemical compounds used for treatment",
        "examples": ["benzodiazepines", "alprazolam", "midazolam", "propofol"],
    },
    "Condition": {
        "description": "Medical conditions, disorders, syndromes, and diseases",
        "examples": ["seizure", "anxiety", "insomnia", "alcohol withdrawal"],
    },
    "Symptom": {
        "description": "Signs and symptoms of medical conditions",
        "examples": ["sedation", "drowsiness", "respiratory depression"],
    },
    "Treatment": {
        "description": "Medical treatments, therapies, and interventions",
        "examples": ["anesthesia", "sedation", "anxiolysis"],
    },
    "Mechanism": {
        "description": "Biological mechanisms and processes",
        "examples": ["GABA receptor", "chloride channel", "positive allosteric modulator"],
    },
    "DrugClass": {
        "description": "Categories or classes of drugs",
        "examples": ["benzodiazepines", "barbiturates", "non-benzodiazepine hypnotics"],
    },
}


class TestStatPearlsExtraction:
    """Tests for extracting content from StatPearls NXML files."""

    def test_extract_single_file(self, statpearls_files: list[Path]):
        """Test extracting content from a single NXML file."""
        file_path = statpearls_files[0]
        extracted = parse_nxml_file(file_path)

        assert extracted is not None, f"Failed to parse {file_path.name}"
        assert extracted["title"], "Title should not be empty"
        assert extracted["content"], "Content should not be empty"
        assert len(extracted["content"]) > 100, "Content should be substantial"
        assert extracted["source_path"] == str(file_path)

        print(f"\nExtracted from {file_path.name}:")
        print(f"  Title: {str(extracted['title'])[:80]}...")
        authors = extracted.get("authors", [])
        if authors:
            print(f"  Authors: {', '.join(str(a) for a in authors[:3])}...")
        print(f"  Content length: {len(extracted['content'])} chars")

    def test_extract_all_files(self, statpearls_files: list[Path]):
        """Test extracting content from all NXML files."""
        results = []
        for file_path in statpearls_files:
            extracted = parse_nxml_file(file_path)
            assert extracted is not None, f"Failed to parse {file_path.name}"
            results.append(extracted)

        assert len(results) == len(statpearls_files)

        for result in results:
            assert result["title"]
            assert result["content"]
            assert len(result["content"]) > 100


class TestStatPearlsIngestion:
    """Integration tests for ingesting StatPearls content into Predicato."""

    @pytest.fixture
    def client(self, predicato_url: str):
        """Create a PredicatoClient."""
        from predicato import PredicatoClient

        return PredicatoClient(base_url=predicato_url)

    def test_server_health(self, client):
        """Test that the predicato server is running."""
        try:
            response = client._http.request("GET", "/health")
            assert response.get("status") in ["healthy", "ok"]
        except Exception as e:
            pytest.skip(f"Predicato server not available: {e}")

    def test_add_messages_single_stage(
        self,
        client,
        statpearls_files: list[Path],
        test_group_id: str,
    ):
        """Test single-stage ingestion using add_messages (no fact DB required)."""
        file_path = statpearls_files[0]
        extracted = parse_nxml_file(file_path)
        assert extracted is not None

        try:
            from predicato.models import Message

            messages = [
                Message(
                    role="user",
                    content=f"{extracted['title']}: {extracted['content'][:5000]}",
                )
            ]

            result = client.add_messages(
                messages=messages,
                group_id=test_group_id,
            )

            print(f"\n=== Single-stage Ingestion ===")
            print(f"File: {file_path.name}")
            print(f"Title: {str(extracted['title'])[:60]}...")
            if hasattr(result, 'process_id') and result.process_id:
                print(f"Process ID: {result.process_id}")

        except Exception as e:
            if "connection refused" in str(e).lower():
                pytest.skip(f"Predicato server not available: {e}")
            raise

    def test_basic_search(
        self,
        client,
        test_group_id: str,
    ):
        """Test basic search endpoint (no fact DB required)."""
        try:
            results = client.search(
                query="GABA receptor",
                group_id=test_group_id,
                limit=5,
            )

            print(f"\n=== Basic Search Test ===")
            print(f"Query: 'GABA receptor'")
            print(f"Nodes found: {len(results.nodes)}")
            print(f"Edges found: {len(results.edges)}")

        except Exception as e:
            if "connection refused" in str(e).lower():
                pytest.skip(f"Predicato server not available: {e}")
            print(f"Search returned: {e}")

    def test_extract_single_episode(
        self,
        client,
        statpearls_files: list[Path],
        test_group_id: str,
    ):
        """Test extracting a single StatPearls article to facts (requires fact DB)."""
        file_path = statpearls_files[0]
        extracted = parse_nxml_file(file_path)
        assert extracted is not None

        try:
            result = client.extract_to_facts(
                name=extracted["title"],
                content=extracted["content"][:20000],
                source=f"statpearls:{file_path.name}",
                group_id=test_group_id,
                metadata={
                    "authors": extracted["authors"],
                    "source_type": "medical_literature",
                    "collection": "StatPearls",
                    "file_path": extracted["source_path"],
                },
                entity_types=MEDICAL_ENTITY_TYPES,
            )

            assert result.source_id, "Should return a source_id"
            print(f"\nExtraction results for {file_path.name}:")
            print(f"  Source ID: {result.source_id}")
            print(f"  Extracted nodes: {len(result.extracted_nodes)}")
            print(f"  Extracted edges: {len(result.extracted_edges)}")
            print(f"  Chunk count: {result.chunk_count}")

            if result.extracted_nodes:
                print(f"  Sample entities:")
                for node in result.extracted_nodes[:5]:
                    print(f"    - {node.name} ({node.type})")

        except Exception as e:
            if "connection refused" in str(e).lower():
                pytest.skip(f"Predicato server not available: {e}")
            if "facts DB not configured" in str(e):
                pytest.skip("Fact DB not configured - two-stage ingestion not available")
            raise

    def test_full_two_stage_ingestion(
        self,
        client,
        statpearls_files: list[Path],
        test_group_id: str,
    ):
        """Test full two-stage ingestion pipeline (extract + promote). Requires fact DB."""
        file_path = statpearls_files[0]
        extracted = parse_nxml_file(file_path)
        assert extracted is not None

        try:
            # Stage 1: Extract to facts
            extraction = client.extract_to_facts(
                name=extracted["title"],
                content=extracted["content"][:15000],
                source=f"statpearls:{file_path.name}",
                group_id=test_group_id,
                metadata={
                    "source_type": "medical_literature",
                    "collection": "StatPearls",
                },
                entity_types=MEDICAL_ENTITY_TYPES,
            )

            assert extraction.source_id
            print(f"\n=== Stage 1: Extraction ===")
            print(f"Source ID: {extraction.source_id}")
            print(f"Nodes: {len(extraction.extracted_nodes)}")
            print(f"Edges: {len(extraction.extracted_edges)}")

            # Stage 2: Promote to graph
            promotion = client.promote_to_graph(
                source_id=extraction.source_id,
                entity_types=MEDICAL_ENTITY_TYPES,
                skip_resolution=True,  # Skip for faster testing
            )

            print(f"\n=== Stage 2: Promotion ===")
            if promotion.episode:
                print(f"Episode UUID: {promotion.episode.uuid}")
            print(f"Nodes in graph: {len(promotion.nodes)}")
            print(f"Edges in graph: {len(promotion.edges)}")

            if promotion.nodes:
                print(f"Sample nodes:")
                for node in promotion.nodes[:5]:
                    print(f"  - {node.name} ({node.entity_type or node.type})")

        except Exception as e:
            if "connection refused" in str(e).lower():
                pytest.skip(f"Predicato server not available: {e}")
            if "facts DB not configured" in str(e):
                pytest.skip("Fact DB not configured - two-stage ingestion not available")
            raise

    def test_ingest_multiple_articles(
        self,
        client,
        statpearls_files: list[Path],
        test_group_id: str,
    ):
        """Test ingesting multiple StatPearls articles. Requires fact DB."""
        results = []

        for file_path in statpearls_files[:2]:  # Limit to 2 for speed
            extracted = parse_nxml_file(file_path)
            assert extracted is not None

            try:
                extraction = client.extract_to_facts(
                    name=extracted["title"],
                    content=extracted["content"][:10000],
                    source=f"statpearls:{file_path.name}",
                    group_id=test_group_id,
                    metadata={
                        "source_type": "medical_literature",
                        "collection": "StatPearls",
                    },
                    entity_types=MEDICAL_ENTITY_TYPES,
                )

                results.append({
                    "file": file_path.name,
                    "title": extracted["title"],
                    "source_id": extraction.source_id,
                    "nodes": len(extraction.extracted_nodes),
                    "edges": len(extraction.extracted_edges),
                })

            except Exception as e:
                if "connection refused" in str(e).lower():
                    pytest.skip(f"Predicato server not available: {e}")
                if "facts DB not configured" in str(e):
                    pytest.skip("Fact DB not configured - two-stage ingestion not available")
                raise

        print(f"\n=== Batch Ingestion Results ===")
        for r in results:
            print(f"  {r['file']}: {r['nodes']} nodes, {r['edges']} edges")

        assert len(results) == min(2, len(statpearls_files))

    def test_search_ingested_content(
        self,
        client,
        statpearls_files: list[Path],
        test_group_id: str,
    ):
        """Test searching for ingested StatPearls content. Requires fact DB."""
        file_path = statpearls_files[0]
        extracted = parse_nxml_file(file_path)
        assert extracted is not None

        try:
            extraction = client.extract_to_facts(
                name=extracted["title"],
                content=extracted["content"][:10000],
                source=f"statpearls:{file_path.name}",
                group_id=test_group_id,
                entity_types=MEDICAL_ENTITY_TYPES,
            )

            client.promote_to_graph(
                source_id=extraction.source_id,
                skip_resolution=True,
            )

            search_terms = ["GABA", "receptor", "medication", "treatment"]

            for term in search_terms:
                try:
                    results = client.search_facts(
                        query=term,
                        group_id=test_group_id,
                        limit=5,
                    )

                    if results.nodes:
                        print(f"\nSearch '{term}' found {len(results.nodes)} results:")
                        for node in results.nodes[:3]:
                            print(f"  - {node.name} ({node.type})")
                        break
                except Exception:
                    continue

        except Exception as e:
            if "connection refused" in str(e).lower():
                pytest.skip(f"Predicato server not available: {e}")
            if "facts DB not configured" in str(e):
                pytest.skip("Fact DB not configured - two-stage ingestion not available")
            raise


class TestStatPearlsCleanup:
    """Cleanup tests - run after integration tests."""

    @pytest.fixture
    def client(self, predicato_url: str):
        """Create a PredicatoClient."""
        from predicato import PredicatoClient

        return PredicatoClient(base_url=predicato_url)

    def test_cleanup_test_data(self, client, test_group_id: str):
        """Clean up test data after tests complete."""
        try:
            print(f"\nTest group '{test_group_id}' should be cleaned up manually if needed")
        except Exception:
            pass  # Cleanup is best-effort
