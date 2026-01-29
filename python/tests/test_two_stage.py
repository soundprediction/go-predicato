"""
Tests for two-stage ingestion (extract_to_facts and promote_to_graph).
"""

from datetime import datetime, timezone

import pytest
from pytest_httpx import HTTPXMock

from predicato import AsyncPredicatoClient, PredicatoClient
from predicato.exceptions import ValidationError
from predicato.models import (
    ExtractedEdge,
    ExtractedNode,
    ExtractEpisodeRequest,
    ExtractionResults,
    PromoteToGraphRequest,
    PromoteToGraphResults,
)


class TestTwoStageModels:
    """Tests for two-stage ingestion models."""

    def test_extract_episode_request(self):
        """Test creating an extract episode request."""
        request = ExtractEpisodeRequest(
            name="Test Episode",
            content="This is test content about Alice and Bob.",
            group_id="test-group",
        )
        assert request.name == "Test Episode"
        assert request.group_id == "test-group"
        assert request.use_yaml is False

    def test_extract_episode_request_with_options(self):
        """Test creating a request with all options."""
        request = ExtractEpisodeRequest(
            id="ep-123",
            name="Test Episode",
            content="Content",
            source="document",
            group_id="test-group",
            entity_types={"Person": {"description": "A person"}},
            excluded_entity_types=["Location"],
            use_yaml=True,
        )
        assert request.id == "ep-123"
        assert request.entity_types == {"Person": {"description": "A person"}}
        assert request.excluded_entity_types == ["Location"]
        assert request.use_yaml is True

    def test_extracted_node(self):
        """Test creating an extracted node."""
        node = ExtractedNode(
            id="node-123",
            source_id="ep-123",
            group_id="test-group",
            name="Alice",
            type="Person",
            description="A software engineer",
        )
        assert node.id == "node-123"
        assert node.name == "Alice"
        assert node.type == "Person"

    def test_extracted_edge(self):
        """Test creating an extracted edge."""
        edge = ExtractedEdge(
            id="edge-123",
            source_id="ep-123",
            group_id="test-group",
            source_node_name="Alice",
            target_node_name="Bob",
            relation="KNOWS",
            description="Alice knows Bob from work",
            weight=0.9,
        )
        assert edge.id == "edge-123"
        assert edge.relation == "KNOWS"
        assert edge.weight == 0.9

    def test_extraction_results(self):
        """Test creating extraction results."""
        results = ExtractionResults(
            source_id="ep-123",
            extracted_nodes=[
                ExtractedNode(
                    id="n1",
                    source_id="ep-123",
                    group_id="g1",
                    name="Alice",
                    type="Person",
                )
            ],
            extracted_edges=[
                ExtractedEdge(
                    id="e1",
                    source_id="ep-123",
                    group_id="g1",
                    source_node_name="Alice",
                    target_node_name="Bob",
                    relation="KNOWS",
                )
            ],
            chunk_count=1,
        )
        assert results.source_id == "ep-123"
        assert len(results.extracted_nodes) == 1
        assert len(results.extracted_edges) == 1

    def test_promote_to_graph_request(self):
        """Test creating a promote to graph request."""
        request = PromoteToGraphRequest(source_id="ep-123")
        assert request.source_id == "ep-123"
        assert request.skip_resolution is False

    def test_promote_to_graph_request_with_options(self):
        """Test creating a request with skip options."""
        request = PromoteToGraphRequest(
            source_id="ep-123",
            skip_reflexion=True,
            skip_resolution=True,
            skip_attributes=True,
            skip_edge_resolution=True,
        )
        assert request.skip_reflexion is True
        assert request.skip_resolution is True


class TestSyncExtractToFacts:
    """Tests for sync extract_to_facts method."""

    def test_extract_to_facts_success(self, httpx_mock: HTTPXMock):
        """Test extracting facts successfully."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8080/api/v1/ingest/extract",
            json={
                "success": True,
                "source_id": "ep-123",
                "extracted_nodes": [
                    {
                        "id": "n1",
                        "source_id": "ep-123",
                        "group_id": "test-group",
                        "name": "Alice",
                        "type": "Person",
                        "description": "A developer",
                        "chunk_index": 0,
                    }
                ],
                "extracted_edges": [
                    {
                        "id": "e1",
                        "source_id": "ep-123",
                        "group_id": "test-group",
                        "source_node_name": "Alice",
                        "target_node_name": "Project",
                        "relation": "WORKS_ON",
                        "description": "Alice works on the project",
                        "weight": 1.0,
                        "chunk_index": 0,
                    }
                ],
                "chunk_count": 1,
                "extraction_time": "1.5s",
            },
            status_code=200,
        )

        with PredicatoClient(base_url="http://localhost:8080") as client:
            result = client.extract_to_facts(
                name="Test Episode",
                content="Alice is a developer working on the project.",
                group_id="test-group",
            )

        assert result.source_id == "ep-123"
        assert len(result.extracted_nodes) == 1
        assert result.extracted_nodes[0].name == "Alice"
        assert len(result.extracted_edges) == 1
        assert result.extracted_edges[0].relation == "WORKS_ON"
        assert result.chunk_count == 1

    def test_extract_to_facts_validates_content_size(self):
        """Test that content size is validated."""
        with PredicatoClient(base_url="http://localhost:8080") as client:
            with pytest.raises(ValidationError) as exc_info:
                client.extract_to_facts(
                    name="Test",
                    content="x" * (1_048_576 + 1),
                    group_id="test-group",
                )
            assert "1MB" in str(exc_info.value)

    def test_extract_to_facts_requires_group_id(self, httpx_mock: HTTPXMock):
        """Test that group_id is required."""
        with PredicatoClient(base_url="http://localhost:8080") as client:
            with pytest.raises(ValidationError) as exc_info:
                client.extract_to_facts(
                    name="Test",
                    content="Content",
                )
            assert "group_id" in str(exc_info.value)


class TestSyncPromoteToGraph:
    """Tests for sync promote_to_graph method."""

    def test_promote_to_graph_success(self, httpx_mock: HTTPXMock):
        """Test promoting to graph successfully."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8080/api/v1/ingest/promote",
            json={
                "success": True,
                "episode": {
                    "uuid": "ep-123",
                    "group_id": "test-group",
                    "name": "Test Episode",
                    "type": "episodic",
                    "created_at": "2024-01-15T10:30:00Z",
                },
                "nodes": [
                    {
                        "uuid": "n1",
                        "group_id": "test-group",
                        "name": "Alice",
                        "type": "entity",
                        "entity_type": "Person",
                        "created_at": "2024-01-15T10:30:00Z",
                    }
                ],
                "edges": [
                    {
                        "uuid": "e1",
                        "group_id": "test-group",
                        "source_node_uuid": "n1",
                        "target_node_uuid": "n2",
                        "type": "entity",
                        "name": "WORKS_ON",
                        "created_at": "2024-01-15T10:30:00Z",
                    }
                ],
                "episodic_edges": [],
                "communities": [],
                "community_edges": [],
            },
            status_code=200,
        )

        with PredicatoClient(base_url="http://localhost:8080") as client:
            result = client.promote_to_graph(source_id="ep-123")

        assert result.episode is not None
        assert result.episode.uuid == "ep-123"
        assert len(result.nodes) == 1
        assert result.nodes[0].name == "Alice"
        assert len(result.edges) == 1
        assert result.edges[0].name == "WORKS_ON"

    def test_promote_to_graph_requires_source_id(self):
        """Test that source_id is required."""
        with PredicatoClient(base_url="http://localhost:8080") as client:
            with pytest.raises(ValidationError) as exc_info:
                client.promote_to_graph(source_id="")
            assert "source_id" in str(exc_info.value)


class TestAsyncExtractToFacts:
    """Tests for async extract_to_facts method."""

    @pytest.mark.asyncio
    async def test_extract_to_facts_success(self, httpx_mock: HTTPXMock):
        """Test extracting facts successfully."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8080/api/v1/ingest/extract",
            json={
                "success": True,
                "source_id": "ep-123",
                "extracted_nodes": [
                    {
                        "id": "n1",
                        "source_id": "ep-123",
                        "group_id": "test-group",
                        "name": "Alice",
                        "type": "Person",
                        "description": "",
                        "chunk_index": 0,
                    }
                ],
                "extracted_edges": [],
                "chunk_count": 1,
            },
            status_code=200,
        )

        async with AsyncPredicatoClient(base_url="http://localhost:8080") as client:
            result = await client.extract_to_facts(
                name="Test Episode",
                content="Alice is a developer.",
                group_id="test-group",
            )

        assert result.source_id == "ep-123"
        assert len(result.extracted_nodes) == 1


class TestAsyncPromoteToGraph:
    """Tests for async promote_to_graph method."""

    @pytest.mark.asyncio
    async def test_promote_to_graph_success(self, httpx_mock: HTTPXMock):
        """Test promoting to graph successfully."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:8080/api/v1/ingest/promote",
            json={
                "success": True,
                "nodes": [],
                "edges": [],
                "episodic_edges": [],
                "communities": [],
                "community_edges": [],
            },
            status_code=200,
        )

        async with AsyncPredicatoClient(base_url="http://localhost:8080") as client:
            result = await client.promote_to_graph(source_id="ep-123")

        assert isinstance(result, PromoteToGraphResults)
