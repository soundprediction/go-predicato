package driver

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/soundprediction/predicato/pkg/types"
)

func TestGraphProvider(t *testing.T) {
	providers := []GraphProvider{
		GraphProviderNeo4j,
		GraphProviderFalkorDB,
		GraphProviderLadybug,
	}

	for _, provider := range providers {
		t.Run(string(provider), func(t *testing.T) {
			// Test that provider constants are defined
			if string(provider) == "" {
				t.Errorf("Provider %s should not be empty", provider)
			}
		})
	}
}

func TestGetRangeIndices(t *testing.T) {
	tests := []struct {
		provider GraphProvider
		expected int // expected number of indices
	}{
		{GraphProviderNeo4j, 20},   // Neo4j has 20 range indices
		{GraphProviderFalkorDB, 6}, // FalkorDB has 6 range indices
		{GraphProviderLadybug, 0},  // ladybug has 0 range indices
	}

	for _, tt := range tests {
		t.Run(string(tt.provider), func(t *testing.T) {
			indices := GetRangeIndices(tt.provider)
			if len(indices) != tt.expected {
				t.Errorf("GetRangeIndices(%s) returned %d indices, expected %d",
					tt.provider, len(indices), tt.expected)
			}

			// Check that all indices contain CREATE INDEX
			if tt.provider != GraphProviderLadybug {
				for _, index := range indices {
					if !strings.Contains(index, "CREATE INDEX") {
						t.Errorf("Index should contain 'CREATE INDEX': %s", index)
					}
				}
			}
		})
	}
}

func TestGetFulltextIndices(t *testing.T) {
	tests := []struct {
		provider GraphProvider
		expected int // expected number of indices
	}{
		{GraphProviderNeo4j, 4},    // Neo4j has 4 fulltext indices
		{GraphProviderFalkorDB, 4}, // FalkorDB has 4 fulltext indices
		{GraphProviderLadybug, 4},  // ladybug has 4 fulltext indices
	}

	for _, tt := range tests {
		t.Run(string(tt.provider), func(t *testing.T) {
			indices := GetFulltextIndices(tt.provider)
			if len(indices) != tt.expected {
				t.Errorf("GetFulltextIndices(%s) returned %d indices, expected %d",
					tt.provider, len(indices), tt.expected)
			}

			// Check that all indices are appropriate for the provider
			for _, index := range indices {
				switch tt.provider {
				case GraphProviderNeo4j:
					if !strings.Contains(index, "FULLTEXT INDEX") {
						t.Errorf("Neo4j index should contain 'FULLTEXT INDEX': %s", index)
					}
				case GraphProviderFalkorDB:
					if !strings.Contains(index, "FULLTEXT INDEX") {
						t.Errorf("FalkorDB index should contain 'FULLTEXT INDEX': %s", index)
					}
				case GraphProviderLadybug:
					if !strings.Contains(index, "CREATE_FTS_INDEX") {
						t.Errorf("ladybug index should contain 'CREATE_FTS_INDEX': %s", index)
					}
				}
			}
		})
	}
}

func TestGetNodesQuery(t *testing.T) {
	tests := []struct {
		provider  GraphProvider
		indexName string
		query     string
		limit     int
		contains  string
	}{
		{GraphProviderNeo4j, "node_name_and_summary", "test", 10, "db.index.fulltext.queryNodes"},
		{GraphProviderFalkorDB, "node_name_and_summary", "test", 10, "db.idx.fulltext.queryNodes"},
		{GraphProviderLadybug, "node_name_and_summary", "test", 10, "QUERY_FTS_INDEX"},
	}

	for _, tt := range tests {
		t.Run(string(tt.provider), func(t *testing.T) {
			query := GetNodesQuery(tt.indexName, tt.query, tt.limit, tt.provider)
			if !strings.Contains(query, tt.contains) {
				t.Errorf("Query should contain '%s': %s", tt.contains, query)
			}
		})
	}
}

func TestGetVectorCosineFuncQuery(t *testing.T) {
	tests := []struct {
		provider GraphProvider
		vec1     string
		vec2     string
		contains string
	}{
		{GraphProviderNeo4j, "n.embedding", "m.embedding", "vector.similarity.cosine"},
		{GraphProviderFalkorDB, "n.embedding", "m.embedding", "vec.cosineDistance"},
		{GraphProviderLadybug, "n.embedding", "m.embedding", "array_cosine_similarity"},
	}

	for _, tt := range tests {
		t.Run(string(tt.provider), func(t *testing.T) {
			query := GetVectorCosineFuncQuery(tt.vec1, tt.vec2, tt.provider)
			if !strings.Contains(query, tt.contains) {
				t.Errorf("Query should contain '%s': %s", tt.contains, query)
			}
		})
	}
}

func TestQueryBuilder(t *testing.T) {
	builder := NewQueryBuilder(GraphProviderNeo4j)

	// Test provider getter
	if builder.GetProvider() != GraphProviderNeo4j {
		t.Errorf("Expected provider to be Neo4j, got %s", builder.GetProvider())
	}

	// Test provider setter
	builder.SetProvider(GraphProviderLadybug)
	if builder.GetProvider() != GraphProviderLadybug {
		t.Errorf("Expected provider to be ladybug, got %s", builder.GetProvider())
	}

	// Test query building methods
	nodeQuery := builder.BuildFulltextNodeQuery("node_name_and_summary", "test", 10)
	if !strings.Contains(nodeQuery, "QUERY_FTS_INDEX") {
		t.Errorf("ladybug node query should contain 'QUERY_FTS_INDEX': %s", nodeQuery)
	}

	relQuery := builder.BuildFulltextRelationshipQuery("edge_name_and_fact", 10)
	if !strings.Contains(relQuery, "QUERY_FTS_INDEX") {
		t.Errorf("ladybug relationship query should contain 'QUERY_FTS_INDEX': %s", relQuery)
	}

	cosineQuery := builder.BuildCosineSimilarityQuery("n.embedding", "m.embedding")
	if !strings.Contains(cosineQuery, "array_cosine_similarity") {
		t.Errorf("ladybug cosine query should contain 'array_cosine_similarity': %s", cosineQuery)
	}

	// Test index queries
	rangeIndices := builder.GetRangeIndexQueries()
	if len(rangeIndices) != 0 { // ladybug should have 0 range indices
		t.Errorf("ladybug should have 0 range indices, got %d", len(rangeIndices))
	}

	fulltextIndices := builder.GetFulltextIndexQueries()
	if len(fulltextIndices) != 4 { // Should have 4 fulltext indices
		t.Errorf("Should have 4 fulltext indices, got %d", len(fulltextIndices))
	}
}

func TestEscapeQueryString(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"simple", "simple"},
		{"with spaces", "with spaces"},
		{`with "quotes"`, `with \"quotes\"`},
		{"with + and -", `with \+ and \-`},
		{"with (parens)", `with \(parens\)`},
		{"with [brackets]", `with \[brackets\]`},
		{"with {braces}", `with \{braces\}`},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := EscapeQueryString(tt.input)
			if result != tt.expected {
				t.Errorf("EscapeQueryString(%s) = %s, expected %s", tt.input, result, tt.expected)
			}
		})
	}
}

func TestGetNodesQueryEscaping(t *testing.T) {
	// Test that special characters in search queries are properly escaped
	// to prevent query injection attacks
	specialCharQueries := []struct {
		name     string
		query    string
		provider GraphProvider
	}{
		{"quotes in query - Neo4j", `test "injection"`, GraphProviderNeo4j},
		{"quotes in query - FalkorDB", `test "injection"`, GraphProviderFalkorDB},
		{"quotes in query - Ladybug", `test "injection"`, GraphProviderLadybug},
		{"special chars - Neo4j", `test + - ! ( ) { } [ ] ^ ~ * ? : | &`, GraphProviderNeo4j},
		{"backslash - Neo4j", `test\path`, GraphProviderNeo4j},
		{"injection attempt - Neo4j", `test") MATCH (n) DELETE n //`, GraphProviderNeo4j},
		{"injection attempt - Ladybug", `test') CALL db.drop() //`, GraphProviderLadybug},
	}

	for _, tc := range specialCharQueries {
		t.Run(tc.name, func(t *testing.T) {
			result := GetNodesQuery("node_name_and_summary", tc.query, 10, tc.provider)

			// The result should NOT contain the raw query with potential injection
			// Instead, special chars should be escaped
			if strings.Contains(result, tc.query) && strings.ContainsAny(tc.query, `"+-!(){}[]^~*?:|&\`) {
				t.Errorf("Query should have escaped special characters, got: %s", result)
			}

			// Verify the query contains escaped versions of special chars
			if strings.Contains(tc.query, `"`) && !strings.Contains(result, `\"`) {
				t.Errorf("Double quotes should be escaped in query: %s", result)
			}
		})
	}
}

func TestBuildParameterizedQuery(t *testing.T) {
	query := "MATCH (n) WHERE n.uuid = $id RETURN n"
	params := map[string]any{
		"id":        "test-id",
		"database_": "neo4j", // Should be filtered out
		"routing_":  "write", // Should be filtered out
		"valid":     "value",
		"nil_value": nil, // Should be filtered out
	}

	resultQuery, resultParams := BuildParameterizedQuery(query, params)

	// Query should remain unchanged
	if resultQuery != query {
		t.Errorf("Query should remain unchanged")
	}

	// Should only contain valid parameters
	expectedParams := map[string]any{
		"id":    "test-id",
		"valid": "value",
	}

	if len(resultParams) != len(expectedParams) {
		t.Errorf("Expected %d parameters, got %d", len(expectedParams), len(resultParams))
	}

	for key, value := range expectedParams {
		if resultParams[key] != value {
			t.Errorf("Expected param %s = %v, got %v", key, value, resultParams[key])
		}
	}
}

func TestEntityEdgeIntegration(t *testing.T) {
	ctx := context.Background()
	groupID := "test-group"

	// Test data
	node1 := &types.Node{
		Uuid:      "entity-1",
		Name:      "Alice",
		Type:      types.EntityNodeType,
		GroupID:   groupID,
		Summary:   "A software engineer who loves Go programming",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		ValidFrom: time.Now(),
	}

	node2 := &types.Node{
		Uuid:      "entity-2",
		Name:      "Bob",
		Type:      types.EntityNodeType,
		GroupID:   groupID,
		Summary:   "A data scientist working with Python",
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		ValidFrom: time.Now(),
	}

	edge := types.NewEntityEdge(
		"edge-1",
		"entity-1",
		"entity-2",
		groupID,
		"WORKS_WITH",
		types.EntityEdgeType,
	)
	edge.Fact = "Alice works with Bob on the ML project"
	edge.CreatedAt = time.Now()
	edge.ValidFrom = time.Now()

	// Setup ladybug driver
	t.Run("ladybug", func(t *testing.T) {
		// Create temp directory for ladybug database
		tempDir := t.TempDir()
		dbPath := filepath.Join(tempDir, "test_db")

		ladybugDriver, err := NewLadybugDriver(dbPath, 4)
		if err != nil {
			t.Fatalf("Failed to create ladybug driver: %v", err)
		}
		defer ladybugDriver.Close()

		// Create indices
		err = ladybugDriver.CreateIndices(ctx)
		if err != nil {
			t.Fatalf("Failed to create indices: %v", err)
		}

		// Upsert nodes
		err = ladybugDriver.UpsertNode(ctx, node1)
		if err != nil {
			t.Fatalf("Failed to upsert node1: %v", err)
		}

		err = ladybugDriver.UpsertNode(ctx, node2)
		if err != nil {
			t.Fatalf("Failed to upsert node2: %v", err)
		}

		// Upsert edge
		err = ladybugDriver.UpsertEdge(ctx, edge)
		if err != nil {
			t.Fatalf("Failed to upsert edge: %v", err)
		}

		// Retrieve and validate node1
		retrievedNode1, err := ladybugDriver.GetNode(ctx, "entity-1", groupID)
		if err != nil {
			t.Fatalf("Failed to retrieve node1: %v", err)
		}
		if retrievedNode1.Uuid != node1.Uuid {
			t.Errorf("Node1 UUID mismatch: got %s, want %s", retrievedNode1.Uuid, node1.Uuid)
		}
		if retrievedNode1.Name != node1.Name {
			t.Errorf("Node1 Name mismatch: got %s, want %s", retrievedNode1.Name, node1.Name)
		}
		if retrievedNode1.Summary != node1.Summary {
			t.Errorf("Node1 Summary mismatch: got %s, want %s", retrievedNode1.Summary, node1.Summary)
		}

		// Retrieve and validate node2
		retrievedNode2, err := ladybugDriver.GetNode(ctx, "entity-2", groupID)
		if err != nil {
			t.Fatalf("Failed to retrieve node2: %v", err)
		}
		if retrievedNode2.Uuid != node2.Uuid {
			t.Errorf("Node2 UUID mismatch: got %s, want %s", retrievedNode2.Uuid, node2.Uuid)
		}
		if retrievedNode2.Name != node2.Name {
			t.Errorf("Node2 Name mismatch: got %s, want %s", retrievedNode2.Name, node2.Name)
		}

		// Retrieve and validate edge
		retrievedEdge, err := ladybugDriver.GetEdge(ctx, "edge-1", groupID)
		if err != nil {
			t.Fatalf("Failed to retrieve edge: %v", err)
		}
		if retrievedEdge.Uuid != edge.Uuid {
			t.Errorf("Edge UUID mismatch: got %s, want %s", retrievedEdge.Uuid, edge.Uuid)
		}
		if retrievedEdge.SourceID != edge.SourceID {
			t.Errorf("Edge SourceID mismatch: got %s, want %s", retrievedEdge.SourceID, edge.SourceID)
		}
		if retrievedEdge.TargetID != edge.TargetID {
			t.Errorf("Edge TargetID mismatch: got %s, want %s", retrievedEdge.TargetID, edge.TargetID)
		}
		if retrievedEdge.Fact != edge.Fact {
			t.Errorf("Edge Fact mismatch: got %s, want %s", retrievedEdge.Fact, edge.Fact)
		}

		t.Logf("✓ ladybug: Successfully created, upserted, and retrieved 2 nodes and 1 edge")
	})

	// Setup Memgraph driver
	t.Run("Memgraph", func(t *testing.T) {
		// Try to connect to Memgraph on default port
		uri := "bolt://localhost:7687"
		username := ""
		password := ""

		memgraphDriver, err := NewMemgraphDriver(uri, username, password, "memgraph")
		if err != nil {
			t.Skipf("Skipping Memgraph test: cannot connect to %s: %v", uri, err)
			return
		}
		defer memgraphDriver.Close()

		// Verify connectivity
		err = memgraphDriver.VerifyConnectivity(ctx)
		if err != nil {
			t.Skipf("Skipping Memgraph test: cannot verify connectivity: %v", err)
			return
		}

		// Clean up before test
		cleanupQuery := `
			MATCH (n {group_id: $group_id})
			DETACH DELETE n
		`
		_, _, _, _ = memgraphDriver.ExecuteQuery(ctx, cleanupQuery, map[string]any{"group_id": groupID})

		// Create indices
		err = memgraphDriver.CreateIndices(ctx)
		if err != nil {
			t.Logf("Warning: Failed to create indices (may already exist): %v", err)
		}

		// Upsert nodes
		err = memgraphDriver.UpsertNode(ctx, node1)
		if err != nil {
			t.Fatalf("Failed to upsert node1: %v", err)
		}

		err = memgraphDriver.UpsertNode(ctx, node2)
		if err != nil {
			t.Fatalf("Failed to upsert node2: %v", err)
		}

		// Upsert edge
		err = memgraphDriver.UpsertEdge(ctx, edge)
		if err != nil {
			t.Fatalf("Failed to upsert edge: %v", err)
		}

		// Retrieve and validate node1
		retrievedNode1, err := memgraphDriver.GetNode(ctx, "entity-1", groupID)
		if err != nil {
			t.Fatalf("Failed to retrieve node1: %v", err)
		}
		if retrievedNode1.Uuid != node1.Uuid {
			t.Errorf("Node1 UUID mismatch: got %s, want %s", retrievedNode1.Uuid, node1.Uuid)
		}
		if retrievedNode1.Name != node1.Name {
			t.Errorf("Node1 Name mismatch: got %s, want %s", retrievedNode1.Name, node1.Name)
		}

		// Retrieve and validate node2
		retrievedNode2, err := memgraphDriver.GetNode(ctx, "entity-2", groupID)
		if err != nil {
			t.Fatalf("Failed to retrieve node2: %v", err)
		}
		if retrievedNode2.Uuid != node2.Uuid {
			t.Errorf("Node2 UUID mismatch: got %s, want %s", retrievedNode2.Uuid, node2.Uuid)
		}
		if retrievedNode2.Name != node2.Name {
			t.Errorf("Node2 Name mismatch: got %s, want %s", retrievedNode2.Name, node2.Name)
		}

		// Retrieve and validate edge
		retrievedEdge, err := memgraphDriver.GetEdge(ctx, "edge-1", groupID)
		if err != nil {
			t.Fatalf("Failed to retrieve edge: %v", err)
		}
		if retrievedEdge.Uuid != edge.Uuid {
			t.Errorf("Edge UUID mismatch: got %s, want %s", retrievedEdge.Uuid, edge.Uuid)
		}
		if retrievedEdge.SourceID != edge.SourceID {
			t.Errorf("Edge SourceID mismatch: got %s, want %s", retrievedEdge.SourceID, edge.SourceID)
		}
		if retrievedEdge.TargetID != edge.TargetID {
			t.Errorf("Edge TargetID mismatch: got %s, want %s", retrievedEdge.TargetID, edge.TargetID)
		}

		// Clean up after test
		_, _, _, _ = memgraphDriver.ExecuteQuery(ctx, cleanupQuery, map[string]any{"group_id": groupID})

		t.Logf("✓ Memgraph: Successfully created, upserted, and retrieved 2 nodes and 1 edge")
	})
}
