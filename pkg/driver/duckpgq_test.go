//go:build system_duckpgq

package driver_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/parquet-go/parquet-go"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func createTempDuckPGQ(t *testing.T) string {
	t.Helper()
	tempDir := t.TempDir()
	return filepath.Join(tempDir, "duckpgq_test.db")
}

func TestNewDuckPGQDriver(t *testing.T) {
	t.Run("file path", func(t *testing.T) {
		dbPath := createTempDuckPGQ(t)
		d, err := driver.NewDuckPGQDriver(dbPath, 1024)
		require.NoError(t, err)
		require.NotNil(t, d)
		assert.Equal(t, driver.GraphProviderDuckPGQ, d.Provider())
		require.NoError(t, d.Close())
	})

	t.Run("in-memory", func(t *testing.T) {
		d, err := driver.NewDuckPGQDriver("", 1024)
		require.NoError(t, err)
		require.NotNil(t, d)
		require.NoError(t, d.Close())
	})
}

func TestDuckPGQDriverInterface(t *testing.T) {
	var _ driver.GraphDriver = (*driver.DuckPGQDriver)(nil)
}

func TestDuckPGQDriver_CreateIndices(t *testing.T) {
	dbPath := createTempDuckPGQ(t)
	d, err := driver.NewDuckPGQDriver(dbPath, 1024)
	require.NoError(t, err)
	defer d.Close()

	err = d.CreateIndices(context.Background())
	require.NoError(t, err)
}

func TestDuckPGQDriver_UpsertNode(t *testing.T) {
	dbPath := createTempDuckPGQ(t)
	d, err := driver.NewDuckPGQDriver(dbPath, 1024)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	now := time.Now().Truncate(time.Millisecond)
	testNode := &types.Node{
		Uuid:       "test-node-123",
		Name:       "Test Entity",
		Type:       types.EntityNodeType,
		GroupID:    "test-group",
		EntityType: "Person",
		Summary:    "A test entity for UpsertNode",
		CreatedAt:  now,
		UpdatedAt:  now,
	}

	err = d.UpsertNode(ctx, testNode)
	require.NoError(t, err, "UpsertNode should succeed")

	retrievedNode, err := d.GetNode(ctx, testNode.Uuid, testNode.GroupID)
	require.NoError(t, err, "GetNode should succeed")
	require.NotNil(t, retrievedNode, "Retrieved node should not be nil")

	assert.Equal(t, testNode.Uuid, retrievedNode.Uuid)
	assert.Equal(t, testNode.Name, retrievedNode.Name)
	assert.Equal(t, testNode.Type, retrievedNode.Type)
	assert.Equal(t, testNode.GroupID, retrievedNode.GroupID)
	assert.Equal(t, testNode.EntityType, retrievedNode.EntityType)
	assert.Equal(t, testNode.Summary, retrievedNode.Summary)

	// Test update
	testNode.Summary = "Updated summary"
	testNode.UpdatedAt = time.Now().Truncate(time.Millisecond)
	err = d.UpsertNode(ctx, testNode)
	require.NoError(t, err, "Update UpsertNode should succeed")

	updatedNode, err := d.GetNode(ctx, testNode.Uuid, testNode.GroupID)
	require.NoError(t, err)
	require.NotNil(t, updatedNode)
	assert.Equal(t, "Updated summary", updatedNode.Summary)
}

func TestDuckPGQDriver_GetNodes(t *testing.T) {
	dbPath := createTempDuckPGQ(t)
	d, err := driver.NewDuckPGQDriver(dbPath, 1024)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	now := time.Now().Truncate(time.Millisecond)
	nodes := []*types.Node{
		{Uuid: "node-1", Name: "Node 1", Type: types.EntityNodeType, GroupID: "test-group", CreatedAt: now, UpdatedAt: now},
		{Uuid: "node-2", Name: "Node 2", Type: types.EntityNodeType, GroupID: "test-group", CreatedAt: now, UpdatedAt: now},
		{Uuid: "node-3", Name: "Node 3", Type: types.EntityNodeType, GroupID: "test-group", CreatedAt: now, UpdatedAt: now},
	}

	err = d.UpsertNodes(ctx, nodes)
	require.NoError(t, err)

	retrieved, err := d.GetNodes(ctx, []string{"node-1", "node-3"}, "test-group")
	require.NoError(t, err)
	assert.Len(t, retrieved, 2)
}

func TestDuckPGQDriver_DeleteNode(t *testing.T) {
	dbPath := createTempDuckPGQ(t)
	d, err := driver.NewDuckPGQDriver(dbPath, 1024)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	now := time.Now().Truncate(time.Millisecond)
	node := &types.Node{Uuid: "del-node", Name: "To Delete", Type: types.EntityNodeType, GroupID: "test-group", CreatedAt: now, UpdatedAt: now}
	err = d.UpsertNode(ctx, node)
	require.NoError(t, err)

	err = d.DeleteNode(ctx, "del-node", "test-group")
	require.NoError(t, err)

	got, err := d.GetNode(ctx, "del-node", "test-group")
	assert.NoError(t, err)
	assert.Nil(t, got)
}

func TestDuckPGQDriver_UpsertEdge(t *testing.T) {
	dbPath := createTempDuckPGQ(t)
	d, err := driver.NewDuckPGQDriver(dbPath, 1024)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	now := time.Now().Truncate(time.Millisecond)

	// Create source and target nodes
	err = d.UpsertNode(ctx, &types.Node{Uuid: "src-node", Name: "Source", Type: types.EntityNodeType, GroupID: "test-group", CreatedAt: now, UpdatedAt: now})
	require.NoError(t, err)
	err = d.UpsertNode(ctx, &types.Node{Uuid: "tgt-node", Name: "Target", Type: types.EntityNodeType, GroupID: "test-group", CreatedAt: now, UpdatedAt: now})
	require.NoError(t, err)

	testEdge := &types.Edge{
		BaseEdge: types.BaseEdge{
			Uuid:         "test-edge-123",
			GroupID:      "test-group",
			SourceNodeID: "src-node",
			TargetNodeID: "tgt-node",
			CreatedAt:    now,
		},
		SourceID:  "src-node",
		TargetID:  "tgt-node",
		Type:      types.EntityEdgeType,
		UpdatedAt: now,
		Name:      "RELATES_TO",
		Fact:      "A test fact",
	}

	err = d.UpsertEdge(ctx, testEdge)
	require.NoError(t, err)

	retrievedEdge, err := d.GetEdge(ctx, testEdge.Uuid, testEdge.GroupID)
	require.NoError(t, err)
	require.NotNil(t, retrievedEdge)

	assert.Equal(t, testEdge.Uuid, retrievedEdge.Uuid)
	assert.Equal(t, testEdge.Name, retrievedEdge.Name)
	assert.Equal(t, testEdge.Type, retrievedEdge.Type)
	assert.Equal(t, testEdge.GroupID, retrievedEdge.GroupID)
	assert.Equal(t, testEdge.Fact, retrievedEdge.Fact)

	// Test update
	testEdge.Fact = "Updated fact"
	err = d.UpsertEdge(ctx, testEdge)
	require.NoError(t, err)

	updatedEdge, err := d.GetEdge(ctx, testEdge.Uuid, testEdge.GroupID)
	require.NoError(t, err)
	require.NotNil(t, updatedEdge)
	assert.Equal(t, "Updated fact", updatedEdge.Fact)
}

func TestDuckPGQDriver_DeleteEdge(t *testing.T) {
	dbPath := createTempDuckPGQ(t)
	d, err := driver.NewDuckPGQDriver(dbPath, 1024)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	now := time.Now().Truncate(time.Millisecond)
	edge := &types.Edge{
		BaseEdge: types.BaseEdge{
			Uuid: "del-edge", GroupID: "test-group",
			SourceNodeID: "n1", TargetNodeID: "n2", CreatedAt: now,
		},
		SourceID: "n1", TargetID: "n2", Type: types.EntityEdgeType, UpdatedAt: now, Name: "TEST",
	}
	err = d.UpsertEdge(ctx, edge)
	require.NoError(t, err)

	err = d.DeleteEdge(ctx, "del-edge", "test-group")
	require.NoError(t, err)

	got, err := d.GetEdge(ctx, "del-edge", "test-group")
	assert.NoError(t, err)
	assert.Nil(t, got)
}

func TestDuckPGQDriver_SearchNodes(t *testing.T) {
	dbPath := createTempDuckPGQ(t)
	d, err := driver.NewDuckPGQDriver(dbPath, 1024)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	now := time.Now().Truncate(time.Millisecond)
	nodes := []*types.Node{
		{Uuid: "apple-1", Name: "Apple Inc", Type: types.EntityNodeType, GroupID: "test-group", Summary: "Technology company", CreatedAt: now, UpdatedAt: now},
		{Uuid: "banana-1", Name: "Banana Corp", Type: types.EntityNodeType, GroupID: "test-group", Summary: "Fruit company", CreatedAt: now, UpdatedAt: now},
		{Uuid: "apple-2", Name: "Apple Pie", Type: types.EntityNodeType, GroupID: "test-group", Summary: "A delicious dessert", CreatedAt: now, UpdatedAt: now},
	}
	err = d.UpsertNodes(ctx, nodes)
	require.NoError(t, err)

	results, err := d.SearchNodes(ctx, "apple", "test-group", nil)
	require.NoError(t, err)
	assert.Len(t, results, 2, "Should find 2 nodes matching 'apple'")
}

func TestDuckPGQDriver_GetNeighbors(t *testing.T) {
	dbPath := createTempDuckPGQ(t)
	d, err := driver.NewDuckPGQDriver(dbPath, 1024)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	now := time.Now().Truncate(time.Millisecond)

	// Create a small graph: A -> B -> C
	err = d.UpsertNodes(ctx, []*types.Node{
		{Uuid: "A", Name: "Node A", Type: types.EntityNodeType, GroupID: "g1", CreatedAt: now, UpdatedAt: now},
		{Uuid: "B", Name: "Node B", Type: types.EntityNodeType, GroupID: "g1", CreatedAt: now, UpdatedAt: now},
		{Uuid: "C", Name: "Node C", Type: types.EntityNodeType, GroupID: "g1", CreatedAt: now, UpdatedAt: now},
	})
	require.NoError(t, err)

	err = d.UpsertEdges(ctx, []*types.Edge{
		{BaseEdge: types.BaseEdge{Uuid: "e1", GroupID: "g1", SourceNodeID: "A", TargetNodeID: "B", CreatedAt: now}, SourceID: "A", TargetID: "B", Type: types.EntityEdgeType, UpdatedAt: now, Name: "KNOWS"},
		{BaseEdge: types.BaseEdge{Uuid: "e2", GroupID: "g1", SourceNodeID: "B", TargetNodeID: "C", CreatedAt: now}, SourceID: "B", TargetID: "C", Type: types.EntityEdgeType, UpdatedAt: now, Name: "KNOWS"},
	})
	require.NoError(t, err)

	// Distance 1 from A: should get B
	neighbors, err := d.GetNeighbors(ctx, "A", "g1", 1)
	require.NoError(t, err)
	assert.Len(t, neighbors, 1)
	assert.Equal(t, "B", neighbors[0].Uuid)

	// Distance 2 from A: should get B and C
	neighbors, err = d.GetNeighbors(ctx, "A", "g1", 2)
	require.NoError(t, err)
	assert.Len(t, neighbors, 2)
}

func TestDuckPGQDriver_GetStats(t *testing.T) {
	dbPath := createTempDuckPGQ(t)
	d, err := driver.NewDuckPGQDriver(dbPath, 1024)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	now := time.Now().Truncate(time.Millisecond)
	err = d.UpsertNodes(ctx, []*types.Node{
		{Uuid: "s1", Name: "N1", Type: types.EntityNodeType, GroupID: "g1", CreatedAt: now, UpdatedAt: now},
		{Uuid: "s2", Name: "N2", Type: types.EntityNodeType, GroupID: "g1", CreatedAt: now, UpdatedAt: now},
	})
	require.NoError(t, err)

	err = d.UpsertEdge(ctx, &types.Edge{
		BaseEdge: types.BaseEdge{Uuid: "se1", GroupID: "g1", SourceNodeID: "s1", TargetNodeID: "s2", CreatedAt: now},
		SourceID: "s1", TargetID: "s2", Type: types.EntityEdgeType, UpdatedAt: now, Name: "REL",
	})
	require.NoError(t, err)

	stats, err := d.GetStats(ctx, "g1")
	require.NoError(t, err)
	require.NotNil(t, stats)
	assert.Equal(t, int64(2), stats.NodeCount)
	assert.Equal(t, int64(1), stats.EdgeCount)
}

func TestDuckPGQDriver_GetEntityNodesByGroup(t *testing.T) {
	dbPath := createTempDuckPGQ(t)
	d, err := driver.NewDuckPGQDriver(dbPath, 1024)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	now := time.Now().Truncate(time.Millisecond)
	err = d.UpsertNodes(ctx, []*types.Node{
		{Uuid: "ent-1", Name: "E1", Type: types.EntityNodeType, GroupID: "g1", CreatedAt: now, UpdatedAt: now},
		{Uuid: "ep-1", Name: "Ep1", Type: types.EpisodicNodeType, GroupID: "g1", CreatedAt: now, UpdatedAt: now},
		{Uuid: "ent-2", Name: "E2", Type: types.EntityNodeType, GroupID: "g1", CreatedAt: now, UpdatedAt: now},
		{Uuid: "ent-3", Name: "E3", Type: types.EntityNodeType, GroupID: "other", CreatedAt: now, UpdatedAt: now},
	})
	require.NoError(t, err)

	entities, err := d.GetEntityNodesByGroup(ctx, "g1")
	require.NoError(t, err)
	assert.Len(t, entities, 2, "Should get only entity nodes for group g1")
}

func TestDuckPGQDriver_GetAllGroupIDs(t *testing.T) {
	dbPath := createTempDuckPGQ(t)
	d, err := driver.NewDuckPGQDriver(dbPath, 1024)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	now := time.Now().Truncate(time.Millisecond)
	err = d.UpsertNodes(ctx, []*types.Node{
		{Uuid: "g-1", Name: "N1", Type: types.EntityNodeType, GroupID: "group-a", CreatedAt: now, UpdatedAt: now},
		{Uuid: "g-2", Name: "N2", Type: types.EntityNodeType, GroupID: "group-b", CreatedAt: now, UpdatedAt: now},
		{Uuid: "g-3", Name: "N3", Type: types.EntityNodeType, GroupID: "group-a", CreatedAt: now, UpdatedAt: now},
	})
	require.NoError(t, err)

	groups, err := d.GetAllGroupIDs(ctx)
	require.NoError(t, err)
	assert.Len(t, groups, 2)
	assert.Contains(t, groups, "group-a")
	assert.Contains(t, groups, "group-b")
}

func TestDuckPGQDriver_VectorSearch(t *testing.T) {
	dbPath := createTempDuckPGQ(t)
	d, err := driver.NewDuckPGQDriver(dbPath, 3) // small embedding dim for testing
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	now := time.Now().Truncate(time.Millisecond)
	err = d.UpsertNodes(ctx, []*types.Node{
		{Uuid: "v1", Name: "Vec1", Type: types.EntityNodeType, GroupID: "g1", Embedding: []float32{1.0, 0.0, 0.0}, CreatedAt: now, UpdatedAt: now},
		{Uuid: "v2", Name: "Vec2", Type: types.EntityNodeType, GroupID: "g1", Embedding: []float32{0.0, 1.0, 0.0}, CreatedAt: now, UpdatedAt: now},
		{Uuid: "v3", Name: "Vec3", Type: types.EntityNodeType, GroupID: "g1", Embedding: []float32{0.9, 0.1, 0.0}, CreatedAt: now, UpdatedAt: now},
	})
	require.NoError(t, err)

	// Search for vectors similar to [1, 0, 0] - should return v1 first, then v3
	results, err := d.SearchNodesByEmbedding(ctx, []float32{1.0, 0.0, 0.0}, "g1", 2)
	require.NoError(t, err)
	require.Len(t, results, 2)
	assert.Equal(t, "v1", results[0].Uuid)
	assert.Equal(t, "v3", results[1].Uuid)
}

func TestDuckPGQDriver_TimeRange(t *testing.T) {
	dbPath := createTempDuckPGQ(t)
	d, err := driver.NewDuckPGQDriver(dbPath, 1024)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	base := time.Date(2025, 1, 1, 0, 0, 0, 0, time.UTC)
	err = d.UpsertNodes(ctx, []*types.Node{
		{Uuid: "t1", Name: "Old", Type: types.EntityNodeType, GroupID: "g1", CreatedAt: base, UpdatedAt: base},
		{Uuid: "t2", Name: "Mid", Type: types.EntityNodeType, GroupID: "g1", CreatedAt: base.Add(24 * time.Hour), UpdatedAt: base.Add(24 * time.Hour)},
		{Uuid: "t3", Name: "New", Type: types.EntityNodeType, GroupID: "g1", CreatedAt: base.Add(48 * time.Hour), UpdatedAt: base.Add(48 * time.Hour)},
	})
	require.NoError(t, err)

	results, err := d.GetNodesInTimeRange(ctx, base.Add(12*time.Hour), base.Add(36*time.Hour), "g1")
	require.NoError(t, err)
	assert.Len(t, results, 1)
	assert.Equal(t, "t2", results[0].Uuid)
}

func TestDuckPGQDriver_GraphExporter(t *testing.T) {
	dbPath := createTempDuckPGQ(t)
	d, err := driver.NewDuckPGQDriver(dbPath, 1024)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	now := time.Now().Truncate(time.Millisecond)
	err = d.UpsertNodes(ctx, []*types.Node{
		{Uuid: "ex1", Name: "N1", Type: types.EntityNodeType, GroupID: "g1", CreatedAt: now, UpdatedAt: now},
		{Uuid: "ex2", Name: "N2", Type: types.EntityNodeType, GroupID: "g1", CreatedAt: now, UpdatedAt: now},
	})
	require.NoError(t, err)

	err = d.UpsertEdge(ctx, &types.Edge{
		BaseEdge: types.BaseEdge{Uuid: "ex-e1", GroupID: "g1", SourceNodeID: "ex1", TargetNodeID: "ex2", CreatedAt: now},
		SourceID: "ex1", TargetID: "ex2", Type: types.EntityEdgeType, UpdatedAt: now, Name: "REL",
	})
	require.NoError(t, err)

	// Test IterateNodes
	var exporter driver.GraphExporter = d
	var nodeCount int
	err = exporter.IterateNodes(ctx, "g1", func(n *types.Node) error {
		nodeCount++
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, 2, nodeCount)

	// Test IterateEdges
	var edgeCount int
	err = exporter.IterateEdges(ctx, "g1", func(e *types.Edge) error {
		edgeCount++
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, 1, edgeCount)
}

// parquetNode matches the factstore nodes.parquet schema for test generation.
type parquetNode struct {
	CreatedAt   time.Time `parquet:"created_at,timestamp(microsecond)"`
	ID          string    `parquet:"id"`
	SourceID    string    `parquet:"source_id"`
	GroupID     string    `parquet:"group_id"`
	Name        string    `parquet:"name"`
	Type        string    `parquet:"type"`
	Description string    `parquet:"description"`
	Embedding   []float32 `parquet:"embedding,list"`
	ChunkIndex  int32     `parquet:"chunk_index"`
}

// parquetTriple matches the factstore extracted_triples.parquet schema for test generation.
type parquetTriple struct {
	CreatedAt         time.Time `parquet:"created_at,timestamp(microsecond)"`
	ID                string    `parquet:"id"`
	SourceID          string    `parquet:"source_id"`
	GroupID           string    `parquet:"group_id"`
	Subject           string    `parquet:"subject"`
	SubjectType       string    `parquet:"subject_type"`
	Predicate         string    `parquet:"predicate"`
	Object            string    `parquet:"object"`
	ObjectType        string    `parquet:"object_type"`
	Description       string    `parquet:"description"`
	Condition         string    `parquet:"condition"`
	Temporal          string    `parquet:"temporal"`
	Location          string    `parquet:"location"`
	Certainty         string    `parquet:"certainty"`
	Scope             string    `parquet:"scope"`
	SourceAttribution string    `parquet:"source_attribution"`
	Embedding         []float32 `parquet:"embedding,list"`
	Confidence        float64   `parquet:"confidence"`
	ChunkIndex        int32     `parquet:"chunk_index"`
}

// parquetRule matches the factstore extracted_rules.parquet schema.
type parquetRule struct {
	CreatedAt         time.Time `parquet:"created_at,timestamp(microsecond)"`
	ID                string    `parquet:"id"`
	SourceID          string    `parquet:"source_id"`
	Antecedent        string    `parquet:"antecedent"`
	Consequent        string    `parquet:"consequent"`
	Exception         string    `parquet:"exception"`
	RuleType          string    `parquet:"rule_type"`
	Scope             string    `parquet:"scope"`
	SourceAttribution string    `parquet:"source_attribution"`
	Embedding         []float32 `parquet:"embedding,list"`
	Confidence        float64   `parquet:"confidence"`
	ChunkIndex        int32     `parquet:"chunk_index"`
}

func writeTestParquet[T any](t *testing.T, path string, rows []T) {
	t.Helper()
	f, err := os.Create(path)
	require.NoError(t, err)
	defer f.Close()
	w := parquet.NewGenericWriter[T](f)
	_, err = w.Write(rows)
	require.NoError(t, err)
	require.NoError(t, w.Close())
}

func TestDuckPGQDriver_BulkLoadFromParquet(t *testing.T) {
	// Create temp dir with test parquet files
	inputDir := t.TempDir()
	now := time.Now().Truncate(time.Microsecond)
	emb := make([]float32, 4) // Use small embeddings for test
	for i := range emb {
		emb[i] = float32(i) * 0.1
	}

	// Write nodes parquet
	nodes := []parquetNode{
		{ID: "n1", SourceID: "wikidata", GroupID: "test", Name: "Aspirin", Type: "MEDICATION", Description: "pain medication", Embedding: emb, CreatedAt: now},
		{ID: "n2", SourceID: "wikidata", GroupID: "test", Name: "Headache", Type: "CONDITION", Description: "head pain", Embedding: emb, CreatedAt: now},
		{ID: "n3", SourceID: "wikidata", GroupID: "test", Name: "Ibuprofen", Type: "MEDICATION", Description: "anti-inflammatory", Embedding: emb, CreatedAt: now},
	}
	writeTestParquet(t, filepath.Join(inputDir, "nodes.parquet"), nodes)

	// Write triples parquet
	triples := []parquetTriple{
		{
			ID: "t1", SourceID: "wikidata", GroupID: "test",
			Subject: "Aspirin", SubjectType: "MEDICATION",
			Predicate: "TREATS",
			Object:    "Headache", ObjectType: "CONDITION",
			Description: "Aspirin TREATS Headache",
			Certainty:   "established", SourceAttribution: "Wikidata",
			Embedding: emb, Confidence: 0.95, CreatedAt: now,
		},
		{
			ID: "t2", SourceID: "wikidata", GroupID: "test",
			Subject: "Ibuprofen", SubjectType: "MEDICATION",
			Predicate: "TREATS",
			Object:    "Headache", ObjectType: "CONDITION",
			Description: "Ibuprofen TREATS Headache",
			Embedding:   emb, Confidence: 0.75, CreatedAt: now,
		},
		{
			ID: "t3", SourceID: "wikidata", GroupID: "test",
			Subject: "Aspirin", SubjectType: "MEDICATION",
			Predicate: "INTERACTS_WITH",
			Object:    "Unknown Drug", ObjectType: "MEDICATION",
			Description: "Aspirin INTERACTS_WITH Unknown Drug",
			Embedding:   emb, Confidence: 0.5, CreatedAt: now,
		},
	}
	writeTestParquet(t, filepath.Join(inputDir, "extracted_triples.parquet.gz"), triples)

	// Write rules parquet
	rules := []parquetRule{
		{
			ID: "r1", SourceID: "wikidata",
			Antecedent: "patient has headache", Consequent: "aspirin may be used",
			Exception: "patient is on blood thinners",
			RuleType:  "treatment_indication", Scope: "general",
			SourceAttribution: "Wikidata",
			Embedding:         emb, Confidence: 0.9, CreatedAt: now,
		},
	}
	writeTestParquet(t, filepath.Join(inputDir, "extracted_rules.parquet.gz"), rules)

	// Create DuckPGQ driver with small embedding dim
	dbPath := filepath.Join(t.TempDir(), "test_parquet_import.db")
	d, err := driver.NewDuckPGQDriver(dbPath, 4)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()

	// Run BulkLoadFromParquet
	nodesLoaded, edgesLoaded, rulesLoaded, err := d.BulkLoadFromParquet(ctx, inputDir, "test")
	require.NoError(t, err)

	// Verify counts
	assert.Equal(t, int64(3), nodesLoaded, "should load 3 nodes")
	assert.Equal(t, int64(2), edgesLoaded, "should load 2 edges (1 skipped for unresolved 'Unknown Drug')")
	assert.Equal(t, int64(1), rulesLoaded, "should load 1 rule")

	// Verify via GetStats
	stats, err := d.GetStats(ctx, "test")
	require.NoError(t, err)
	assert.Equal(t, int64(3), stats.NodeCount, "stats should show 3 nodes")

	// Verify we can read back a node
	node, err := d.GetNode(ctx, "n1", "test")
	require.NoError(t, err)
	require.NotNil(t, node)
	assert.Equal(t, "Aspirin", node.Name)
	assert.Equal(t, "MEDICATION", node.EntityType)
	assert.Equal(t, "pain medication", node.Content)

	// Verify edge was created correctly
	edge, err := d.GetEdge(ctx, "t1", "test")
	require.NoError(t, err)
	require.NotNil(t, edge)
	assert.Equal(t, "TREATS", edge.Name)
	assert.Equal(t, "n1", edge.SourceID) // Aspirin
	assert.Equal(t, "n2", edge.TargetID) // Headache
}
