package driver_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// getMemgraphConnectionInfo returns connection info from environment or defaults
// Set MEMGRAPH_URI, MEMGRAPH_USER, MEMGRAPH_PASSWORD env vars to override
func getMemgraphConnectionInfo() (uri, user, password string) {
	uri = os.Getenv("MEMGRAPH_URI")
	if uri == "" {
		uri = "bolt://localhost:7687"
	}
	user = os.Getenv("MEMGRAPH_USER")
	if user == "" {
		user = ""
	}
	password = os.Getenv("MEMGRAPH_PASSWORD")
	if password == "" {
		password = ""
	}
	return
}

// skipIfMemgraphUnavailable skips the test if Memgraph is not available
func skipIfMemgraphUnavailable(t *testing.T) *driver.MemgraphDriver {
	t.Helper()

	uri, user, password := getMemgraphConnectionInfo()
	d, err := driver.NewMemgraphDriver(uri, user, password, "memgraph")
	if err != nil {
		t.Skipf("Memgraph not available at %s: %v", uri, err)
		return nil
	}

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = d.VerifyConnectivity(ctx)
	if err != nil {
		d.Close()
		t.Skipf("Memgraph connection failed: %v", err)
		return nil
	}

	err = d.CreateIndices(ctx)
	if err != nil {
		d.Close()
		t.Skipf("Memgraph connection failed: %v", err)
		return nil
	}

	return d
}

func TestNewMemgraphDriver(t *testing.T) {
	t.Run("valid connection", func(t *testing.T) {
		uri, user, password := getMemgraphConnectionInfo()
		d, err := driver.NewMemgraphDriver(uri, user, password, "memgraph")

		if err != nil {
			t.Skipf("Memgraph not available: %v", err)
			return
		}

		require.NotNil(t, d)

		// Test that Close works
		err = d.Close()
		assert.NoError(t, err)
	})

	t.Run("custom database", func(t *testing.T) {
		uri, user, password := getMemgraphConnectionInfo()
		d, err := driver.NewMemgraphDriver(uri, user, password, "testdb")

		if err != nil {
			t.Skipf("Memgraph not available: %v", err)
			return
		}

		require.NotNil(t, d)

		// Test that Close works
		err = d.Close()
		assert.NoError(t, err)
	})
}

// TestMemgraphDriverInterface verifies that MemgraphDriver implements GraphDriver interface
func TestMemgraphDriverInterface(t *testing.T) {
	var _ driver.GraphDriver = (*driver.MemgraphDriver)(nil)
}

func TestMemgraphDriver_UpsertNode(t *testing.T) {
	d := skipIfMemgraphUnavailable(t)
	if d == nil {
		return
	}
	// defer d.Close()

	ctx := context.Background()

	// Create indices for the database
	err := d.CreateIndices(ctx)
	require.NoError(t, err)

	// Create a test node with unique ID
	now := time.Now()
	testID := "test-node-memgraph-" + time.Now().Format("20060102150405")
	testNode := &types.Node{
		Uuid:       testID,
		Name:       "Test Entity Memgraph",
		Type:       types.EntityNodeType,
		GroupID:    "test-group-memgraph",
		EntityType: "Person",
		Summary:    "A test entity for UpsertNode in Memgraph",
		CreatedAt:  now,
		UpdatedAt:  now,
	}

	// Cleanup at the end
	defer func() {
		_ = d.DeleteNode(ctx, testNode.Uuid, testNode.GroupID)
	}()

	// Upsert the node
	err = d.UpsertNode(ctx, testNode)
	require.NoError(t, err, "UpsertNode should succeed")

	// Read the node back from the database
	retrievedNode, err := d.GetNode(ctx, testNode.Uuid, testNode.GroupID)
	require.NoError(t, err, "GetNode should succeed")
	require.NotNil(t, retrievedNode, "Retrieved node should not be nil")

	// Verify the node data matches
	assert.Equal(t, testNode.Uuid, retrievedNode.Uuid, "Node ID should match")
	assert.Equal(t, testNode.Name, retrievedNode.Name, "Node name should match")
	assert.Equal(t, testNode.Type, retrievedNode.Type, "Node type should match")
	assert.Equal(t, testNode.GroupID, retrievedNode.GroupID, "Node GroupID should match")
	assert.Equal(t, testNode.EntityType, retrievedNode.EntityType, "Node EntityType should match")
	assert.Equal(t, testNode.Summary, retrievedNode.Summary, "Node summary should match")

	// Test updating the same node (upsert should update existing)
	testNode.Summary = "Updated summary for test entity in Memgraph"
	testNode.UpdatedAt = time.Now()

	err = d.UpsertNode(ctx, testNode)
	require.NoError(t, err, "Second UpsertNode (update) should succeed")

	// Read the updated node back
	updatedNode, err := d.GetNode(ctx, testNode.Uuid, testNode.GroupID)
	require.NoError(t, err, "GetNode after update should succeed")
	require.NotNil(t, updatedNode, "Updated node should not be nil")

	// Verify the update was applied
	assert.Equal(t, "Updated summary for test entity in Memgraph", updatedNode.Summary, "Node summary should be updated")
	assert.Equal(t, testNode.Uuid, updatedNode.Uuid, "Node ID should remain the same")
	assert.Equal(t, testNode.Name, updatedNode.Name, "Node name should remain the same")
}

func TestMemgraphDriver_UpsertEdge(t *testing.T) {
	d := skipIfMemgraphUnavailable(t)
	if d == nil {
		return
	}
	defer d.Close()

	ctx := context.Background()

	// Create indices for the database
	err := d.CreateIndices(ctx)
	require.NoError(t, err)

	// Create source and target nodes with unique IDs
	timestamp := time.Now().Format("20060102150405")
	sourceNode := &types.Node{
		Uuid:    "source-node-memgraph-" + timestamp,
		Name:    "Source Node Memgraph",
		Type:    types.EntityNodeType,
		GroupID: "test-group-memgraph",
	}
	targetNode := &types.Node{
		Uuid:    "target-node-memgraph-" + timestamp,
		Name:    "Target Node Memgraph",
		Type:    types.EntityNodeType,
		GroupID: "test-group-memgraph",
	}

	// Cleanup at the end
	defer func() {
		_ = d.DeleteNode(ctx, sourceNode.Uuid, sourceNode.GroupID)
		_ = d.DeleteNode(ctx, targetNode.Uuid, targetNode.GroupID)
	}()

	err = d.UpsertNode(ctx, sourceNode)
	require.NoError(t, err, "Upserting source node should succeed")
	err = d.UpsertNode(ctx, targetNode)
	require.NoError(t, err, "Upserting target node should succeed")

	// Create a test edge
	now := time.Now()
	testEdge := &types.Edge{
		BaseEdge: types.BaseEdge{
			Uuid:         "test-edge-memgraph-" + timestamp,
			GroupID:      "test-group-memgraph",
			SourceNodeID: sourceNode.Uuid,
			TargetNodeID: targetNode.Uuid,
			CreatedAt:    now,
		},
		SourceID:  sourceNode.Uuid,
		TargetID:  targetNode.Uuid,
		Type:      types.EntityEdgeType,
		UpdatedAt: now,
		Name:      "RELATES_TO",
		Fact:      "A test fact for UpsertEdge in Memgraph",
	}

	// Cleanup edge at the end
	defer func() {
		_ = d.DeleteEdge(ctx, testEdge.Uuid, testEdge.GroupID)
	}()

	// Upsert the edge
	err = d.UpsertEdge(ctx, testEdge)
	require.NoError(t, err, "UpsertEdge should succeed")

	// Read the edge back from the database
	retrievedEdge, err := d.GetEdge(ctx, testEdge.Uuid, testEdge.GroupID)
	require.NoError(t, err, "GetEdge should succeed")
	require.NotNil(t, retrievedEdge, "Retrieved edge should not be nil")

	// Verify the edge data matches
	assert.Equal(t, testEdge.Uuid, retrievedEdge.Uuid, "Edge ID should match")
	assert.Equal(t, testEdge.Name, retrievedEdge.Name, "Edge name should match")
	assert.Equal(t, testEdge.Type, retrievedEdge.Type, "Edge type should match")
	assert.Equal(t, testEdge.GroupID, retrievedEdge.GroupID, "Edge GroupID should match")
	assert.Equal(t, testEdge.SourceNodeID, retrievedEdge.SourceNodeID, "Edge SourceNodeID should match")
	assert.Equal(t, testEdge.TargetNodeID, retrievedEdge.TargetNodeID, "Edge TargetNodeID should match")
	assert.Equal(t, testEdge.Fact, retrievedEdge.Fact, "Edge fact should match")

	// Test updating the same edge (upsert should update existing)
	testEdge.Fact = "Updated fact for test edge in Memgraph"
	testEdge.UpdatedAt = time.Now()

	err = d.UpsertEdge(ctx, testEdge)
	require.NoError(t, err, "Second UpsertEdge (update) should succeed")

	// Read the updated edge back
	updatedEdge, err := d.GetEdge(ctx, testEdge.Uuid, testEdge.GroupID)
	require.NoError(t, err, "GetEdge after update should succeed")
	require.NotNil(t, updatedEdge, "Updated edge should not be nil")

	// Verify the update was applied
	assert.Equal(t, "Updated fact for test edge in Memgraph", updatedEdge.Fact, "Edge fact should be updated")
	assert.Equal(t, testEdge.Uuid, updatedEdge.Uuid, "Edge ID should remain the same")
	assert.Equal(t, testEdge.Name, updatedEdge.Name, "Edge name should remain the same")
}

func TestMemgraphDriver_NodeExists(t *testing.T) {
	d := skipIfMemgraphUnavailable(t)
	if d == nil {
		return
	}
	defer d.Close()

	ctx := context.Background()

	// Create a test node
	testNode := &types.Node{
		Uuid:    "exists-test-memgraph-" + time.Now().Format("20060102150405"),
		Name:    "Exists Test",
		Type:    types.EntityNodeType,
		GroupID: "test-group-memgraph",
	}

	defer func() {
		_ = d.DeleteNode(ctx, testNode.Uuid, testNode.GroupID)
	}()

	// Should not exist initially
	exists := d.NodeExists(ctx, testNode)
	assert.False(t, exists, "Node should not exist initially")

	// Create the node
	err := d.UpsertNode(ctx, testNode)
	require.NoError(t, err)

	// Should exist now
	exists = d.NodeExists(ctx, testNode)
	assert.True(t, exists, "Node should exist after creation")

	// Test with nil node
	exists = d.NodeExists(ctx, nil)
	assert.False(t, exists, "NodeExists with nil should return false")
}

func TestMemgraphDriver_EdgeExists(t *testing.T) {
	d := skipIfMemgraphUnavailable(t)
	if d == nil {
		return
	}
	defer d.Close()

	ctx := context.Background()

	timestamp := time.Now().Format("20060102150405")

	// Create source and target nodes
	sourceNode := &types.Node{
		Uuid:    "source-exists-memgraph-" + timestamp,
		Name:    "Source Exists Test",
		Type:    types.EntityNodeType,
		GroupID: "test-group-memgraph",
	}
	targetNode := &types.Node{
		Uuid:    "target-exists-memgraph-" + timestamp,
		Name:    "Target Exists Test",
		Type:    types.EntityNodeType,
		GroupID: "test-group-memgraph",
	}

	defer func() {
		_ = d.DeleteNode(ctx, sourceNode.Uuid, sourceNode.GroupID)
		_ = d.DeleteNode(ctx, targetNode.Uuid, targetNode.GroupID)
	}()

	err := d.UpsertNode(ctx, sourceNode)
	require.NoError(t, err)
	err = d.UpsertNode(ctx, targetNode)
	require.NoError(t, err)

	// Create test edge
	testEdge := &types.Edge{
		BaseEdge: types.BaseEdge{
			Uuid:         "edge-exists-memgraph-" + timestamp,
			GroupID:      "test-group-memgraph",
			SourceNodeID: sourceNode.Uuid,
			TargetNodeID: targetNode.Uuid,
		},
		SourceID: sourceNode.Uuid,
		TargetID: targetNode.Uuid,
		Type:     types.EntityEdgeType,
	}

	defer func() {
		_ = d.DeleteEdge(ctx, testEdge.Uuid, testEdge.GroupID)
	}()

	// Should not exist initially
	exists := d.EdgeExists(ctx, testEdge)
	assert.False(t, exists, "Edge should not exist initially")

	// Create the edge
	err = d.UpsertEdge(ctx, testEdge)
	require.NoError(t, err)

	// Should exist now
	exists = d.EdgeExists(ctx, testEdge)
	assert.True(t, exists, "Edge should exist after creation")

	// Test with nil edge
	exists = d.EdgeExists(ctx, nil)
	assert.False(t, exists, "EdgeExists with nil should return false")
}

func TestMemgraphDriver_GetNodes(t *testing.T) {
	d := skipIfMemgraphUnavailable(t)
	if d == nil {
		return
	}
	defer d.Close()

	ctx := context.Background()

	timestamp := time.Now().Format("20060102150405")
	groupID := "test-group-memgraph-batch-" + timestamp

	// Create multiple nodes
	nodes := []*types.Node{
		{
			Uuid:    "batch-node-1-" + timestamp,
			Name:    "Batch Node 1",
			Type:    types.EntityNodeType,
			GroupID: groupID,
		},
		{
			Uuid:    "batch-node-2-" + timestamp,
			Name:    "Batch Node 2",
			Type:    types.EntityNodeType,
			GroupID: groupID,
		},
		{
			Uuid:    "batch-node-3-" + timestamp,
			Name:    "Batch Node 3",
			Type:    types.EntityNodeType,
			GroupID: groupID,
		},
	}

	defer func() {
		for _, node := range nodes {
			_ = d.DeleteNode(ctx, node.Uuid, node.GroupID)
		}
	}()

	// Upsert all nodes
	for _, node := range nodes {
		err := d.UpsertNode(ctx, node)
		require.NoError(t, err)
	}

	// Get all nodes
	nodeIDs := []string{nodes[0].Uuid, nodes[1].Uuid, nodes[2].Uuid}
	retrievedNodes, err := d.GetNodes(ctx, nodeIDs, groupID)
	require.NoError(t, err)
	assert.Len(t, retrievedNodes, 3, "Should retrieve all 3 nodes")
}

func TestMemgraphDriver_Provider(t *testing.T) {
	uri, user, password := getMemgraphConnectionInfo()
	d, err := driver.NewMemgraphDriver(uri, user, password, "memgraph")
	if err != nil {
		t.Skipf("Memgraph not available: %v", err)
		return
	}
	defer d.Close()

	provider := d.Provider()
	assert.Equal(t, driver.GraphProviderMemgraph, provider, "Provider should be Memgraph")
}
