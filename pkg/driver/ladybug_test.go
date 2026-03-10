//go:build cgo && system_ladybug

package driver_test

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// createTempLadybugDB creates a temporary directory for ladybug database testing
func createTempLadybugDB(t *testing.T) string {
	t.Helper()
	tempDir := t.TempDir()
	return filepath.Join(tempDir, "ladybug_test.db")
}

func TestNewLadybugDriver(t *testing.T) {
	t.Run("default path", func(t *testing.T) {
		d, err := driver.NewLadybugDriver("", 1)
		require.NoError(t, err)
		assert.NotNil(t, d)

		// Test that Close works
		err = d.Close()
		assert.NoError(t, err)
	})

	t.Run("custom path", func(t *testing.T) {
		dbPath := createTempLadybugDB(t)
		d, err := driver.NewLadybugDriver(dbPath, 1)
		require.NoError(t, err)
		assert.NotNil(t, d)

		// Test that Close works
		err = d.Close()
		assert.NoError(t, err)
	})
}

// TestLadybugDriverStubImplementation is now deprecated since LadybugDriver is fully implemented
// Kept as a placeholder to maintain test compatibility, but skipped
func TestLadybugDriverStubImplementation(t *testing.T) {
	t.Skip("LadybugDriver is now fully implemented - this stub test is no longer needed")
}

// TestLadybugDriverInterface verifies that LadybugDriver implements GraphDriver interface
func TestLadybugDriverInterface(t *testing.T) {
	var _ driver.GraphDriver = (*driver.LadybugDriver)(nil)
}

// Example test showing expected usage once the full implementation is available
func TestLadybugDriverUsageExample(t *testing.T) {
	// t.Skip("Skip until ladybug library is available")

	// This test demonstrates expected usage patterns but is skipped
	// until the actual ladybug library dependency is available
	d, err := driver.NewLadybugDriver("./test_ladybug_db", 1)
	require.NoError(t, err)
	defer d.Close()

	// In a real scenario, you would:
	// 1. Create nodes
	// node := &types.Node{
	//     ID: "test-node",
	//     Name: "Test Node",
	//     Type: types.NodeTypeEntity,
	//     GroupID: "test-group",
	// }
	// err = d.UpsertNode(ctx, node)
	// require.NoError(t, err)
	//
	// 2. Create edges
	// edge := &types.Edge{
	//     ID: "test-edge",
	//     Type: types.EdgeTypeEntity,
	//     GroupID: "test-group",
	//     SourceID: "source-node",
	//     TargetID: "target-node",
	// }
	// err = d.UpsertEdge(ctx, edge)
	// require.NoError(t, err)
	//
	// 3. Query neighbors
	// neighbors, err := d.GetNeighbors(ctx, "test-node", "test-group", 2)
	// require.NoError(t, err)
	// assert.NotEmpty(t, neighbors)
}

func TestLadybugDriver_UpsertNode(t *testing.T) {
	dbPath := createTempLadybugDB(t)
	d, err := driver.NewLadybugDriver(dbPath, 1)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()

	// Create indices for the database
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	// Create a test node
	now := time.Now()
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

	// Upsert the node
	err = d.UpsertNode(ctx, testNode)
	require.NoError(t, err, "UpsertNode should succeed")

	// Read the node back from the database
	retrievedNode, err := d.GetNode(ctx, testNode.Uuid, testNode.GroupID)
	require.NoError(t, err, "GetNode should succeed")
	require.NotNil(t, retrievedNode, "Retrieved node should not be nil")

	// Verify the node data matches
	assert.Equal(t, testNode.Uuid, retrievedNode.Uuid, "Node UUID should match")
	assert.Equal(t, testNode.Name, retrievedNode.Name, "Node name should match")
	assert.Equal(t, testNode.Type, retrievedNode.Type, "Node type should match")
	assert.Equal(t, testNode.GroupID, retrievedNode.GroupID, "Node GroupID should match")
	assert.Equal(t, testNode.EntityType, retrievedNode.EntityType, "Node EntityType should match")
	assert.Equal(t, testNode.Summary, retrievedNode.Summary, "Node summary should match")

	// Test updating the same node (upsert should update existing)
	testNode.Summary = "Updated summary for test entity"
	testNode.UpdatedAt = time.Now()

	err = d.UpsertNode(ctx, testNode)
	require.NoError(t, err, "Second UpsertNode (update) should succeed")

	// Read the updated node back
	updatedNode, err := d.GetNode(ctx, testNode.Uuid, testNode.GroupID)
	require.NoError(t, err, "GetNode after update should succeed")
	require.NotNil(t, updatedNode, "Updated node should not be nil")

	// Verify the update was applied
	assert.Equal(t, "Updated summary for test entity", updatedNode.Summary, "Node summary should be updated")
	assert.Equal(t, testNode.Uuid, updatedNode.Uuid, "Node ID should remain the same")
	assert.Equal(t, testNode.Name, updatedNode.Name, "Node name should remain the same")
}

func TestLadybugDriver_UpsertEdge(t *testing.T) {
	dbPath := createTempLadybugDB(t)
	d, err := driver.NewLadybugDriver(dbPath, 1)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()

	// Create indices for the database
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	// Create source and target nodes
	sourceNode := &types.Node{
		Uuid:    "source-node",
		Name:    "Source Node",
		Type:    types.EntityNodeType,
		GroupID: "test-group",
	}
	targetNode := &types.Node{
		Uuid:    "target-node",
		Name:    "Target Node",
		Type:    types.EntityNodeType,
		GroupID: "test-group",
	}

	err = d.UpsertNode(ctx, sourceNode)
	require.NoError(t, err, "Upserting source node should succeed")
	err = d.UpsertNode(ctx, targetNode)
	require.NoError(t, err, "Upserting target node should succeed")

	// Create a test edge
	now := time.Now()
	testEdge := &types.Edge{
		BaseEdge: types.BaseEdge{
			Uuid:         "test-edge-123",
			GroupID:      "test-group",
			SourceNodeID: "source-node",
			TargetNodeID: "target-node",
			CreatedAt:    now,
		},
		SourceID:  "source-node",
		TargetID:  "target-node",
		Type:      types.EntityEdgeType,
		UpdatedAt: now,
		Name:      "RELATES_TO",
		Fact:      "A test fact for UpsertEdge",
	}

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
	testEdge.Fact = "Updated fact for test edge"
	testEdge.UpdatedAt = time.Now()

	err = d.UpsertEdge(ctx, testEdge)
	require.NoError(t, err, "Second UpsertEdge (update) should succeed")

	// Read the updated edge back
	updatedEdge, err := d.GetEdge(ctx, testEdge.Uuid, testEdge.GroupID)
	require.NoError(t, err, "GetEdge after update should succeed")
	require.NotNil(t, updatedEdge, "Updated edge should not be nil")

	// Verify the update was applied
	assert.Equal(t, "Updated fact for test edge", updatedEdge.Fact, "Edge fact should be updated")
	assert.Equal(t, testEdge.Uuid, updatedEdge.Uuid, "Edge ID should remain the same")
	assert.Equal(t, testEdge.Name, updatedEdge.Name, "Edge name should remain the same")
}

func TestLadybugDriver_UpsertEpisodicNode(t *testing.T) {
	dbPath := createTempLadybugDB(t)
	d, err := driver.NewLadybugDriver(dbPath, 1)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()

	// Create indices for the database
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	// Create a test episodic node
	now := time.Now()
	testNode := &types.Node{
		Uuid:        "test-episode-123",
		Name:        "Test Episode",
		Type:        types.EpisodicNodeType,
		GroupID:     "test-group",
		EpisodeType: types.ConversationEpisodeType,
		Content:     "This is the content of the test episode",
		CreatedAt:   now,
		UpdatedAt:   now,
		ValidFrom:   now,
	}

	// Upsert the node (CREATE)
	err = d.UpsertNode(ctx, testNode)
	require.NoError(t, err, "UpsertNode should succeed on create")

	// Read the node back from the database to verify source field was set
	retrievedNode, err := d.GetNode(ctx, testNode.Uuid, testNode.GroupID)
	require.NoError(t, err, "GetNode should succeed")
	require.NotNil(t, retrievedNode, "Retrieved node should not be nil")

	// Verify the node data matches
	assert.Equal(t, testNode.Uuid, retrievedNode.Uuid, "Node UUID should match")
	assert.Equal(t, testNode.Name, retrievedNode.Name, "Node name should match")
	assert.Equal(t, testNode.Type, retrievedNode.Type, "Node type should match")
	assert.Equal(t, testNode.GroupID, retrievedNode.GroupID, "Node GroupID should match")
	assert.Equal(t, testNode.Content, retrievedNode.Content, "Node content should match")
	assert.Equal(t, testNode.EpisodeType, retrievedNode.EpisodeType, "Node EpisodeType should match")

	// Update the node's content and name (to test UPDATE path)
	testNode.Name = "Updated Test Episode"
	testNode.Content = "This is the updated content"
	testNode.UpdatedAt = time.Now()

	// Upsert the node again (UPDATE)
	err = d.UpsertNode(ctx, testNode)
	require.NoError(t, err, "Second UpsertNode (update) should succeed")

	// Read the updated node back
	updatedNode, err := d.GetNode(ctx, testNode.Uuid, testNode.GroupID)
	require.NoError(t, err, "GetNode after update should succeed")
	require.NotNil(t, updatedNode, "Updated node should not be nil")

	// Verify the update was applied
	assert.Equal(t, "Updated Test Episode", updatedNode.Name, "Node name should be updated")
	assert.Equal(t, "This is the updated content", updatedNode.Content, "Node content should be updated")
	assert.Equal(t, testNode.Uuid, updatedNode.Uuid, "Node UUID should remain the same")
	assert.Equal(t, testNode.GroupID, updatedNode.GroupID, "Node GroupID should remain the same")
	assert.Equal(t, testNode.EpisodeType, updatedNode.EpisodeType, "Node EpisodeType should remain the same")

	// Verify that source is preserved (it's derived from EpisodeType)
	// The source field should be set to the string value of EpisodeType
	assert.Equal(t, testNode.EpisodeType, updatedNode.EpisodeType, "EpisodeType should be preserved after update")
}

func TestLadybugDriver_UpsertEpisodicEdge(t *testing.T) {
	dbPath := createTempLadybugDB(t)
	d, err := driver.NewLadybugDriver(dbPath, 1)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()

	// Create indices for the database
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	// Create an episodic node
	now := time.Now()
	episodeNode := &types.Node{
		Uuid:        "episode-test-123",
		Name:        "Test Episode",
		Type:        types.EpisodicNodeType,
		GroupID:     "test-group",
		EpisodeType: types.ConversationEpisodeType,
		Content:     "Episode content",
		CreatedAt:   now,
		UpdatedAt:   now,
		ValidFrom:   now,
	}

	err = d.UpsertNode(ctx, episodeNode)
	require.NoError(t, err, "Creating episode node should succeed")

	// Create an entity node
	entityNode := &types.Node{
		Uuid:       "entity-test-123",
		Name:       "Test Entity",
		Type:       types.EntityNodeType,
		GroupID:    "test-group",
		EntityType: "Person",
		Summary:    "An entity for testing",
		CreatedAt:  now,
		UpdatedAt:  now,
	}

	err = d.UpsertNode(ctx, entityNode)
	require.NoError(t, err, "Creating entity node should succeed")

	// Create episodic edge (MENTIONS relationship)
	err = d.UpsertEpisodicEdge(ctx, episodeNode.Uuid, entityNode.Uuid, "test-group")
	require.NoError(t, err, "UpsertEpisodicEdge should succeed")

	// Verify the edge was created by querying it
	query := `
		MATCH (e:Episodic {uuid: $episode_uuid})-[m:MENTIONS]->(n:Entity {uuid: $entity_uuid})
		RETURN m.group_id AS group_id, m.created_at AS created_at
	`
	result, _, _, err := d.ExecuteQuery(ctx, query, map[string]any{
		"episode_uuid": episodeNode.Uuid,
		"entity_uuid":  entityNode.Uuid,
	})
	require.NoError(t, err, "Querying MENTIONS edge should succeed")

	resultList, ok := result.([]map[string]any)
	require.True(t, ok, "Result should be a list of maps")
	require.Len(t, resultList, 1, "Should find exactly one MENTIONS edge")

	assert.Equal(t, "test-group", resultList[0]["group_id"], "Group ID should match")

	// Test idempotency - upserting again should not fail
	err = d.UpsertEpisodicEdge(ctx, episodeNode.Uuid, entityNode.Uuid, "test-group")
	require.NoError(t, err, "Second UpsertEpisodicEdge should succeed (idempotent)")
}

func TestLadybugDriver_UpsertCommunityEdge(t *testing.T) {
	dbPath := createTempLadybugDB(t)
	d, err := driver.NewLadybugDriver(dbPath, 1)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()

	// Create indices for the database
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	// Create a community node
	now := time.Now()
	communityNode := &types.Node{
		Uuid:      "community-test-123",
		Name:      "Test Community",
		Type:      types.CommunityNodeType,
		GroupID:   "test-group",
		Summary:   "A test community",
		CreatedAt: now,
		UpdatedAt: now,
	}

	err = d.UpsertNode(ctx, communityNode)
	require.NoError(t, err, "Creating community node should succeed")

	// Create an entity node
	entityNode := &types.Node{
		Uuid:       "entity-test-456",
		Name:       "Test Entity",
		Type:       types.EntityNodeType,
		GroupID:    "test-group",
		EntityType: "Person",
		Summary:    "An entity for testing",
		CreatedAt:  now,
		UpdatedAt:  now,
	}

	err = d.UpsertNode(ctx, entityNode)
	require.NoError(t, err, "Creating entity node should succeed")

	// Create community edge (HAS_MEMBER relationship)
	edgeUUID := "community-edge-123"
	err = d.UpsertCommunityEdge(ctx, communityNode.Uuid, entityNode.Uuid, edgeUUID, "test-group")
	require.NoError(t, err, "UpsertCommunityEdge should succeed")

	// Verify the edge was created by querying it
	query := `
		MATCH (c:Community {uuid: $community_uuid})-[h:HAS_MEMBER {uuid: $edge_uuid}]->(n:Entity {uuid: $entity_uuid})
		RETURN h.group_id AS group_id, h.created_at AS created_at, h.uuid AS uuid
	`
	result, _, _, err := d.ExecuteQuery(ctx, query, map[string]any{
		"community_uuid": communityNode.Uuid,
		"entity_uuid":    entityNode.Uuid,
		"edge_uuid":      edgeUUID,
	})
	require.NoError(t, err, "Querying HAS_MEMBER edge should succeed")

	resultList, ok := result.([]map[string]any)
	require.True(t, ok, "Result should be a list of maps")
	require.Len(t, resultList, 1, "Should find exactly one HAS_MEMBER edge")

	assert.Equal(t, "test-group", resultList[0]["group_id"], "Group ID should match")
	assert.Equal(t, edgeUUID, resultList[0]["uuid"], "Edge UUID should match")

	// Test idempotency - upserting again should not fail
	err = d.UpsertCommunityEdge(ctx, communityNode.Uuid, entityNode.Uuid, edgeUUID, "test-group")
	require.NoError(t, err, "Second UpsertCommunityEdge should succeed (idempotent)")
}
