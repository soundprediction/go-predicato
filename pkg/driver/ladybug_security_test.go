//go:build cgo && system_ladybug

package driver_test

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// createSecurityTestDB creates a temporary directory for security testing
func createSecurityTestDB(t *testing.T) string {
	t.Helper()
	tempDir := t.TempDir()
	return filepath.Join(tempDir, "ladybug_security_test.db")
}

// TestLadybugDriver_CypherInjection_DeleteNode tests that DeleteNode is safe against Cypher injection
func TestLadybugDriver_CypherInjection_DeleteNode(t *testing.T) {
	dbPath := createSecurityTestDB(t)
	d, err := driver.NewLadybugDriver(dbPath, 1)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()

	// Create indices
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	// Create a legitimate test node
	testNode := &types.Node{
		Uuid:    "safe-node-123",
		Name:    "Safe Node",
		Type:    types.EntityNodeType,
		GroupID: "test-group",
	}
	err = d.UpsertNode(ctx, testNode)
	require.NoError(t, err)

	// Test various injection attempts - these should NOT cause injection
	injectionAttempts := []struct {
		name    string
		nodeID  string
		groupID string
	}{
		{
			name:    "single quote injection in nodeID",
			nodeID:  "' OR 1=1 --",
			groupID: "test-group",
		},
		{
			name:    "single quote injection in groupID",
			nodeID:  "safe-node-123",
			groupID: "' OR 1=1 --",
		},
		{
			name:    "double quote injection",
			nodeID:  `" OR 1=1 --`,
			groupID: "test-group",
		},
		{
			name:    "UNION injection attempt",
			nodeID:  "' UNION MATCH (n) DELETE n --",
			groupID: "test-group",
		},
		{
			name:    "backtick injection",
			nodeID:  "`; DROP TABLE Entity; --",
			groupID: "test-group",
		},
		{
			name:    "newline injection",
			nodeID:  "test\n DELETE n",
			groupID: "test-group",
		},
		{
			name:    "cypher comment injection",
			nodeID:  "test // DELETE n",
			groupID: "test-group",
		},
		{
			name:    "property escape attempt",
			nodeID:  "test'} DELETE n //",
			groupID: "test-group",
		},
	}

	for _, tc := range injectionAttempts {
		t.Run(tc.name, func(t *testing.T) {
			// This should not panic or cause unintended data deletion
			// The query should simply not find any matching nodes
			err := d.DeleteNode(ctx, tc.nodeID, tc.groupID)
			// Should not return error (parameterized queries handle special chars)
			assert.NoError(t, err, "DeleteNode should handle injection attempt gracefully")
		})
	}

	// Verify the safe node still exists (injection didn't delete it)
	retrievedNode, err := d.GetNode(ctx, testNode.Uuid, testNode.GroupID)
	require.NoError(t, err, "Safe node should still exist after injection attempts")
	assert.Equal(t, testNode.Uuid, retrievedNode.Uuid, "Safe node UUID should be unchanged")
}

// TestLadybugDriver_CypherInjection_DeleteEdge tests that DeleteEdge is safe against Cypher injection
func TestLadybugDriver_CypherInjection_DeleteEdge(t *testing.T) {
	dbPath := createSecurityTestDB(t)
	d, err := driver.NewLadybugDriver(dbPath, 1)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()

	// Create indices
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	// Create source and target nodes
	sourceNode := &types.Node{
		Uuid:    "source-node-sec",
		Name:    "Source Node",
		Type:    types.EntityNodeType,
		GroupID: "test-group",
	}
	targetNode := &types.Node{
		Uuid:    "target-node-sec",
		Name:    "Target Node",
		Type:    types.EntityNodeType,
		GroupID: "test-group",
	}

	err = d.UpsertNode(ctx, sourceNode)
	require.NoError(t, err)
	err = d.UpsertNode(ctx, targetNode)
	require.NoError(t, err)

	// Create a legitimate edge
	testEdge := &types.Edge{
		BaseEdge: types.BaseEdge{
			Uuid:         "safe-edge-123",
			GroupID:      "test-group",
			SourceNodeID: "source-node-sec",
			TargetNodeID: "target-node-sec",
		},
		SourceID: "source-node-sec",
		TargetID: "target-node-sec",
		Type:     types.EntityEdgeType,
		Name:     "RELATES_TO",
		Fact:     "Test edge",
	}
	err = d.UpsertEdge(ctx, testEdge)
	require.NoError(t, err)

	// Test injection attempts
	injectionAttempts := []struct {
		name    string
		edgeID  string
		groupID string
	}{
		{
			name:    "single quote injection in edgeID",
			edgeID:  "' OR 1=1 --",
			groupID: "test-group",
		},
		{
			name:    "single quote injection in groupID",
			edgeID:  "safe-edge-123",
			groupID: "' OR 1=1 DELETE rel --",
		},
		{
			name:    "MATCH injection attempt",
			edgeID:  "' MATCH (n) DELETE n --",
			groupID: "test-group",
		},
		{
			name:    "WHERE clause bypass",
			edgeID:  "x' OR rel.uuid IS NOT NULL OR '1'='1",
			groupID: "test-group",
		},
	}

	for _, tc := range injectionAttempts {
		t.Run(tc.name, func(t *testing.T) {
			err := d.DeleteEdge(ctx, tc.edgeID, tc.groupID)
			assert.NoError(t, err, "DeleteEdge should handle injection attempt gracefully")
		})
	}

	// Verify the safe edge still exists
	retrievedEdge, err := d.GetEdge(ctx, testEdge.Uuid, testEdge.GroupID)
	require.NoError(t, err, "Safe edge should still exist after injection attempts")
	assert.Equal(t, testEdge.Uuid, retrievedEdge.Uuid, "Safe edge UUID should be unchanged")
}

// TestLadybugDriver_CypherInjection_GetNeighbors tests that GetNeighbors is safe against Cypher injection
func TestLadybugDriver_CypherInjection_GetNeighbors(t *testing.T) {
	dbPath := createSecurityTestDB(t)
	d, err := driver.NewLadybugDriver(dbPath, 1)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()

	// Create indices
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	// Create a node
	testNode := &types.Node{
		Uuid:    "center-node",
		Name:    "Center Node",
		Type:    types.EntityNodeType,
		GroupID: "test-group",
	}
	err = d.UpsertNode(ctx, testNode)
	require.NoError(t, err)

	// Test injection attempts
	injectionAttempts := []struct {
		name    string
		nodeID  string
		groupID string
	}{
		{
			name:    "single quote injection in nodeID",
			nodeID:  "' OR 1=1 RETURN n --",
			groupID: "test-group",
		},
		{
			name:    "single quote injection in groupID",
			nodeID:  "center-node",
			groupID: "' OR 1=1 --",
		},
		{
			name:    "path traversal attempt",
			nodeID:  "test'-[*0..100]-(m) RETURN m --",
			groupID: "test-group",
		},
		{
			name:    "CALL procedure injection",
			nodeID:  "x' CALL db.labels() YIELD label RETURN label --",
			groupID: "test-group",
		},
	}

	for _, tc := range injectionAttempts {
		t.Run(tc.name, func(t *testing.T) {
			neighbors, err := d.GetNeighbors(ctx, tc.nodeID, tc.groupID, 1)
			// Should return empty results, not panic or return unintended data
			assert.NoError(t, err, "GetNeighbors should handle injection attempt gracefully")
			assert.Empty(t, neighbors, "GetNeighbors with injection should return no results")
		})
	}
}

// TestLadybugDriver_CypherInjection_GetNode tests that GetNode is safe against Cypher injection
func TestLadybugDriver_CypherInjection_GetNode(t *testing.T) {
	dbPath := createSecurityTestDB(t)
	d, err := driver.NewLadybugDriver(dbPath, 1)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()

	// Create indices
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	// Create a legitimate node
	testNode := &types.Node{
		Uuid:    "get-test-node",
		Name:    "Get Test Node",
		Type:    types.EntityNodeType,
		GroupID: "test-group",
	}
	err = d.UpsertNode(ctx, testNode)
	require.NoError(t, err)

	// Test injection attempts
	injectionAttempts := []struct {
		name    string
		nodeID  string
		groupID string
	}{
		{
			name:    "single quote injection",
			nodeID:  "' OR n.uuid IS NOT NULL --",
			groupID: "test-group",
		},
		{
			name:    "wildcard attempt",
			nodeID:  "*",
			groupID: "test-group",
		},
		{
			name:    "regex injection",
			nodeID:  ".*",
			groupID: "test-group",
		},
	}

	for _, tc := range injectionAttempts {
		t.Run(tc.name, func(t *testing.T) {
			node, err := d.GetNode(ctx, tc.nodeID, tc.groupID)
			// Should return not found error, not return unintended data
			assert.Error(t, err, "GetNode with injection should return error (not found)")
			assert.Nil(t, node, "GetNode with injection should not return data")
		})
	}

	// Verify legitimate node retrieval still works
	retrievedNode, err := d.GetNode(ctx, testNode.Uuid, testNode.GroupID)
	require.NoError(t, err, "Legitimate GetNode should still work")
	assert.Equal(t, testNode.Uuid, retrievedNode.Uuid)
}

// TestLadybugDriver_CypherInjection_GetEdge tests that GetEdge is safe against Cypher injection
func TestLadybugDriver_CypherInjection_GetEdge(t *testing.T) {
	dbPath := createSecurityTestDB(t)
	d, err := driver.NewLadybugDriver(dbPath, 1)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()

	// Create indices
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	// Test injection attempts - should not return any data
	injectionAttempts := []struct {
		name    string
		edgeID  string
		groupID string
	}{
		{
			name:    "single quote injection",
			edgeID:  "' OR rel.uuid IS NOT NULL --",
			groupID: "test-group",
		},
		{
			name:    "boolean bypass",
			edgeID:  "x' OR '1'='1",
			groupID: "test-group",
		},
	}

	for _, tc := range injectionAttempts {
		t.Run(tc.name, func(t *testing.T) {
			edge, err := d.GetEdge(ctx, tc.edgeID, tc.groupID)
			// Should return not found error, not return unintended data
			assert.Error(t, err, "GetEdge with injection should return error (not found)")
			assert.Nil(t, edge, "GetEdge with injection should not return data")
		})
	}
}

// TestLadybugDriver_SpecialCharacters tests that special characters in legitimate data work correctly
func TestLadybugDriver_SpecialCharacters(t *testing.T) {
	dbPath := createSecurityTestDB(t)
	d, err := driver.NewLadybugDriver(dbPath, 1)
	require.NoError(t, err)
	defer d.Close()

	ctx := context.Background()

	// Create indices
	err = d.CreateIndices(ctx)
	require.NoError(t, err)

	// Test that nodes with special characters in their names can be created and retrieved
	specialCharNodes := []struct {
		name    string
		uuid    string
		content string
	}{
		{
			name:    "single quotes in name",
			uuid:    "special-1",
			content: "O'Reilly's Book Store",
		},
		{
			name:    "double quotes in name",
			uuid:    "special-2",
			content: `He said "hello"`,
		},
		{
			name:    "backslash in name",
			uuid:    "special-3",
			content: `C:\Users\test`,
		},
		{
			name:    "unicode characters",
			uuid:    "special-4",
			content: "日本語テスト 🎉",
		},
		{
			name:    "newlines in content",
			uuid:    "special-5",
			content: "Line 1\nLine 2\nLine 3",
		},
	}

	for _, tc := range specialCharNodes {
		t.Run(tc.name, func(t *testing.T) {
			node := &types.Node{
				Uuid:    tc.uuid,
				Name:    tc.content,
				Type:    types.EntityNodeType,
				GroupID: "test-group",
				Summary: tc.content,
			}

			err := d.UpsertNode(ctx, node)
			require.NoError(t, err, "Should be able to create node with special characters")

			retrieved, err := d.GetNode(ctx, tc.uuid, "test-group")
			require.NoError(t, err, "Should be able to retrieve node with special characters")
			assert.Equal(t, tc.content, retrieved.Name, "Name with special characters should be preserved")
			assert.Equal(t, tc.content, retrieved.Summary, "Summary with special characters should be preserved")
		})
	}
}
