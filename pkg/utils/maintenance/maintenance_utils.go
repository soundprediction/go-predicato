package maintenance

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/types"
)

// MaintenanceUtils provides general utility functions for maintenance operations
type MaintenanceUtils struct {
	driver driver.GraphDriver
}

// NewMaintenanceUtils creates a new MaintenanceUtils instance
func NewMaintenanceUtils(driver driver.GraphDriver) *MaintenanceUtils {
	return &MaintenanceUtils{
		driver: driver,
	}
}

// GetEntitiesAndEdges retrieves all entities and edges for a given group ID
func (mu *MaintenanceUtils) GetEntitiesAndEdges(ctx context.Context, groupID string) ([]*types.Node, []*types.Edge, error) {
	log.Printf("Retrieving all entities and edges for group: %s", groupID)

	// Get all entity nodes
	entityOptions := &driver.SearchOptions{
		Limit:     10000, // Large limit to get all entities
		NodeTypes: []types.NodeType{types.EntityNodeType},
	}

	entities, err := mu.driver.SearchNodes(ctx, "", groupID, entityOptions)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to retrieve entities: %w", err)
	}

	// Get all edges
	edgeOptions := &driver.SearchOptions{
		Limit:     10000, // Large limit to get all edges
		EdgeTypes: []types.EdgeType{types.EntityEdgeType, types.EpisodicEdgeType, types.CommunityEdgeType},
	}

	edges, err := mu.driver.SearchEdges(ctx, "", groupID, edgeOptions)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to retrieve edges: %w", err)
	}

	log.Printf("Retrieved %d entities and %d edges", len(entities), len(edges))
	return entities, edges, nil
}

// GetEntitiesByType retrieves all entities of a specific type for a given group ID
func (mu *MaintenanceUtils) GetEntitiesByType(ctx context.Context, groupID string, nodeType types.NodeType) ([]*types.Node, error) {
	options := &driver.SearchOptions{
		Limit:     10000,
		NodeTypes: []types.NodeType{nodeType},
	}

	nodes, err := mu.driver.SearchNodes(ctx, "", groupID, options)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve nodes of type %s: %w", nodeType, err)
	}

	log.Printf("Retrieved %d nodes of type %s", len(nodes), nodeType)
	return nodes, nil
}

// GetEdgesByType retrieves all edges of a specific type for a given group ID
func (mu *MaintenanceUtils) GetEdgesByType(ctx context.Context, groupID string, edgeType types.EdgeType) ([]*types.Edge, error) {
	options := &driver.SearchOptions{
		Limit:     10000,
		EdgeTypes: []types.EdgeType{edgeType},
	}

	edges, err := mu.driver.SearchEdges(ctx, "", groupID, options)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve edges of type %s: %w", edgeType, err)
	}

	log.Printf("Retrieved %d edges of type %s", len(edges), edgeType)
	return edges, nil
}

// GetNodesConnectedToNode retrieves all nodes connected to a specific node
func (mu *MaintenanceUtils) GetNodesConnectedToNode(ctx context.Context, nodeID, groupID string, maxDistance int) ([]*types.Node, error) {
	if maxDistance <= 0 {
		maxDistance = 1
	}

	neighbors, err := mu.driver.GetNeighbors(ctx, nodeID, groupID, maxDistance)
	if err != nil {
		return nil, fmt.Errorf("failed to get neighbors for node %s: %w", nodeID, err)
	}

	log.Printf("Retrieved %d neighbors for node %s within distance %d", len(neighbors), nodeID, maxDistance)
	return neighbors, nil
}

// GetEdgesForNode retrieves all edges connected to a specific node
func (mu *MaintenanceUtils) GetEdgesForNode(ctx context.Context, nodeID, groupID string) ([]*types.Edge, error) {
	// Search for edges where this node is either source or target
	// This is a simplified approach - in a production system you might want more efficient queries
	allEdges, err := mu.driver.SearchEdges(ctx, "", groupID, &driver.SearchOptions{Limit: 10000})
	if err != nil {
		return nil, fmt.Errorf("failed to search edges: %w", err)
	}

	var connectedEdges []*types.Edge
	for _, edge := range allEdges {
		if edge.SourceID == nodeID || edge.TargetID == nodeID {
			connectedEdges = append(connectedEdges, edge)
		}
	}

	log.Printf("Retrieved %d edges connected to node %s", len(connectedEdges), nodeID)
	return connectedEdges, nil
}

// GetGraphStatistics returns basic statistics about the graph
func (mu *MaintenanceUtils) GetGraphStatistics(ctx context.Context, groupID string) (*GraphStatistics, error) {
	stats, err := mu.driver.GetStats(ctx, groupID)
	if err != nil {
		return nil, fmt.Errorf("failed to get graph stats: %w", err)
	}

	return &GraphStatistics{
		NodeCount:      stats.NodeCount,
		EdgeCount:      stats.EdgeCount,
		NodesByType:    stats.NodesByType,
		EdgesByType:    stats.EdgesByType,
		CommunityCount: stats.CommunityCount,
		LastUpdated:    stats.LastUpdated,
	}, nil
}

// GraphStatistics holds statistics about the graph
type GraphStatistics struct {
	LastUpdated    time.Time        `json:"last_updated"`
	NodesByType    map[string]int64 `json:"nodes_by_type"`
	EdgesByType    map[string]int64 `json:"edges_by_type"`
	NodeCount      int64            `json:"node_count"`
	EdgeCount      int64            `json:"edge_count"`
	CommunityCount int64            `json:"community_count"`
}

// CleanupOrphanedEdges removes edges that reference non-existent nodes
func (mu *MaintenanceUtils) CleanupOrphanedEdges(ctx context.Context, groupID string) (int, error) {
	log.Printf("Cleaning up orphaned edges for group: %s", groupID)

	// Get all edges
	edges, err := mu.driver.SearchEdges(ctx, "", groupID, &driver.SearchOptions{Limit: 10000})
	if err != nil {
		return 0, fmt.Errorf("failed to get edges: %w", err)
	}

	// Get all nodes to create a lookup map
	nodes, err := mu.driver.SearchNodes(ctx, "", groupID, &driver.SearchOptions{Limit: 10000})
	if err != nil {
		return 0, fmt.Errorf("failed to get nodes: %w", err)
	}

	nodeExists := make(map[string]bool)
	for _, node := range nodes {
		nodeExists[node.Uuid] = true
	}

	// Find and delete orphaned edges
	orphanedCount := 0
	for _, edge := range edges {
		if !nodeExists[edge.SourceID] || !nodeExists[edge.TargetID] {
			if err := mu.driver.DeleteEdge(ctx, edge.Uuid, groupID); err != nil {
				log.Printf("Warning: failed to delete orphaned edge %s: %v", edge.Uuid, err)
			} else {
				orphanedCount++
			}
		}
	}

	log.Printf("Cleaned up %d orphaned edges", orphanedCount)
	return orphanedCount, nil
}

// ValidateGraphIntegrity performs basic integrity checks on the graph
func (mu *MaintenanceUtils) ValidateGraphIntegrity(ctx context.Context, groupID string) ([]string, error) {
	log.Printf("Validating graph integrity for group: %s", groupID)

	var issues []string

	// Get all nodes and edges
	nodes, edges, err := mu.GetEntitiesAndEdges(ctx, groupID)
	if err != nil {
		return nil, fmt.Errorf("failed to get nodes and edges: %w", err)
	}

	// Create node existence map
	nodeExists := make(map[string]bool)
	for _, node := range nodes {
		nodeExists[node.Uuid] = true
	}

	// Check for edges referencing non-existent nodes
	for _, edge := range edges {
		if !nodeExists[edge.SourceID] {
			issues = append(issues, fmt.Sprintf("Edge %s references non-existent source node %s", edge.Uuid, edge.SourceID))
		}
		if !nodeExists[edge.TargetID] {
			issues = append(issues, fmt.Sprintf("Edge %s references non-existent target node %s", edge.Uuid, edge.TargetID))
		}
	}

	// Check for nodes without embeddings
	for _, node := range nodes {
		if len(node.Embedding) == 0 {
			issues = append(issues, fmt.Sprintf("Node %s (%s) has no embedding", node.Uuid, node.Name))
		}
	}

	// Check for edges without embeddings
	for _, edge := range edges {
		if len(edge.Embedding) == 0 {
			issues = append(issues, fmt.Sprintf("Edge %s has no embedding", edge.Uuid))
		}
	}

	log.Printf("Found %d integrity issues", len(issues))
	return issues, nil
}
