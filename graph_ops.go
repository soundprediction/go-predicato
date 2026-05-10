package predicato

import (
	"context"
	"fmt"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/types"
)

// ClearGraph removes all nodes and edges from the knowledge graph for a specific group.
func (c *Client) ClearGraph(ctx context.Context, groupID string) error {
	if groupID == "" {
		groupID = c.config.GroupID
	}

	// First, get all nodes for this group
	allNodes, err := c.getAllNodesForGroup(ctx, groupID)
	if err != nil {
		return fmt.Errorf("failed to get nodes for clearing: %w", err)
	}

	// Delete all nodes (this will also delete associated edges in most graph databases)
	for _, node := range allNodes {
		if err := c.driver.DeleteNode(ctx, node.Uuid, groupID); err != nil {
			return fmt.Errorf("failed to delete node %s: %w", node.Uuid, err)
		}
	}

	return nil
}

// getAllNodesForGroup retrieves all nodes for a specific group
func (c *Client) getAllNodesForGroup(ctx context.Context, groupID string) ([]*types.Node, error) {
	// Search for all nodes with a high limit and no type filter
	searchOptions := &driver.SearchOptions{
		Limit: 100000, // Large limit to get all nodes
	}

	return c.driver.SearchNodes(ctx, "", groupID, searchOptions)
}

// CreateIndices creates database indices and constraints for optimal performance.
func (c *Client) CreateIndices(ctx context.Context) error {
	return c.driver.CreateIndices(ctx)
}

// RemoveEpisode removes an episode and its associated nodes and edges from the knowledge graph.
func (c *Client) RemoveEpisode(ctx context.Context, episodeUUID string) error {
	episode, err := types.GetEpisodicNodeByUUID(ctx, c.driver, episodeUUID)
	if err != nil {
		return fmt.Errorf("failed to get episode: %w", err)
	}

	wrapper := &driverWrapper{c.driver}
	edges, err := types.GetEntityEdgesByUUIDs(ctx, wrapper, episode.EntityEdges)
	if err != nil {
		return fmt.Errorf("failed to get entity edges: %w", err)
	}

	// Only delete edges created by this episode (the episode is first in its episodes list).
	var edgesToDelete []*types.Edge
	for _, edge := range edges {
		if len(edge.Episodes) > 0 && edge.Episodes[0] == episode.Uuid {
			edgesToDelete = append(edgesToDelete, edge)
		}
	}

	mentionedNodes, err := types.GetMentionedNodes(ctx, c.driver, []*types.Node{episode})
	if err != nil {
		return fmt.Errorf("failed to get mentioned nodes: %w", err)
	}

	// Find which mentioned nodes are referenced only by this episode in a single query.
	var nodesToDelete []*types.Node
	if len(mentionedNodes) > 0 {
		uuids := make([]string, len(mentionedNodes))
		for i, n := range mentionedNodes {
			uuids[i] = n.Uuid
		}
		query := `MATCH (e:Episodic)-[:MENTIONS]->(n:Entity) WHERE n.uuid IN $uuids RETURN n.uuid AS uuid, count(*) AS episode_count`
		records, _, _, err := c.driver.ExecuteQuery(ctx, query, map[string]interface{}{"uuids": uuids})
		if err != nil {
			c.logger.Warn("failed to check episode counts; skipping orphan node deletion", "error", err)
		} else if recordList, ok := records.([]map[string]interface{}); ok {
			counts := make(map[string]int64, len(recordList))
			for _, record := range recordList {
				uuid, _ := record["uuid"].(string)
				if count, ok := record["episode_count"].(int64); ok {
					counts[uuid] = count
				}
			}
			for _, node := range mentionedNodes {
				if counts[node.Uuid] == 1 {
					nodesToDelete = append(nodesToDelete, node)
				}
			}
		}
	}

	if len(edgesToDelete) > 0 {
		edgeUUIDs := make([]string, len(edgesToDelete))
		for i, edge := range edgesToDelete {
			edgeUUIDs[i] = edge.Uuid
		}
		if err := types.DeleteEdgesByUUIDs(ctx, wrapper, edgeUUIDs); err != nil {
			return fmt.Errorf("failed to delete edges: %w", err)
		}
	}

	if len(nodesToDelete) > 0 {
		nodeUUIDs := make([]string, len(nodesToDelete))
		for i, node := range nodesToDelete {
			nodeUUIDs[i] = node.Uuid
		}
		if err := types.DeleteNodesByUUIDs(ctx, c.driver, nodeUUIDs); err != nil {
			return fmt.Errorf("failed to delete nodes: %w", err)
		}
	}

	if err := types.DeleteNode(ctx, c.driver, episode); err != nil {
		return fmt.Errorf("failed to delete episode: %w", err)
	}

	return nil
}

// Close closes the client and all its connections.
func (c *Client) Close(ctx context.Context) error {
	if c == nil || c.driver == nil {
		return nil
	}
	return c.driver.Close()
}

// ExecuteQuery executes a raw Cypher query against the graph database.
// This exposes the underlying driver's query execution capability.
func (c *Client) ExecuteQuery(ctx context.Context, query string, params map[string]interface{}) (interface{}, interface{}, interface{}, error) {
	return c.driver.ExecuteQuery(ctx, query, params)
}
