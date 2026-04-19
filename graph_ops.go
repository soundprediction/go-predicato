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
// This is an exact translation of the Python Predicato.remove_episode() method.
func (c *Client) RemoveEpisode(ctx context.Context, episodeUUID string) error {
	// Find the episode to be deleted
	// Equivalent to: episode = await EpisodicNode.get_by_uuid(self.driver, episode_uuid)
	episode, err := types.GetEpisodicNodeByUUID(ctx, c.driver, episodeUUID)
	if err != nil {
		return fmt.Errorf("failed to get episode: %w", err)
	}

	// Find edges mentioned by the episode
	// Equivalent to: edges = await EntityEdge.get_by_uuids(self.driver, episode.entity_edges)
	wrapper := &driverWrapper{c.driver}
	edges, err := types.GetEntityEdgesByUUIDs(ctx, wrapper, episode.EntityEdges)
	if err != nil {
		return fmt.Errorf("failed to get entity edges: %w", err)
	}

	// We should only delete edges created by the episode
	// Equivalent to: if edge.episodes and edge.episodes[0] == episode.uuid:
	var edgesToDelete []*types.Edge
	for _, edge := range edges {
		if len(edge.Episodes) > 0 && edge.Episodes[0] == episode.Uuid {
			edgesToDelete = append(edgesToDelete, edge)
		}
	}

	// Find nodes mentioned by the episode
	// Equivalent to: nodes = await get_mentioned_nodes(self.driver, [episode])
	mentionedNodes, err := types.GetMentionedNodes(ctx, c.driver, []*types.Node{episode})
	if err != nil {
		return fmt.Errorf("failed to get mentioned nodes: %w", err)
	}

	// We should delete all nodes that are only mentioned in the deleted episode
	var nodesToDelete []*types.Node
	for _, node := range mentionedNodes {
		// Equivalent to: query: LiteralString = 'MATCH (e:Episodic)-[:MENTIONS]->(n:Entity {uuid: $uuid}) RETURN count(*) AS episode_count'
		query := `MATCH (e:Episodic)-[:MENTIONS]->(n:Entity {uuid: $uuid}) RETURN count(*) AS episode_count`
		records, _, _, err := c.driver.ExecuteQuery(ctx, query, map[string]interface{}{
			"uuid": node.Uuid,
		})
		if err != nil {
			c.logger.Warn("failed to check episode count for node, skipping deletion",
				"node_uuid", node.Uuid,
				"error", err)
			continue // Skip on error, don't delete
		}

		// Check if only one episode mentions this node
		if recordList, ok := records.([]map[string]interface{}); ok {
			for _, record := range recordList {
				if count, ok := record["episode_count"].(int64); ok && count == 1 {
					nodesToDelete = append(nodesToDelete, node)
				}
			}
		}
	}

	// Delete edges first
	// Equivalent to: await Edge.delete_by_uuids(self.driver, [edge.uuid for edge in edges_to_delete])
	if len(edgesToDelete) > 0 {
		edgeUUIDs := make([]string, len(edgesToDelete))
		for i, edge := range edgesToDelete {
			edgeUUIDs[i] = edge.Uuid
		}
		if err := types.DeleteEdgesByUUIDs(ctx, wrapper, edgeUUIDs); err != nil {
			return fmt.Errorf("failed to delete edges: %w", err)
		}
	}

	// Delete nodes
	// Equivalent to: await Node.delete_by_uuids(self.driver, [node.uuid for node in nodes_to_delete])
	if len(nodesToDelete) > 0 {
		nodeUUIDs := make([]string, len(nodesToDelete))
		for i, node := range nodesToDelete {
			nodeUUIDs[i] = node.Uuid
		}
		if err := types.DeleteNodesByUUIDs(ctx, c.driver, nodeUUIDs); err != nil {
			return fmt.Errorf("failed to delete nodes: %w", err)
		}
	}

	// Finally, delete the episode itself
	// Equivalent to: await episode.delete(self.driver)
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
