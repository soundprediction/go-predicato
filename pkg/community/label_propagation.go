package community

import (
	"context"
	"fmt"
	"sort"

	"github.com/soundprediction/predicato/pkg/types"
)

// labelPropagation implements the label propagation community detection algorithm
func (b *Builder) labelPropagation(projection map[string][]types.Neighbor) [][]string {
	if len(projection) == 0 {
		return nil
	}

	// Initialize each node to its own community
	communityMap := make(map[string]int)
	nodeIndex := 0
	for uuid := range projection {
		communityMap[uuid] = nodeIndex
		nodeIndex++
	}

	maxIterations := 100 // Prevent infinite loops
	for iteration := 0; iteration < maxIterations; iteration++ {
		noChange := true
		newCommunityMap := make(map[string]int)

		for uuid, neighbors := range projection {
			currentCommunity := communityMap[uuid]

			// Count community occurrences among neighbors, weighted by edge count
			communityCandidates := make(map[int]int)
			for _, neighbor := range neighbors {
				if neighborCommunity, exists := communityMap[neighbor.NodeUUID]; exists {
					communityCandidates[neighborCommunity] += neighbor.EdgeCount
				}
			}

			// Find the community with highest weighted count
			newCommunity := currentCommunity

			// Convert to slice for sorting
			type communityScore struct {
				community int
				count     int
			}

			var scores []communityScore
			for community, count := range communityCandidates {
				scores = append(scores, communityScore{community: community, count: count})
			}

			// Sort by count (descending), then by community ID for tie-breaking
			sort.Slice(scores, func(i, j int) bool {
				if scores[i].count != scores[j].count {
					return scores[i].count > scores[j].count
				}
				return scores[i].community > scores[j].community
			})

			if len(scores) > 0 {
				topScore := scores[0]
				if topScore.count > 1 { // Only change if there's significant support
					newCommunity = topScore.community
				} else {
					// Keep the maximum of current and candidate
					if topScore.community > currentCommunity {
						newCommunity = topScore.community
					}
				}
			}

			newCommunityMap[uuid] = newCommunity

			if newCommunity != currentCommunity {
				noChange = false
			}
		}

		if noChange {
			break
		}

		communityMap = newCommunityMap
	}

	// Group nodes by community
	communityClusterMap := make(map[int][]string)
	for uuid, community := range communityMap {
		communityClusterMap[community] = append(communityClusterMap[community], uuid)
	}

	// Convert to slice of clusters
	var clusters [][]string
	for _, cluster := range communityClusterMap {
		if len(cluster) > 1 { // Only include clusters with more than one node
			clusters = append(clusters, cluster)
		}
	}

	return clusters
}

// buildProjection builds the neighbor projection for community detection
func (b *Builder) buildProjection(ctx context.Context, nodes []*types.Node, groupID string) (map[string][]types.Neighbor, error) {
	projection := make(map[string][]types.Neighbor)

	for _, node := range nodes {
		neighbors, err := b.getNodeNeighbors(ctx, node.Uuid, groupID)
		if err != nil {
			return nil, fmt.Errorf("failed to get neighbors for node %s: %w", node.Uuid, err)
		}
		projection[node.Uuid] = neighbors
	}

	return projection, nil
}

// getNodeNeighbors gets the neighbors of a node with edge counts
func (b *Builder) getNodeNeighbors(ctx context.Context, nodeUUID, groupID string) ([]types.Neighbor, error) {
	// Check if this is a Ladybug driver to use the appropriate query
	return b.driver.GetNodeNeighbors(ctx, nodeUUID, groupID)
}

// getAllGroupIDs gets all distinct group IDs from entity nodes
func (b *Builder) getAllGroupIDs(ctx context.Context) ([]string, error) {
	return b.driver.GetAllGroupIDs(ctx)
}

// getEntityNodesByGroup gets all entity nodes for a specific group
func (b *Builder) getEntityNodesByGroup(ctx context.Context, groupID string) ([]*types.Node, error) {
	return b.driver.GetEntityNodesByGroup(ctx, groupID)
}

// getNodesByUUIDs gets nodes by their UUIDs
func (b *Builder) getNodesByUUIDs(ctx context.Context, uuids []string, groupID string) ([]*types.Node, error) {
	var nodes []*types.Node

	for _, uuid := range uuids {
		node, err := b.driver.GetNode(ctx, uuid, groupID)
		if err != nil {
			continue // Skip nodes that can't be found
		}
		nodes = append(nodes, node)
	}

	return nodes, nil
}

// DummyMemgraphNode mimics a memgraph.Node struct
type DummyMemgraphNode struct {
	ID    int64
	Props map[string]interface{}
}

func (n DummyMemgraphNode) GetID() int64                     { return n.ID }
func (n DummyMemgraphNode) GetProps() map[string]interface{} { return n.Props }

// DummyNeo4jNode mimics a neo4j.Node struct
type DummyNeo4jNode struct {
	Id    int64 // Note the lowercase 'd'
	Props map[string]interface{}
}

func (n DummyNeo4jNode) GetID() int64                     { return n.Id }
func (n DummyNeo4jNode) GetProps() map[string]interface{} { return n.Props }

// GraphNode defines the methods we expect a node object from the driver to have.
type GraphNode interface {
	GetID() int64
	GetProps() map[string]interface{}
}
