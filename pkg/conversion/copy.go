package conversion

import (
	"context"
	"fmt"
	"log"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/types"
)

const defaultBatchSize = 1000

// CopyGraph copies all nodes and edges from src to dest.
// It is idempotent (safe to retry) since it uses upsert semantics.
// If src implements driver.GraphExporter, streaming iteration is used.
// Otherwise, falls back to GetAllGroupIDs + GetEntityNodesByGroup.
func CopyGraph(ctx context.Context, src driver.GraphDriver, dest driver.GraphDriver) error {
	nodeCount, edgeCount, err := copyGraphWithCounts(ctx, src, dest)
	if err != nil {
		return err
	}
	log.Printf("Copy complete: %d nodes, %d edges", nodeCount, edgeCount)
	return nil
}

func copyGraphWithCounts(ctx context.Context, src driver.GraphDriver, dest driver.GraphDriver) (int, int, error) {
	// Get all group IDs from source
	groupIDs, err := src.GetAllGroupIDs(ctx)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to get group IDs from source: %w", err)
	}

	totalNodes := 0
	totalEdges := 0

	if exporter, ok := src.(driver.GraphExporter); ok {
		// Use streaming iteration if available
		for _, groupID := range groupIDs {
			n, e, err := copyGroupStreaming(ctx, exporter, dest, groupID)
			if err != nil {
				return totalNodes, totalEdges, fmt.Errorf("failed to copy group %s: %w", groupID, err)
			}
			totalNodes += n
			totalEdges += e
		}
	} else {
		// Fallback: use GetEntityNodesByGroup
		for _, groupID := range groupIDs {
			n, e, err := copyGroupFallback(ctx, src, dest, groupID)
			if err != nil {
				return totalNodes, totalEdges, fmt.Errorf("failed to copy group %s: %w", groupID, err)
			}
			totalNodes += n
			totalEdges += e
		}
	}

	return totalNodes, totalEdges, nil
}

func copyGroupStreaming(ctx context.Context, exporter driver.GraphExporter, dest driver.GraphDriver, groupID string) (int, int, error) {
	nodeCount := 0
	edgeCount := 0

	// Copy nodes in batches
	batch := make([]*types.Node, 0, defaultBatchSize)
	err := exporter.IterateNodes(ctx, groupID, func(node *types.Node) error {
		batch = append(batch, node)
		if len(batch) >= defaultBatchSize {
			if err := dest.UpsertNodes(ctx, batch); err != nil {
				return fmt.Errorf("failed to upsert node batch: %w", err)
			}
			nodeCount += len(batch)
			log.Printf("  Copied %d nodes...", nodeCount)
			batch = batch[:0]
		}
		return nil
	})
	if err != nil {
		return nodeCount, edgeCount, fmt.Errorf("node iteration failed: %w", err)
	}
	// Flush remaining nodes
	if len(batch) > 0 {
		if err := dest.UpsertNodes(ctx, batch); err != nil {
			return nodeCount, edgeCount, fmt.Errorf("failed to upsert final node batch: %w", err)
		}
		nodeCount += len(batch)
	}

	// Copy edges in batches
	edgeBatch := make([]*types.Edge, 0, defaultBatchSize)
	err = exporter.IterateEdges(ctx, groupID, func(edge *types.Edge) error {
		edgeBatch = append(edgeBatch, edge)
		if len(edgeBatch) >= defaultBatchSize {
			if err := dest.UpsertEdges(ctx, edgeBatch); err != nil {
				return fmt.Errorf("failed to upsert edge batch: %w", err)
			}
			edgeCount += len(edgeBatch)
			log.Printf("  Copied %d edges...", edgeCount)
			edgeBatch = edgeBatch[:0]
		}
		return nil
	})
	if err != nil {
		return nodeCount, edgeCount, fmt.Errorf("edge iteration failed: %w", err)
	}
	if len(edgeBatch) > 0 {
		if err := dest.UpsertEdges(ctx, edgeBatch); err != nil {
			return nodeCount, edgeCount, fmt.Errorf("failed to upsert final edge batch: %w", err)
		}
		edgeCount += len(edgeBatch)
	}

	log.Printf("Group %s: %d nodes, %d edges", groupID, nodeCount, edgeCount)
	return nodeCount, edgeCount, nil
}

func copyGroupFallback(ctx context.Context, src driver.GraphDriver, dest driver.GraphDriver, groupID string) (int, int, error) {
	// Get all entity nodes for this group
	nodes, err := src.GetEntityNodesByGroup(ctx, groupID)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to get nodes for group %s: %w", groupID, err)
	}

	// Upsert nodes in batches
	nodeCount := 0
	for i := 0; i < len(nodes); i += defaultBatchSize {
		end := i + defaultBatchSize
		if end > len(nodes) {
			end = len(nodes)
		}
		if err := dest.UpsertNodes(ctx, nodes[i:end]); err != nil {
			return nodeCount, 0, fmt.Errorf("failed to upsert nodes: %w", err)
		}
		nodeCount += end - i
		log.Printf("  Copied %d/%d nodes...", nodeCount, len(nodes))
	}

	// Get edges between all pairs — use GetBetweenNodes for each source node
	// This is the fallback path so we collect edges by iterating neighbors
	edgeCount := 0
	seen := make(map[string]bool)
	for _, node := range nodes {
		neighbors, err := src.GetNodeNeighbors(ctx, node.Uuid, groupID)
		if err != nil {
			continue // best effort
		}
		for _, neighbor := range neighbors {
			edges, err := src.GetBetweenNodes(ctx, node.Uuid, neighbor.NodeUUID)
			if err != nil {
				continue
			}
			batch := make([]*types.Edge, 0)
			for _, edge := range edges {
				if !seen[edge.Uuid] {
					seen[edge.Uuid] = true
					batch = append(batch, edge)
				}
			}
			if len(batch) > 0 {
				if err := dest.UpsertEdges(ctx, batch); err != nil {
					return nodeCount, edgeCount, fmt.Errorf("failed to upsert edges: %w", err)
				}
				edgeCount += len(batch)
			}
		}
	}

	log.Printf("Group %s: %d nodes, %d edges", groupID, nodeCount, edgeCount)
	return nodeCount, edgeCount, nil
}
