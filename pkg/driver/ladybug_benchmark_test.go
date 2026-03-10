//go:build cgo && system_ladybug

package driver_test

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"
	"time"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/types"
)

// BenchmarkGetNodesBatch compares batch query vs individual queries for GetNodes.
// This benchmark demonstrates the performance improvement of the batch query optimization
// implemented in Phase 3.1 (fixing N+1 queries).
//
// The batch approach uses: MATCH (n) WHERE n.uuid IN $uuids RETURN n
// The individual approach would use: MATCH (n {uuid: $uuid}) RETURN n (per node)
func BenchmarkGetNodesBatch(b *testing.B) {
	tempDir := b.TempDir()
	dbPath := filepath.Join(tempDir, "bench_ladybug.db")
	d, err := driver.NewLadybugDriver(dbPath, 1)
	if err != nil {
		b.Fatalf("Failed to create driver: %v", err)
	}
	defer d.Close()

	ctx := context.Background()

	// Create indices
	if err := d.CreateIndices(ctx); err != nil {
		b.Fatalf("Failed to create indices: %v", err)
	}

	// Setup: create test nodes
	const numNodes = 100
	uuids := make([]string, numNodes)
	now := time.Now()

	for i := range numNodes {
		uuid := fmt.Sprintf("bench-node-%d", i)
		uuids[i] = uuid
		node := &types.Node{
			Uuid:       uuid,
			Name:       fmt.Sprintf("Benchmark Node %d", i),
			Type:       types.EntityNodeType,
			GroupID:    "bench-group",
			EntityType: "BenchEntity",
			Summary:    fmt.Sprintf("Benchmark node %d summary", i),
			CreatedAt:  now,
			UpdatedAt:  now,
		}
		if err := d.UpsertNode(ctx, node); err != nil {
			b.Fatalf("Failed to create node %d: %v", i, err)
		}
	}

	// Benchmark: GetNodes with batch query (current implementation)
	b.Run("BatchQuery_100nodes", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := d.GetNodes(ctx, uuids, "bench-group")
			if err != nil {
				b.Fatalf("GetNodes failed: %v", err)
			}
		}
	})

	// Benchmark: Simulating individual queries (N+1 pattern)
	b.Run("IndividualQueries_100nodes", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, uuid := range uuids {
				_, err := d.GetNode(ctx, uuid, "bench-group")
				if err != nil {
					b.Fatalf("GetNode failed for %s: %v", uuid, err)
				}
			}
		}
	})

	// Benchmark with different batch sizes
	for _, batchSize := range []int{10, 25, 50} {
		b.Run(fmt.Sprintf("BatchQuery_%dnodes", batchSize), func(b *testing.B) {
			subset := uuids[:batchSize]
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := d.GetNodes(ctx, subset, "bench-group")
				if err != nil {
					b.Fatalf("GetNodes failed: %v", err)
				}
			}
		})

		b.Run(fmt.Sprintf("IndividualQueries_%dnodes", batchSize), func(b *testing.B) {
			subset := uuids[:batchSize]
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for _, uuid := range subset {
					_, err := d.GetNode(ctx, uuid, "bench-group")
					if err != nil {
						b.Fatalf("GetNode failed: %v", err)
					}
				}
			}
		})
	}
}

// BenchmarkGetEdgesBatch compares batch query vs individual queries for GetEdges.
func BenchmarkGetEdgesBatch(b *testing.B) {
	tempDir := b.TempDir()
	dbPath := filepath.Join(tempDir, "bench_ladybug_edges.db")
	d, err := driver.NewLadybugDriver(dbPath, 1)
	if err != nil {
		b.Fatalf("Failed to create driver: %v", err)
	}
	defer d.Close()

	ctx := context.Background()

	// Create indices
	if err := d.CreateIndices(ctx); err != nil {
		b.Fatalf("Failed to create indices: %v", err)
	}

	// Setup: create test nodes first
	const numEdges = 50
	now := time.Now()

	// Create source and target nodes
	for i := range numEdges * 2 {
		node := &types.Node{
			Uuid:       fmt.Sprintf("bench-edge-node-%d", i),
			Name:       fmt.Sprintf("Edge Node %d", i),
			Type:       types.EntityNodeType,
			GroupID:    "bench-group",
			EntityType: "BenchEntity",
			CreatedAt:  now,
			UpdatedAt:  now,
		}
		if err := d.UpsertNode(ctx, node); err != nil {
			b.Fatalf("Failed to create node %d: %v", i, err)
		}
	}

	// Create test edges
	edgeUUIDs := make([]string, numEdges)
	for i := range numEdges {
		uuid := fmt.Sprintf("bench-edge-%d", i)
		edgeUUIDs[i] = uuid
		edge := &types.Edge{
			BaseEdge: types.BaseEdge{
				Uuid:         uuid,
				GroupID:      "bench-group",
				SourceNodeID: fmt.Sprintf("bench-edge-node-%d", i*2),
				TargetNodeID: fmt.Sprintf("bench-edge-node-%d", i*2+1),
				CreatedAt:    now,
			},
			SourceID:  fmt.Sprintf("bench-edge-node-%d", i*2),
			TargetID:  fmt.Sprintf("bench-edge-node-%d", i*2+1),
			Type:      types.EntityEdgeType,
			UpdatedAt: now,
			Name:      "BENCH_RELATION",
			Fact:      fmt.Sprintf("Benchmark edge %d fact", i),
		}
		if err := d.UpsertEdge(ctx, edge); err != nil {
			b.Fatalf("Failed to create edge %d: %v", i, err)
		}
	}

	// Benchmark: GetEdges with batch query
	b.Run("BatchQuery_50edges", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := d.GetEdges(ctx, edgeUUIDs, "bench-group")
			if err != nil {
				b.Fatalf("GetEdges failed: %v", err)
			}
		}
	})

	// Benchmark: Simulating individual queries
	b.Run("IndividualQueries_50edges", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, uuid := range edgeUUIDs {
				_, err := d.GetEdge(ctx, uuid, "bench-group")
				if err != nil {
					b.Fatalf("GetEdge failed for %s: %v", uuid, err)
				}
			}
		}
	})

	// Benchmark with smaller batch
	b.Run("BatchQuery_10edges", func(b *testing.B) {
		subset := edgeUUIDs[:10]
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, err := d.GetEdges(ctx, subset, "bench-group")
			if err != nil {
				b.Fatalf("GetEdges failed: %v", err)
			}
		}
	})

	b.Run("IndividualQueries_10edges", func(b *testing.B) {
		subset := edgeUUIDs[:10]
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for _, uuid := range subset {
				_, err := d.GetEdge(ctx, uuid, "bench-group")
				if err != nil {
					b.Fatalf("GetEdge failed: %v", err)
				}
			}
		}
	})
}
