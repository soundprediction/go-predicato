//go:build cgo

package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/types"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <topic-graph-dir> [--dry-run]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  Scans all topic graph databases and removes low-value edges.\n")
		fmt.Fprintf(os.Stderr, "  --dry-run  Show what would be removed without deleting\n")
		os.Exit(1)
	}

	dir := os.Args[1]
	dryRun := false
	for _, arg := range os.Args[2:] {
		if arg == "--dry-run" {
			dryRun = true
		}
	}

	if dryRun {
		fmt.Println("=== DRY RUN MODE ===")
	}

	// Find all databases
	entries, err := os.ReadDir(dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to read directory: %v\n", err)
		os.Exit(1)
	}

	var totalRemoved, totalKept, totalDBs int

	for _, entry := range entries {
		name := entry.Name()
		fullPath := filepath.Join(dir, name)

		var dbType string
		var slug string

		if strings.HasSuffix(name, ".duckdb") {
			dbType = "duckpgq"
			slug = strings.TrimSuffix(name, ".duckdb")
		} else if strings.HasSuffix(name, ".ladybug") {
			dbType = "ladybug"
			slug = strings.TrimSuffix(name, ".ladybug")
		} else if strings.HasSuffix(name, ".lbug") {
			dbType = "ladybug"
			slug = strings.TrimSuffix(name, ".lbug")
		} else {
			continue
		}

		fmt.Printf("\n--- %s (%s) ---\n", slug, dbType)

		removed, kept, err := cleanupDB(fullPath, dbType, slug, dryRun)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  ERROR: %v\n", err)
			continue
		}

		totalRemoved += removed
		totalKept += kept
		totalDBs++
	}

	fmt.Printf("\n=== SUMMARY ===\n")
	fmt.Printf("Databases processed: %d\n", totalDBs)
	fmt.Printf("Edges kept: %d\n", totalKept)
	fmt.Printf("Edges removed: %d\n", totalRemoved)
	if totalKept+totalRemoved > 0 {
		fmt.Printf("Removal rate: %.1f%%\n", float64(totalRemoved)/float64(totalKept+totalRemoved)*100)
	}
}

func cleanupDB(dbPath, dbType, slug string, dryRun bool) (removed, kept int, err error) {
	ctx := context.Background()

	switch dbType {
	case "duckpgq":
		return cleanupDuckPGQ(ctx, dbPath, slug, dryRun)
	case "ladybug":
		return cleanupLadybug(ctx, dbPath, slug, dryRun)
	default:
		return 0, 0, fmt.Errorf("unsupported db type: %s", dbType)
	}
}

func cleanupDuckPGQ(ctx context.Context, dbPath, slug string, dryRun bool) (removed, kept int, err error) {
	d, err := driver.NewDuckPGQDriverWithConfig(driver.DuckPGQDriverConfig{
		URI:          dbPath,
		EmbeddingDim: 1024,
		ReadOnly:     dryRun, // read-write only when actually deleting
	})
	if err != nil {
		return 0, 0, fmt.Errorf("failed to open: %w", err)
	}
	defer d.Close()

	var toDelete []string

	err = d.IterateEdges(ctx, "wikidata", func(e *types.Edge) error {
		if shouldRemoveEdge(e) {
			toDelete = append(toDelete, e.Uuid)
			return nil
		}
		kept++
		return nil
	})
	if err != nil {
		// Try with default group
		toDelete = nil
		kept = 0
		err = d.IterateEdges(ctx, "default", func(e *types.Edge) error {
			if shouldRemoveEdge(e) {
				toDelete = append(toDelete, e.Uuid)
				return nil
			}
			kept++
			return nil
		})
		if err != nil {
			return 0, 0, fmt.Errorf("iterate failed: %w", err)
		}
	}

	removed = len(toDelete)
	printStats(slug, kept, removed, toDelete, dryRun)

	if !dryRun && removed > 0 {
		fmt.Printf("  Deleting %d edges...\n", removed)
		for i, uuid := range toDelete {
			if err := d.DeleteEdge(ctx, uuid, "wikidata"); err != nil {
				// Try default group
				_ = d.DeleteEdge(ctx, uuid, "default")
			}
			if (i+1)%1000 == 0 {
				fmt.Printf("    deleted %d/%d\n", i+1, removed)
			}
		}
		fmt.Printf("  Done.\n")
	}

	return removed, kept, nil
}

func cleanupLadybug(ctx context.Context, dbPath, slug string, dryRun bool) (removed, kept int, err error) {
	cfg := driver.DefaultLadybugDriverConfig()
	cfg.DBPath = dbPath
	cfg.ReadOnly = dryRun
	cfg.BufferPoolSize = 768 * 1024 * 1024

	d, err := driver.NewLadybugDriverWithConfig(cfg)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to open: %w", err)
	}
	defer d.Close()

	// Query all edges via Cypher
	query := `
		MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)
		RETURN e.uuid AS uuid, e.group_id AS group_id, e.name AS name,
		       e.fact AS fact, n.uuid AS source_uuid, m.uuid AS target_uuid
	`

	result, _, _, err := d.ExecuteQuery(ctx, query, nil)
	if err != nil {
		return 0, 0, fmt.Errorf("query all edges failed: %w", err)
	}

	rows, ok := result.([]map[string]any)
	if !ok {
		return 0, 0, fmt.Errorf("unexpected result type: %T", result)
	}

	var toDelete []struct {
		uuid    string
		groupID string
	}

	for _, row := range rows {
		uuid, _ := row["uuid"].(string)
		groupID, _ := row["group_id"].(string)
		name, _ := row["name"].(string)
		fact, _ := row["fact"].(string)

		edge := &types.Edge{
			BaseEdge: types.BaseEdge{Uuid: uuid, GroupID: groupID},
			Name:     name,
			Fact:     fact,
		}

		if shouldRemoveEdge(edge) {
			toDelete = append(toDelete, struct {
				uuid    string
				groupID string
			}{uuid, groupID})
		} else {
			kept++
		}
	}

	removed = len(toDelete)

	// Print sample of what's being removed
	printStatsLadybug(slug, kept, removed, rows, dryRun)

	if !dryRun && removed > 0 {
		fmt.Printf("  Deleting %d edges...\n", removed)
		deleteStart := time.Now()
		for i, item := range toDelete {
			if err := d.DeleteEdge(ctx, item.uuid, item.groupID); err != nil {
				fmt.Fprintf(os.Stderr, "    delete failed for %s: %v\n", item.uuid, err)
			}
			if (i+1)%500 == 0 {
				fmt.Printf("    deleted %d/%d (%.0fs elapsed)\n", i+1, removed, time.Since(deleteStart).Seconds())
			}
		}
		fmt.Printf("  Done in %.1fs.\n", time.Since(deleteStart).Seconds())
	}

	return removed, kept, nil
}

func shouldRemoveEdge(e *types.Edge) bool {
	// 1. Check for generic terms in fact
	src, tgt := types.ExtractFactConcepts(e.Fact, e.Name)
	if types.IsGenericTerm(tgt) || types.IsGenericTerm(src) {
		return true
	}

	// 2. Tautological check (skip IS_A)
	if e.Name != "IS_A" && types.IsTautological(src, tgt) {
		return true
	}

	// 3. Clinical trial metadata edges (HAS_CONDITION from trials)
	if types.IsClinicalMetadataEdge(e.Name, src, tgt) {
		return true
	}

	// 4. Also check subject/target directly if fact parsing didn't find concepts
	// For edges where fact text doesn't follow standard patterns, check names
	if src == "" && tgt == "" {
		// Try to extract from the fact by splitting on the predicate name
		lowerFact := strings.ToLower(e.Fact)
		lowerPred := strings.ToLower(e.Name)
		if idx := strings.Index(lowerFact, lowerPred); idx > 0 {
			src = strings.TrimSpace(e.Fact[:idx])
			tgt = strings.TrimSpace(e.Fact[idx+len(e.Name):])
			if types.IsGenericTerm(tgt) || types.IsGenericTerm(src) {
				return true
			}
			if e.Name != "IS_A" && types.IsTautological(src, tgt) {
				return true
			}
		}
	}

	return false
}

func printStats(slug string, kept, removed int, toDelete []string, dryRun bool) {
	total := kept + removed
	if total == 0 {
		fmt.Printf("  No edges found\n")
		return
	}
	fmt.Printf("  Total: %d, Keep: %d, Remove: %d (%.1f%%)\n",
		total, kept, removed, float64(removed)/float64(total)*100)
}

func printStatsLadybug(slug string, kept, removed int, rows []map[string]any, dryRun bool) {
	total := kept + removed
	if total == 0 {
		fmt.Printf("  No edges found\n")
		return
	}
	fmt.Printf("  Total: %d, Keep: %d, Remove: %d (%.1f%%)\n",
		total, kept, removed, float64(removed)/float64(total)*100)

	// Show sample of removed edges
	count := 0
	for _, row := range rows {
		name, _ := row["name"].(string)
		fact, _ := row["fact"].(string)
		uuid, _ := row["uuid"].(string)
		groupID, _ := row["group_id"].(string)

		edge := &types.Edge{
			BaseEdge: types.BaseEdge{Uuid: uuid, GroupID: groupID},
			Name:     name,
			Fact:     fact,
		}

		if shouldRemoveEdge(edge) {
			if count < 5 {
				factPreview := fact
				if len(factPreview) > 80 {
					factPreview = factPreview[:80] + "..."
				}
				fmt.Printf("    [REMOVE] [%s] %s\n", name, factPreview)
			}
			count++
		}
	}
	if count > 5 {
		fmt.Printf("    ... and %d more\n", count-5)
	}
}
