package predicato

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/spf13/cobra"
)

// ParquetImporter is implemented by graph drivers that support bulk parquet import.
type ParquetImporter interface {
	BulkLoadFromParquet(ctx context.Context, inputDir, groupID string) (int64, int64, int64, error)
}

var parquetImportCmd = &cobra.Command{
	Use:   "parquet-import",
	Short: "Import parquet files into a knowledge graph",
	Long: `Load predicato-format parquet files directly into a knowledge graph,
bypassing PostgreSQL entirely.

Supported drivers: duckpgq (DuckDB+DuckPGQ), ladybug (LadybugDB).

This is the fastest path for bulk-loading parquet files produced by the
wikidata pipeline (duckdb_to_predicato.py or wikidata_to_predicato.py).

Expected input files in --input-dir:
  nodes.parquet                  — entities with embeddings
  extracted_triples.parquet.gz   — triples with embeddings
  extracted_rules.parquet.gz     — rules (optional)

Examples:
  # DuckPGQ (default for HPC/batch use):
  predicato parquet-import --driver duckpgq --input-dir ./intermediate --output ./graph.duckdb

  # DuckPGQ with tuning for large-memory nodes:
  predicato parquet-import --driver duckpgq --input-dir ./intermediate --output ./graph.duckdb --threads 24 --memory-limit 900GB

  # Ladybug:
  predicato parquet-import --driver ladybug --input-dir ./intermediate --output ./graph.db`,
	RunE: runParquetImport,
}

func init() {
	rootCmd.AddCommand(parquetImportCmd)

	parquetImportCmd.Flags().String("driver", "duckpgq", "graph driver (duckpgq, ladybug)")
	parquetImportCmd.Flags().String("input-dir", "", "directory containing parquet files")
	parquetImportCmd.Flags().String("output", "", "output database file path")
	parquetImportCmd.Flags().String("group-id", "wikidata", "group ID for multi-tenant isolation")
	parquetImportCmd.Flags().Int("embedding-dim", 1024, "embedding vector dimension")
	parquetImportCmd.Flags().Int("threads", 0, "DuckDB thread count (0 = system default, duckpgq only)")
	parquetImportCmd.Flags().String("memory-limit", "", "DuckDB memory limit (e.g., '900GB', duckpgq only)")
	parquetImportCmd.Flags().Bool("force", false, "overwrite output file if it exists")
	parquetImportCmd.Flags().Bool("skip-indexes", false, "skip index creation")

	_ = parquetImportCmd.MarkFlagRequired("input-dir")
	_ = parquetImportCmd.MarkFlagRequired("output")
}

func runParquetImport(cmd *cobra.Command, args []string) error {
	driverName, _ := cmd.Flags().GetString("driver")
	inputDir, _ := cmd.Flags().GetString("input-dir")
	output, _ := cmd.Flags().GetString("output")
	groupID, _ := cmd.Flags().GetString("group-id")
	embDim, _ := cmd.Flags().GetInt("embedding-dim")
	threads, _ := cmd.Flags().GetInt("threads")
	memLimit, _ := cmd.Flags().GetString("memory-limit")
	force, _ := cmd.Flags().GetBool("force")
	skipIndexes, _ := cmd.Flags().GetBool("skip-indexes")

	// Validate input directory
	info, err := os.Stat(inputDir)
	if err != nil || !info.IsDir() {
		return fmt.Errorf("input directory not found: %s", inputDir)
	}

	// Check output path
	if _, err := os.Stat(output); err == nil {
		if !force {
			return fmt.Errorf("output file already exists: %s (use --force to overwrite)", output)
		}
		os.Remove(output)
		os.Remove(output + ".wal")
	}

	fmt.Printf("=== Parquet -> %s Knowledge Graph ===\n", driverName)
	fmt.Printf("Driver:         %s\n", driverName)
	fmt.Printf("Input:          %s\n", inputDir)
	fmt.Printf("Output:         %s\n", output)
	fmt.Printf("Group ID:       %s\n", groupID)
	fmt.Printf("Embedding dim:  %d\n", embDim)
	if threads > 0 {
		fmt.Printf("Threads:        %d\n", threads)
	}
	if memLimit != "" {
		fmt.Printf("Memory limit:   %s\n", memLimit)
	}
	fmt.Println()

	totalStart := time.Now()

	// Create the graph driver
	graphDriver, err := newGraphDriver(driverName, output, embDim)
	if err != nil {
		return fmt.Errorf("failed to create %s driver: %w", driverName, err)
	}
	defer graphDriver.Close()

	ctx := context.Background()

	// Driver-specific configuration
	if driverName == "duckpgq" {
		if threads > 0 {
			if _, _, _, err := graphDriver.ExecuteQuery(ctx, fmt.Sprintf("SET threads TO %d", threads), nil); err != nil {
				fmt.Printf("Warning: failed to set threads: %v\n", err)
			}
		}
		if memLimit != "" {
			if _, _, _, err := graphDriver.ExecuteQuery(ctx, fmt.Sprintf("SET memory_limit = '%s'", memLimit), nil); err != nil {
				fmt.Printf("Warning: failed to set memory limit: %v\n", err)
			}
		}
	}

	// Check that the driver supports parquet import
	importer, ok := graphDriver.(ParquetImporter)
	if !ok {
		return fmt.Errorf("driver %q does not support parquet import", driverName)
	}

	// Bulk load from parquet
	fmt.Println("Loading parquet files...")
	loadStart := time.Now()

	nodesLoaded, edgesLoaded, rulesLoaded, err := importer.BulkLoadFromParquet(ctx, inputDir, groupID)
	if err != nil {
		return fmt.Errorf("bulk load failed: %w", err)
	}

	fmt.Printf("  Nodes:  %d\n", nodesLoaded)
	fmt.Printf("  Edges:  %d\n", edgesLoaded)
	fmt.Printf("  Rules:  %d\n", rulesLoaded)
	fmt.Printf("  Elapsed: %s\n\n", time.Since(loadStart).Round(time.Second))

	// Create indexes
	if !skipIndexes {
		fmt.Println("Creating indexes...")
		indexStart := time.Now()
		if err := graphDriver.CreateIndices(ctx); err != nil {
			fmt.Printf("Warning: index creation partially failed: %v\n", err)
		}
		fmt.Printf("  Elapsed: %s\n\n", time.Since(indexStart).Round(time.Second))
	}

	// Validation stats
	fmt.Println("=== Validation ===")
	stats, err := graphDriver.GetStats(ctx, groupID)
	if err != nil {
		fmt.Printf("Warning: could not get stats: %v\n", err)
	} else {
		fmt.Printf("Total nodes: %d\n", stats.NodeCount)
		fmt.Printf("Total edges: %d\n", stats.EdgeCount)
		if len(stats.NodesByType) > 0 {
			fmt.Println("\nNode type distribution:")
			for t, c := range stats.NodesByType {
				fmt.Printf("  %-30s %10d\n", t, c)
			}
		}
		if len(stats.EdgesByType) > 0 {
			fmt.Println("\nEdge type distribution:")
			for t, c := range stats.EdgesByType {
				fmt.Printf("  %-30s %10d\n", t, c)
			}
		}
	}

	// DuckPGQ-specific: checkpoint
	if driverName == "duckpgq" {
		if _, _, _, err := graphDriver.ExecuteQuery(ctx, "CHECKPOINT", nil); err != nil {
			fmt.Printf("Warning: checkpoint failed: %v\n", err)
		}
	}

	fmt.Printf("\n=== Complete ===\n")
	fmt.Printf("Total elapsed: %s\n", time.Since(totalStart).Round(time.Second))
	fmt.Printf("Output: %s\n", output)

	if info, err := os.Stat(output); err == nil {
		sizeGB := float64(info.Size()) / (1024 * 1024 * 1024)
		fmt.Printf("Size: %.2f GB\n", sizeGB)
	}

	return nil
}
