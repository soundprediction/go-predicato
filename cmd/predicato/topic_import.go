package predicato

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/soundprediction/predicato/pkg/config"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/embedder"
	"github.com/spf13/cobra"
)

// FilteredParquetImporter is implemented by drivers that support topic-filtered parquet import.
type FilteredParquetImporter interface {
	BulkLoadFromParquetWithFilter(ctx context.Context, inputDir, groupID string, filter *driver.TopicFilter) (int64, int64, int64, error)
}

var topicImportCmd = &cobra.Command{
	Use:   "topic-import",
	Short: "Import a topic-filtered subset of parquet files into a knowledge graph",
	Long: `Load a semantically filtered subset of predicato-format parquet files into a
knowledge graph. Only triples and rules whose embeddings are similar to the
given topic vector (cosine similarity >= threshold) are loaded. Nodes are
restricted to those referenced by the selected triples and rules.

Exactly one of --topic or --topic-vector must be provided:
  --topic "diabetes mellitus"      embeds the string at runtime via the configured embedder
  --topic-vector /path/vector.json loads a pre-computed []float32 JSON array (offline-safe)

Examples:
  # Embed topic at runtime (requires embedder config):
  predicato topic-import --input-dir ./wikidata-output --output ./diabetes.duckdb \
    --topic "diabetes mellitus type 2 treatment" --threshold 0.65

  # Use a pre-computed vector (no API key needed):
  predicato topic-import --input-dir ./wikidata-output --output ./diabetes.duckdb \
    --topic-vector ./diabetes_vector.json --threshold 0.65`,
	RunE: runTopicImport,
}

func init() {
	rootCmd.AddCommand(topicImportCmd)

	topicImportCmd.Flags().String("driver", "duckpgq", "graph driver (duckpgq)")
	topicImportCmd.Flags().String("input-dir", "", "directory containing parquet files")
	topicImportCmd.Flags().String("output", "", "output database file path")
	topicImportCmd.Flags().String("topic", "", "topic string to embed at runtime")
	topicImportCmd.Flags().String("topic-vector", "", "path to JSON file with pre-computed []float32 vector")
	topicImportCmd.Flags().Float64("threshold", 0.65, "minimum cosine similarity [0,1] for inclusion")
	topicImportCmd.Flags().String("group-id", "wikidata", "group ID for multi-tenant isolation")
	topicImportCmd.Flags().Int("embedding-dim", 1024, "embedding vector dimension")
	topicImportCmd.Flags().Int("threads", 0, "DuckDB thread count (0 = system default, duckpgq only)")
	topicImportCmd.Flags().String("memory-limit", "", "DuckDB memory limit (e.g., '900GB', duckpgq only)")
	topicImportCmd.Flags().String("temp-dir", "", "DuckDB temp directory for disk spill (duckpgq only, defaults to system temp)")
	topicImportCmd.Flags().Int64("max-nodes", 0, "maximum nodes to insert (0 = unlimited)")
	topicImportCmd.Flags().Int64("max-edges", 0, "maximum edges to insert (0 = unlimited)")
	topicImportCmd.Flags().Bool("force", false, "overwrite output file if it exists")
	topicImportCmd.Flags().Bool("skip-indexes", false, "skip index creation")

	_ = topicImportCmd.MarkFlagRequired("input-dir")
	_ = topicImportCmd.MarkFlagRequired("output")
}

func runTopicImport(cmd *cobra.Command, args []string) error {
	driverName, _ := cmd.Flags().GetString("driver")
	inputDir, _ := cmd.Flags().GetString("input-dir")
	output, _ := cmd.Flags().GetString("output")
	topicStr, _ := cmd.Flags().GetString("topic")
	topicVectorPath, _ := cmd.Flags().GetString("topic-vector")
	threshold, _ := cmd.Flags().GetFloat64("threshold")
	groupID, _ := cmd.Flags().GetString("group-id")
	embDim, _ := cmd.Flags().GetInt("embedding-dim")
	threads, _ := cmd.Flags().GetInt("threads")
	memLimit, _ := cmd.Flags().GetString("memory-limit")
	tempDir, _ := cmd.Flags().GetString("temp-dir")
	maxNodes, _ := cmd.Flags().GetInt64("max-nodes")
	maxEdges, _ := cmd.Flags().GetInt64("max-edges")
	force, _ := cmd.Flags().GetBool("force")
	skipIndexes, _ := cmd.Flags().GetBool("skip-indexes")

	// Validate: exactly one of --topic or --topic-vector required.
	if topicStr == "" && topicVectorPath == "" {
		return fmt.Errorf("exactly one of --topic or --topic-vector must be provided")
	}
	if topicStr != "" && topicVectorPath != "" {
		return fmt.Errorf("exactly one of --topic or --topic-vector must be provided, not both")
	}

	// Validate input directory.
	info, err := os.Stat(inputDir)
	if err != nil || !info.IsDir() {
		return fmt.Errorf("input directory not found: %s", inputDir)
	}

	// Check output path.
	if _, err := os.Stat(output); err == nil {
		if !force {
			return fmt.Errorf("output file already exists: %s (use --force to overwrite)", output)
		}
		os.Remove(output)
		os.Remove(output + ".wal")
	}

	// Build or load the topic vector.
	var topicVec []float32
	if topicVectorPath != "" {
		topicVec, err = loadVectorJSON(topicVectorPath)
		if err != nil {
			return fmt.Errorf("failed to load topic vector from %s: %w", topicVectorPath, err)
		}
		if len(topicVec) != embDim {
			return fmt.Errorf("vector in %s has %d dimensions, expected %d (adjust --embedding-dim)", topicVectorPath, len(topicVec), embDim)
		}
	} else {
		cfg, err := config.Load()
		if err != nil {
			return fmt.Errorf("failed to load config for embedder: %w", err)
		}
		if cfg.Embedding.APIKey == "" {
			return fmt.Errorf("--topic requires an embedder API key; set PREDICATO_EMBEDDING_API_KEY or use --topic-vector for offline use")
		}
		emb := embedder.NewOpenAIEmbedder(cfg.Embedding.APIKey, embedder.Config{
			Model:   cfg.Embedding.Model,
			BaseURL: cfg.Embedding.BaseURL,
		})
		defer emb.Close()

		ctx := context.Background()
		topicVec, err = emb.EmbedSingle(ctx, topicStr)
		if err != nil {
			return fmt.Errorf("failed to embed topic %q: %w", topicStr, err)
		}
		if len(topicVec) != embDim {
			return fmt.Errorf("embedder returned %d dimensions, expected %d (adjust --embedding-dim)", len(topicVec), embDim)
		}
	}

	fmt.Printf("=== Topic Import -> %s Knowledge Graph ===\n", driverName)
	if topicStr != "" {
		fmt.Printf("Topic:          %s\n", topicStr)
	} else {
		fmt.Printf("Topic vector:   %s (%d dims)\n", topicVectorPath, len(topicVec))
	}
	fmt.Printf("Threshold:      %.2f\n", threshold)
	fmt.Printf("Driver:         %s\n", driverName)
	fmt.Printf("Input:          %s\n", inputDir)
	fmt.Printf("Output:         %s\n", output)
	fmt.Printf("Group ID:       %s\n", groupID)
	fmt.Printf("Embedding dim:  %d\n", embDim)
	if maxNodes > 0 {
		fmt.Printf("Max nodes:      %d\n", maxNodes)
	}
	if maxEdges > 0 {
		fmt.Printf("Max edges:      %d\n", maxEdges)
	}
	fmt.Println()

	totalStart := time.Now()

	graphDriver, err := newGraphDriver(driverName, output, embDim)
	if err != nil {
		return fmt.Errorf("failed to create %s driver: %w", driverName, err)
	}
	defer graphDriver.Close()

	ctx := context.Background()

	// Driver-specific configuration.
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
		if tempDir != "" {
			if err := os.MkdirAll(tempDir, 0o755); err != nil {
				fmt.Printf("Warning: failed to create temp dir %s: %v\n", tempDir, err)
			}
			if _, _, _, err := graphDriver.ExecuteQuery(ctx, fmt.Sprintf("SET temp_directory = '%s'", tempDir), nil); err != nil {
				fmt.Printf("Warning: failed to set temp_directory: %v\n", err)
			}
			if _, _, _, err := graphDriver.ExecuteQuery(ctx, "SET max_temp_directory_size = '1TB'", nil); err != nil {
				fmt.Printf("Warning: failed to set max_temp_directory_size: %v\n", err)
			}
		}
	}

	filtImporter, ok := graphDriver.(FilteredParquetImporter)
	if !ok {
		return fmt.Errorf("driver %q does not support topic-filtered parquet import", driverName)
	}

	filter := &driver.TopicFilter{
		Embedding: topicVec,
		Threshold: threshold,
		MaxNodes:  maxNodes,
		MaxEdges:  maxEdges,
	}

	fmt.Println("Loading topic-filtered parquet files...")
	loadStart := time.Now()

	nodesLoaded, edgesLoaded, rulesLoaded, err := filtImporter.BulkLoadFromParquetWithFilter(ctx, inputDir, groupID, filter)
	if err != nil {
		return fmt.Errorf("bulk load failed: %w", err)
	}

	fmt.Printf("  Nodes:   %d\n", nodesLoaded)
	fmt.Printf("  Edges:   %d\n", edgesLoaded)
	fmt.Printf("  Rules:   %d\n", rulesLoaded)
	fmt.Printf("  Elapsed: %s\n\n", time.Since(loadStart).Round(time.Second))

	if !skipIndexes {
		fmt.Println("Creating indexes...")
		indexStart := time.Now()
		if err := graphDriver.CreateIndices(ctx); err != nil {
			fmt.Printf("Warning: index creation partially failed: %v\n", err)
		}
		fmt.Printf("  Elapsed: %s\n\n", time.Since(indexStart).Round(time.Second))
	}

	fmt.Println("=== Validation ===")
	stats, err := graphDriver.GetStats(ctx, groupID)
	if err != nil {
		fmt.Printf("Warning: could not get stats: %v\n", err)
	} else {
		fmt.Printf("Loaded %d nodes, %d edges\n", stats.NodeCount, stats.EdgeCount)
	}

	if driverName == "duckpgq" {
		if _, _, _, err := graphDriver.ExecuteQuery(ctx, "CHECKPOINT", nil); err != nil {
			fmt.Printf("Warning: checkpoint failed: %v\n", err)
		}
	}

	fmt.Printf("\n=== Complete ===\n")
	fmt.Printf("Total elapsed: %s\n", time.Since(totalStart).Round(time.Second))
	fmt.Printf("Output: %s\n", output)

	if fi, err := os.Stat(output); err == nil {
		sizeGB := float64(fi.Size()) / (1024 * 1024 * 1024)
		fmt.Printf("Size: %.2f GB\n", sizeGB)
	}

	return nil
}

// loadVectorJSON reads a JSON file containing a []float32 array.
func loadVectorJSON(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var vec []float32
	if err := json.Unmarshal(data, &vec); err != nil {
		return nil, fmt.Errorf("invalid JSON vector: %w", err)
	}
	if len(vec) == 0 {
		return nil, fmt.Errorf("vector file is empty")
	}
	return vec, nil
}
