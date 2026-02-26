package predicato

import (
	"context"
	"fmt"
	"os"

	"github.com/soundprediction/predicato/pkg/config"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/embedder"
	"github.com/spf13/cobra"
)

// ParquetExplorer is implemented by drivers that can inspect parquet files.
type ParquetExplorer interface {
	ExploreParquet(ctx context.Context, inputDir string) (*driver.ParquetStats, error)
}

var topicExploreCmd = &cobra.Command{
	Use:   "topic-explore",
	Short: "Inspect parquet files and print frequency statistics",
	Long: `Inspect predicato-format parquet files and print frequency distributions
of entity types, predicates, rule types, and embedding coverage.

Does not require an output database file — all analysis is done in memory.

Examples:
  # Print distribution of entity types, predicates, and embedding coverage:
  predicato topic-explore --input-dir ./wikidata-output

  # Print top-20 triples most similar to a topic (requires embedder config):
  predicato topic-explore --input-dir ./wikidata-output --topic "diabetes mellitus"`,
	RunE: runTopicExplore,
}

func init() {
	rootCmd.AddCommand(topicExploreCmd)

	topicExploreCmd.Flags().String("input-dir", "", "directory containing parquet files")
	topicExploreCmd.Flags().String("topic", "", "optional topic string to rank triples/rules by similarity (requires embedder config)")
	topicExploreCmd.Flags().Int("top", 20, "number of top results to show")
	topicExploreCmd.Flags().Int("embedding-dim", 1024, "embedding vector dimension")

	_ = topicExploreCmd.MarkFlagRequired("input-dir")
}

func runTopicExplore(cmd *cobra.Command, args []string) error {
	inputDir, _ := cmd.Flags().GetString("input-dir")
	topic, _ := cmd.Flags().GetString("topic")
	top, _ := cmd.Flags().GetInt("top")
	embDim, _ := cmd.Flags().GetInt("embedding-dim")

	info, err := os.Stat(inputDir)
	if err != nil || !info.IsDir() {
		return fmt.Errorf("input directory not found: %s", inputDir)
	}

	// Use an in-memory DuckPGQ driver for parquet exploration.
	d, err := driver.NewDuckPGQDriver("", embDim)
	if err != nil {
		return fmt.Errorf("failed to create in-memory driver: %w", err)
	}
	defer d.Close()

	ctx := context.Background()

	stats, err := d.ExploreParquet(ctx, inputDir)
	if err != nil {
		return fmt.Errorf("exploration failed: %w", err)
	}

	fmt.Println("=== Parquet Statistics ===")
	fmt.Printf("Nodes:   %d\n", stats.NodeCount)
	fmt.Printf("Triples: %d\n", stats.TripleCount)
	fmt.Printf("Rules:   %d\n\n", stats.RuleCount)

	fmt.Println("--- Top Entity Types ---")
	for i, tc := range stats.TopEntityTypes {
		if i >= top {
			break
		}
		fmt.Printf("  %-40s %10d\n", tc.Value, tc.Count)
	}

	fmt.Println("\n--- Top Predicates ---")
	for i, tc := range stats.TopPredicates {
		if i >= top {
			break
		}
		fmt.Printf("  %-40s %10d\n", tc.Value, tc.Count)
	}

	if len(stats.TopRuleTypes) > 0 {
		fmt.Println("\n--- Top Rule Types ---")
		for i, tc := range stats.TopRuleTypes {
			if i >= top {
				break
			}
			fmt.Printf("  %-40s %10d\n", tc.Value, tc.Count)
		}
	}

	if len(stats.TopScopes) > 0 {
		fmt.Println("\n--- Top Rule Scopes ---")
		for i, tc := range stats.TopScopes {
			if i >= top {
				break
			}
			fmt.Printf("  %-40s %10d\n", tc.Value, tc.Count)
		}
	}

	fmt.Println("\n--- Embedding Coverage ---")
	fmt.Printf("  Nodes:   %.1f%%\n", stats.EmbeddingCoverage.NodesFraction*100)
	fmt.Printf("  Triples: %.1f%%\n", stats.EmbeddingCoverage.TriplesFraction*100)
	fmt.Printf("  Rules:   %.1f%%\n", stats.EmbeddingCoverage.RulesFraction*100)

	if topic != "" {
		if err := runTopicSimilarityReport(ctx, d, inputDir, topic, top, embDim); err != nil {
			return err
		}
	}

	return nil
}

// runTopicSimilarityReport embeds the topic string and prints the top-N most similar
// triples and rules from the parquet files.
func runTopicSimilarityReport(ctx context.Context, d *driver.DuckPGQDriver, inputDir, topic string, top, embDim int) error {
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("failed to load config for embedder: %w", err)
	}
	if cfg.Embedding.APIKey == "" {
		return fmt.Errorf("--topic requires an embedder API key; set PREDICATO_EMBEDDING_API_KEY or configure embedding.api_key in ~/.pregnancy.yaml")
	}

	var emb embedder.Client
	switch cfg.Embedding.Provider {
	case "openai", "":
		emb = embedder.NewOpenAIEmbedder(cfg.Embedding.APIKey, embedder.Config{
			Model:   cfg.Embedding.Model,
			BaseURL: cfg.Embedding.BaseURL,
		})
	default:
		return fmt.Errorf("unsupported embedding provider %q; only 'openai' is supported for topic-explore", cfg.Embedding.Provider)
	}
	defer emb.Close()

	topicVec, err := emb.EmbedSingle(ctx, topic)
	if err != nil {
		return fmt.Errorf("failed to embed topic %q: %w", topic, err)
	}
	if len(topicVec) != embDim {
		return fmt.Errorf("embedder returned %d dimensions, expected %d (adjust --embedding-dim)", len(topicVec), embDim)
	}

	triples, err := d.TopSimilarTriples(ctx, inputDir, topicVec, top)
	if err != nil {
		return fmt.Errorf("similarity query failed: %w", err)
	}

	fmt.Printf("\n--- Top %d Triples Most Similar to %q ---\n", top, topic)
	for _, r := range triples {
		fmt.Printf("  [%.4f] %s %s %s\n", r.Similarity, r.Subject, r.Predicate, r.Object)
	}

	rules, err := d.TopSimilarRules(ctx, inputDir, topicVec, top)
	if err != nil {
		return fmt.Errorf("rules similarity query failed: %w", err)
	}
	if len(rules) > 0 {
		fmt.Printf("\n--- Top %d Rules Most Similar to %q ---\n", top, topic)
		for _, r := range rules {
			fmt.Printf("  [%.4f] IF %s THEN %s\n", r.Similarity, r.Antecedent, r.Consequent)
		}
	}

	return nil
}
