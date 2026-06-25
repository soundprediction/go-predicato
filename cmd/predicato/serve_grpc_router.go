//go:build cgo

package predicato

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	"github.com/soundprediction/predicato/pkg/config"
	"github.com/soundprediction/predicato/pkg/crossencoder"
	"github.com/soundprediction/predicato/pkg/embedder"
	"github.com/soundprediction/predicato/pkg/grpcsvc"
	predicatoLogger "github.com/soundprediction/predicato/pkg/logger"
	"github.com/soundprediction/predicato/pkg/router"
	"github.com/spf13/cobra"
)

// runServeGRPCRouterMode wires the existing multi-graph router engine
// (pkg/router) into the gRPC server: it discovers topic graphs under sourceDir,
// builds a shared embedder + NLP client (reusing the same helpers as the
// single-graph path, so no throwaway graph is opened), optionally builds a
// cross-encoder reranker, constructs the Manager/classifier/Router, and serves
// a RouterPredicato adapter that routes Search across topic graphs.
func runServeGRPCRouterMode(cmd *cobra.Command, cfg *config.Config, sourceDir, addr string) error {
	logger := slog.New(predicatoLogger.NewColorHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	strategy, _ := cmd.Flags().GetString("strategy")
	mergeStrategy, _ := cmd.Flags().GetString("merge-strategy")
	maxGraphs, _ := cmd.Flags().GetInt("max-graphs")
	minConfidence, _ := cmd.Flags().GetFloat64("min-confidence")
	maxOpenGraphs, _ := cmd.Flags().GetInt("max-open-graphs")

	routerCfg := router.RouterConfig{
		Strategy:      strategy,
		MergeStrategy: mergeStrategy,
		MinConfidence: minConfidence,
		MaxGraphs:     maxGraphs,
	}

	// Discover topic graphs and their routing descriptors.
	fmt.Printf("Discovering topic graphs in %s...\n", sourceDir)
	configs, routingTexts, err := router.DiscoverTopicGraphs(sourceDir)
	if err != nil {
		return fmt.Errorf("failed to discover topic graphs: %w", err)
	}
	if len(configs) == 0 {
		return fmt.Errorf("no topic graphs discovered in %s (expected <slug>.md paired with <slug>.duckdb/.lbug/.ladybug)", sourceDir)
	}

	// Build shared embedder + NLP client without opening a throwaway graph.
	fmt.Println("Initializing embedder...")
	embedderClient, err := buildEmbedder(cfg)
	if err != nil {
		return fmt.Errorf("failed to build embedder: %w", err)
	}
	nlpClient, err := buildNLP(cmd, cfg, &logger)
	if err != nil {
		return fmt.Errorf("failed to build NLP client: %w", err)
	}

	// Build a cross-encoder reranker when the cross_encoder strategy is selected.
	var reranker crossencoder.Client
	if routerCfg.GetStrategyOrDefault() == "cross_encoder" {
		reranker, err = buildReranker(cmd, embedderClient)
		if err != nil {
			return fmt.Errorf("failed to build reranker: %w", err)
		}
	}

	// Wire the router engine.
	mgr := router.NewManagerWithOptions(configs, nlpClient, embedderClient, logger, maxOpenGraphs)
	if reranker != nil {
		mgr.SetCrossEncoder(reranker)
	}

	classifier, err := router.NewTopicClassifier(configs, routerCfg, embedderClient, reranker, routingTexts)
	if err != nil {
		return fmt.Errorf("failed to build topic classifier: %w", err)
	}

	r := router.NewRouter(mgr, classifier, routerCfg, logger)

	def, err := mgr.GetDefaultClient()
	if err != nil {
		return fmt.Errorf("failed to get default client: %w", err)
	}

	adapter := router.NewRouterPredicato(r, def)

	// Startup banner.
	fmt.Printf("\nPredicato gRPC server (ROUTER MODE)\n")
	fmt.Printf("  topics:         %d\n", len(configs))
	fmt.Printf("  strategy:       %s\n", routerCfg.GetStrategyOrDefault())
	fmt.Printf("  merge:          %s\n", routerCfg.GetMergeStrategyOrDefault())
	fmt.Printf("  max-graphs:     %d\n", routerCfg.GetMaxGraphsOrDefault())
	fmt.Printf("  min-confidence: %.2f\n", routerCfg.GetMinConfidenceOrDefault())
	fmt.Printf("  max-open:       %d\n", maxOpenGraphs)
	for _, c := range configs {
		marker := ""
		if c.Default {
			marker = " (default)"
		}
		if c.Fallback {
			marker += " (fallback)"
		}
		fmt.Printf("    - %s%s\n", c.Name, marker)
	}
	fmt.Printf("Listening on %s (service %s)\n\n", addr, grpcsvc.ServiceName)

	// Ensure clients are closed on shutdown of Serve (best-effort).
	defer func() {
		if cerr := r.Close(context.Background()); cerr != nil {
			logger.Warn("error closing router", "error", cerr)
		}
	}()

	return grpcsvc.Serve(adapter, addr)
}

// buildReranker builds a cross-encoder reranker Client for cross_encoder
// routing. When --reranker-provider is empty it defaults to an embedding-based
// reranker that reuses the shared embedder (no extra model loaded).
func buildReranker(cmd *cobra.Command, embedderClient embedder.Client) (crossencoder.Client, error) {
	provider, _ := cmd.Flags().GetString("reranker-provider")
	model, _ := cmd.Flags().GetString("reranker-model")
	baseURL, _ := cmd.Flags().GetString("reranker-base-url")

	switch provider {
	case "", "embedding", "candle":
		if embedderClient == nil {
			return nil, fmt.Errorf("embedding reranker requires an embedder; none available")
		}
		fmt.Println("Reranker: embedding-based (reuses shared embedder)")
		return crossencoder.NewEmbeddingRerankerClient(embedderClient, crossencoder.EmbeddingConfig{}), nil

	case "reranker":
		fmt.Printf("Reranker: Jina-compatible API (%s)\n", baseURL)
		return crossencoder.NewRerankerClient(crossencoder.RerankerConfig{
			BaseURL: baseURL,
			Config:  crossencoder.Config{Model: model},
		}), nil

	case "local":
		fmt.Println("Reranker: local text-similarity")
		return crossencoder.NewLocalRerankerClient(crossencoder.Config{}), nil

	default:
		return nil, fmt.Errorf("unsupported reranker provider: %s (supported: embedding, reranker, local)", provider)
	}
}
