package predicato

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/soundprediction/predicato/pkg/config"
	"github.com/soundprediction/predicato/pkg/grpcsvc"
	"github.com/spf13/cobra"
)

var grpcAddr string

var serveGRPCCmd = &cobra.Command{
	Use:   "serve-grpc",
	Short: "Start the Predicato knowledge-graph gRPC server",
	Long: `Run the Predicato knowledge graph as a standalone gRPC server so other
services (e.g. humn) can query and ingest over the network — in a separate process
or on a different machine.

It exposes the core graph operations (Search, AddEpisode, GetNode, GetEdge,
GetStats, Health) over gRPC using a JSON codec, so Go callers share predicato's
types with no proto codegen. Configure the graph/embedder/NLP the same way as the
HTTP 'server' command (config file, env, or flags).`,
	RunE: runServeGRPC,
}

func init() {
	rootCmd.AddCommand(serveGRPCCmd)

	serveGRPCCmd.Flags().StringVar(&grpcAddr, "addr", ":50071", "gRPC listen address (host:port)")

	// Mirror the HTTP server's graph/NLP/embedder flags so initializePredicato can
	// build the engine the same way (overrideConfigWithFlags reads these).
	defaultDBPath := "./ladybug_db"
	if home, err := os.UserHomeDir(); err == nil {
		defaultDBPath = filepath.Join(home, ".predicato", "ladybug_db")
	}
	serveGRPCCmd.Flags().String("db-driver", "ladybug", "Database driver (ladybug, neo4j, falkordb)")
	serveGRPCCmd.Flags().String("db-uri", defaultDBPath, "Database URI/path")
	serveGRPCCmd.Flags().String("db-username", "", "Database username (not used for ladybug)")
	serveGRPCCmd.Flags().String("db-password", "", "Database password (not used for ladybug)")
	serveGRPCCmd.Flags().String("db-database", "", "Database name (not used for ladybug)")

	serveGRPCCmd.Flags().BoolVar(&useGLiNER2, "gliner2", false, "Use GLiNER2 as the NLP provider")
	serveGRPCCmd.Flags().String("gliner2-endpoint", "http://localhost:11435", "GLiNER2 service endpoint")

	serveGRPCCmd.Flags().String("nlp-provider", "openai", "NLP provider")
	serveGRPCCmd.Flags().String("nlp-model", "gpt-4", "NLP model")
	serveGRPCCmd.Flags().String("nlp-api-key", "", "NLP API key")
	serveGRPCCmd.Flags().String("nlp-base-url", "", "NLP base URL")
	serveGRPCCmd.Flags().Float32("nlp-temperature", 0.1, "NLP temperature")
	serveGRPCCmd.Flags().Int("nlp-max-tokens", 2048, "NLP max tokens")

	serveGRPCCmd.Flags().String("embedding-provider", "openai", "Embedding provider")
	serveGRPCCmd.Flags().String("embedding-model", "text-embedding-3-small", "Embedding model")
	serveGRPCCmd.Flags().String("embedding-api-key", "", "Embedding API key")
	serveGRPCCmd.Flags().String("embedding-base-url", "", "Embedding base URL")

	serveGRPCCmd.Flags().String("telemetry-parquet-path", "", "Path to directory for telemetry")

	// Router-mode flags. When --source-dir is set, the server discovers topic
	// graphs in that directory and routes/merges queries across them instead of
	// serving a single graph.
	serveGRPCCmd.Flags().String("source-dir", "", "Directory of topic graphs (<slug>.duckdb/.lbug + <slug>.md). When set, enables multi-graph router mode.")
	serveGRPCCmd.Flags().String("strategy", "cross_encoder", "Routing strategy: cross_encoder, embedding, keyword, topic_based")
	serveGRPCCmd.Flags().String("merge-strategy", "rrf", "Result merge strategy: rrf, simple_concat, max_score")
	serveGRPCCmd.Flags().Int("max-graphs", 2, "Max number of topic graphs to query per request")
	serveGRPCCmd.Flags().Float64("min-confidence", 0.3, "Minimum routing confidence to include a topic graph")
	serveGRPCCmd.Flags().Int("max-open-graphs", 8, "Max number of topic graph databases kept open simultaneously (LRU eviction)")

	// Reranker selection for cross_encoder routing. When empty, an
	// embedding-based reranker is built from the shared embedder (no extra model).
	serveGRPCCmd.Flags().String("reranker-provider", "", "Cross-encoder reranker provider for cross_encoder strategy: embedding (default), reranker, local, openai")
	serveGRPCCmd.Flags().String("reranker-model", "", "Reranker model name (for reranker provider)")
	serveGRPCCmd.Flags().String("reranker-base-url", "", "Reranker base URL (for Jina-compatible reranker provider)")
}

func runServeGRPC(cmd *cobra.Command, _ []string) error {
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}
	overrideConfigWithFlags(cmd, cfg)

	// Multi-graph router mode: when --source-dir is set, discover topic graphs
	// in that directory and route/merge queries across them.
	if sourceDir, _ := cmd.Flags().GetString("source-dir"); sourceDir != "" {
		return runServeGRPCRouterMode(cmd, cfg, sourceDir, grpcAddr)
	}

	// Single-graph mode (unchanged).
	fmt.Println("Initializing Predicato...")
	predicatoInstance, _, _, err := initializePredicato(cmd, cfg)
	if err != nil {
		return fmt.Errorf("failed to initialize Predicato: %w", err)
	}

	fmt.Printf("Predicato gRPC server listening on %s (service %s)\n", grpcAddr, grpcsvc.ServiceName)
	return grpcsvc.Serve(predicatoInstance, grpcAddr)
}
