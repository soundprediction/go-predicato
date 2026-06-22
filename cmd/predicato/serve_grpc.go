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
}

func runServeGRPC(cmd *cobra.Command, _ []string) error {
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}
	overrideConfigWithFlags(cmd, cfg)

	fmt.Println("Initializing Predicato...")
	predicatoInstance, _, _, err := initializePredicato(cmd, cfg)
	if err != nil {
		return fmt.Errorf("failed to initialize Predicato: %w", err)
	}

	fmt.Printf("Predicato gRPC server listening on %s (service %s)\n", grpcAddr, grpcsvc.ServiceName)
	return grpcsvc.Serve(predicatoInstance, grpcAddr)
}
