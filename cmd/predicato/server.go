package predicato

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/soundprediction/predicato"
	"github.com/soundprediction/predicato/pkg/config"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/embedder"
	"github.com/soundprediction/predicato/pkg/factstore"
	predicatoLogger "github.com/soundprediction/predicato/pkg/logger"
	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/rustbert"
	"github.com/soundprediction/predicato/pkg/server"
	"github.com/soundprediction/predicato/pkg/telemetry"
	"github.com/spf13/cobra"
)

var serverCmd = &cobra.Command{
	Use:   "server",
	Short: "Start the Go-Predicato HTTP server",
	Long: `Start the Go-Predicato HTTP server to provide REST API access to the knowledge graph.

The server provides endpoints for:
- Ingesting data (messages, entities)
- Searching the knowledge graph
- Retrieving episodes and memory
- Health checks

Configuration can be provided through config files, environment variables, or command-line flags.`,
	RunE: runServer,
}

var (
	serverHost string
	serverPort int
	serverMode string
)

func init() {
	rootCmd.AddCommand(serverCmd)

	// Server-specific flags
	serverCmd.Flags().StringVar(&serverHost, "host", "localhost", "Server host")
	serverCmd.Flags().IntVar(&serverPort, "port", 19898, "Server port")
	serverCmd.Flags().StringVar(&serverMode, "mode", "debug", "Server mode (debug, release, test)")

	// Database flags
	serverCmd.Flags().String("db-driver", "ladybug", "Database driver (ladybug, neo4j, falkordb)")
	serverCmd.Flags().String("db-uri", "./ladybug_db", "Database URI/path")
	serverCmd.Flags().String("db-username", "", "Database username (not used for ladybug)")
	serverCmd.Flags().String("db-password", "", "Database password (not used for ladybug)")
	serverCmd.Flags().String("db-database", "", "Database name (not used for ladybug)")

	// NLP flags
	serverCmd.Flags().String("nlp-provider", "openai", "NLP provider")
	serverCmd.Flags().String("nlp-model", "gpt-4", "NLP model")
	serverCmd.Flags().String("nlp-api-key", "", "NLP API key")
	serverCmd.Flags().String("nlp-base-url", "", "NLP base URL")
	serverCmd.Flags().Float32("nlp-temperature", 0.1, "NLP temperature")
	serverCmd.Flags().Int("nlp-max-tokens", 2048, "NLP max tokens")

	// Embedding flags
	serverCmd.Flags().String("embedding-provider", "openai", "Embedding provider")
	serverCmd.Flags().String("embedding-model", "text-embedding-3-small", "Embedding model")
	serverCmd.Flags().String("embedding-api-key", "", "Embedding API key")
	serverCmd.Flags().String("embedding-base-url", "", "Embedding base URL")

	// Telemetry flags
	serverCmd.Flags().String("telemetry-parquet-path", "", "Path to directory for telemetry (errors and token usage)")
}

func runServer(cmd *cobra.Command, args []string) error {
	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	// Override config with command-line flags
	overrideConfigWithFlags(cmd, cfg)

	// Validate configuration
	if err := validateServerConfig(cfg); err != nil {
		return fmt.Errorf("invalid configuration: %w", err)
	}

	// Report the listen address early, before heavy initialization
	fmt.Printf("Predicato server will listen on http://%s:%d\n", cfg.Server.Host, cfg.Server.Port)

	// Initialize Predicato
	fmt.Println("Initializing Predicato...")
	predicatoInstance, err := initializePredicato(cfg)
	if err != nil {
		return fmt.Errorf("failed to initialize Predicato: %w", err)
	}

	// Create and setup server
	srv := server.New(cfg, predicatoInstance)
	srv.Setup()

	// Setup graceful shutdown
	// ctx, cancel := context.WithCancel(context.Background())
	// defer cancel()

	// Handle signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Start server in a goroutine
	serverErrChan := make(chan error, 1)
	go func() {
		if err := srv.Start(); err != nil {
			serverErrChan <- err
		}
	}()

	// Wait for shutdown signal or server error
	select {
	case err := <-serverErrChan:
		return fmt.Errorf("server error: %w", err)
	case sig := <-sigChan:
		fmt.Printf("\nReceived signal: %v\n", sig)

		// Create shutdown context with timeout
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer shutdownCancel()

		// Shutdown server
		if err := srv.Stop(shutdownCtx); err != nil {
			return fmt.Errorf("server shutdown error: %w", err)
		}

		fmt.Println("Server stopped gracefully")
		return nil
	}
}

func overrideConfigWithFlags(cmd *cobra.Command, cfg *config.Config) {
	// Server flags
	if cmd.Flags().Changed("host") {
		cfg.Server.Host = serverHost
	}
	if cmd.Flags().Changed("port") {
		cfg.Server.Port = serverPort
	}
	if cmd.Flags().Changed("mode") {
		cfg.Server.Mode = serverMode
	}

	// Database flags
	if cmd.Flags().Changed("db-driver") {
		cfg.Database.Driver, _ = cmd.Flags().GetString("db-driver")
	}
	if cmd.Flags().Changed("db-uri") {
		cfg.Database.URI, _ = cmd.Flags().GetString("db-uri")
	}
	if cmd.Flags().Changed("db-username") {
		cfg.Database.Username, _ = cmd.Flags().GetString("db-username")
	}
	if cmd.Flags().Changed("db-password") {
		cfg.Database.Password, _ = cmd.Flags().GetString("db-password")
	}
	if cmd.Flags().Changed("db-database") {
		cfg.Database.Database, _ = cmd.Flags().GetString("db-database")
	}

	// NLP flags
	if cmd.Flags().Changed("nlp-provider") {
		m := cfg.NLP.Models["default"]
		m.Provider, _ = cmd.Flags().GetString("nlp-provider")
		cfg.NLP.Models["default"] = m
	}
	if cmd.Flags().Changed("nlp-model") {
		m := cfg.NLP.Models["default"]
		m.Model, _ = cmd.Flags().GetString("nlp-model")
		cfg.NLP.Models["default"] = m
	}
	if cmd.Flags().Changed("nlp-api-key") {
		m := cfg.NLP.Models["default"]
		m.APIKey, _ = cmd.Flags().GetString("nlp-api-key")
		cfg.NLP.Models["default"] = m
	}
	if cmd.Flags().Changed("nlp-base-url") {
		m := cfg.NLP.Models["default"]
		m.BaseURL, _ = cmd.Flags().GetString("nlp-base-url")
		cfg.NLP.Models["default"] = m
	}
	if cmd.Flags().Changed("nlp-temperature") {
		m := cfg.NLP.Models["default"]
		m.Temperature, _ = cmd.Flags().GetFloat32("nlp-temperature")
		cfg.NLP.Models["default"] = m
	}
	if cmd.Flags().Changed("nlp-max-tokens") {
		m := cfg.NLP.Models["default"]
		m.MaxTokens, _ = cmd.Flags().GetInt("nlp-max-tokens")
		cfg.NLP.Models["default"] = m
	}

	// Embedding flags
	if cmd.Flags().Changed("embedding-provider") {
		cfg.Embedding.Provider, _ = cmd.Flags().GetString("embedding-provider")
	}
	if cmd.Flags().Changed("embedding-model") {
		cfg.Embedding.Model, _ = cmd.Flags().GetString("embedding-model")
	}
	if cmd.Flags().Changed("embedding-api-key") {
		cfg.Embedding.APIKey, _ = cmd.Flags().GetString("embedding-api-key")
	}
	if cmd.Flags().Changed("embedding-base-url") {
		cfg.Embedding.BaseURL, _ = cmd.Flags().GetString("embedding-base-url")
	}

	// Telemetry flags
	if cmd.Flags().Changed("telemetry-parquet-path") {
		cfg.Telemetry.ParquetPath, _ = cmd.Flags().GetString("telemetry-parquet-path")
	}
}

func validateServerConfig(cfg *config.Config) error {
	if cfg.Server.Port <= 0 || cfg.Server.Port > 65535 {
		return fmt.Errorf("invalid port: %d", cfg.Server.Port)
	}

	if cfg.Database.URI == "" {
		return fmt.Errorf("database URI is required")
	}
	return nil
}

func initializePredicato(cfg *config.Config) (predicato.Predicato, error) {
	// Initialize database driver
	var graphDriver driver.GraphDriver
	var err error
	logger := slog.New(predicatoLogger.NewColorHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	switch cfg.Database.Driver {
	case "ladybug":
		fmt.Printf("LadybugDB path: %s\n", cfg.Database.URI)
		graphDriver, err = driver.NewLadybugDriver(cfg.Database.URI, 16)
		if err != nil {
			return nil, fmt.Errorf("failed to create ladybug driver: %w", err)
		}

	case "falkordb":
		// FalkorDB support would be implemented here
		return nil, fmt.Errorf("FalkorDB driver not yet implemented")
	default:
		return nil, fmt.Errorf("unsupported database driver: %s", cfg.Database.Driver)
	}

	// Initialize NLP client
	var nlProcessor nlp.Client
	defaultModel := cfg.NLP.Models["default"]
	if defaultModel.APIKey != "" {
		switch defaultModel.Provider {
		case "openai":
			nlpConfig := nlp.Config{
				Model:       defaultModel.Model,
				Temperature: &defaultModel.Temperature,
				BaseURL:     defaultModel.BaseURL,
			}
			baseNLPClient, err := nlp.NewOpenAIClient(defaultModel.APIKey, nlpConfig)
			if err != nil {
				return nil, fmt.Errorf("failed to create NLP client: %w", err)
			}
			// Wrap with retry client for automatic retry on errors
			retryClient, err := nlp.NewRetryClient(baseNLPClient, nlp.DefaultRetryConfig())
			if err != nil {
				return nil, fmt.Errorf("failed to create retry client: %w", err)
			}

			// Telemetry using Parquet
			trackingPath := cfg.Telemetry.ParquetPath
			if trackingPath == "" {
				homeDir, err := os.UserHomeDir()
				if err != nil {
					return nil, fmt.Errorf("failed to get user home directory: %w", err)
				}
				trackingPath = fmt.Sprintf("%s/.predicato/telemetry", homeDir)
			}

			// Ensure directory exists
			if err := os.MkdirAll(trackingPath, 0755); err != nil {
				return nil, fmt.Errorf("failed to create telemetry directory: %w", err)
			}

			// Initialize Token Tracker
			tracker, err := nlp.NewTokenTracker(trackingPath)
			if err != nil {
				fmt.Printf("Warning: Failed to initialize token tracker: %v\n", err)
				nlProcessor = retryClient
			} else {
				nlProcessor = nlp.NewTokenTrackingClient(retryClient, tracker)
				fmt.Printf("Token tracking enabled at: %s\n", trackingPath)
			}

			// Initialize Error Tracking Logger
			colorHandler := predicatoLogger.NewColorHandler(os.Stderr, &slog.HandlerOptions{
				Level: slog.LevelInfo,
			})

			parquetHandler, err := telemetry.NewParquetHandler(colorHandler, trackingPath)
			if err != nil {
				fmt.Printf("Warning: Failed to initialize error tracking: %v\n", err)
			} else {
				// Update the global logger to use our new handler
				logger = slog.New(parquetHandler)
				fmt.Printf("Error tracking enabled\n")
			}
		default:
			return nil, fmt.Errorf("unsupported NLP provider: %s", defaultModel.Provider)
		}
	} else {
		// Default to internal RustBert NLP client (no external API required)
		fmt.Println("Initializing internal RustBert NLP service...")
		rustbertClient := rustbert.NewClient(rustbert.Config{})
		if err := rustbertClient.LoadTextGenerationModel(); err != nil {
			fmt.Printf("Warning: Failed to load RustBert text generation model: %v\n", err)
			fmt.Println("Continuing without NLP service - some features will be unavailable")
		} else {
			nlProcessor = rustbert.NewLLMAdapter(rustbertClient, "text_generation")
			fmt.Println("RustBert NLP service initialized (internal, no API key required)")
		}
	}

	// Initialize embedder client
	var embedderClient embedder.Client
	if cfg.Embedding.APIKey != "" {
		switch cfg.Embedding.Provider {
		case "openai":
			embedderConfig := embedder.Config{
				Model:   cfg.Embedding.Model,
				BaseURL: cfg.Embedding.BaseURL,
			}
			embedderClient = embedder.NewOpenAIEmbedder(cfg.Embedding.APIKey, embedderConfig)
		default:
			return nil, fmt.Errorf("unsupported embedding provider: %s", cfg.Embedding.Provider)
		}
	} else {
		// Default to internal EmbedEverything embedder (no external API required)
		fmt.Println("Initializing internal EmbedEverything embedder service...")
		embedderConfig := &embedder.EmbedEverythingConfig{
			Config: &embedder.Config{
				Model:      "qwen/qwen3-embedding-0.6b",
				Dimensions: 1024,
				BatchSize:  32,
			},
		}
		internalEmbedder, err := embedder.NewEmbedEverythingClient(embedderConfig)
		if err != nil {
			fmt.Printf("Warning: Failed to initialize EmbedEverything embedder: %v\n", err)
			fmt.Println("Continuing without embedder - semantic search will be unavailable")
		} else {
			embedderClient = internalEmbedder
			cfg.Embedding.Provider = "embedeverything"
			cfg.Embedding.Model = "qwen/qwen3-embedding-0.6b"
			fmt.Println("EmbedEverything embedder initialized (internal, no API key required)")
		}
	}

	// Initialize FactStore
	// Priority: 1. PostgreSQL connection string (VectorChord), 2. Embedded Dolt
	var predicatoConfig *predicato.Config

	if cfg.FactStore.ConnectionString != "" {
		// External PostgreSQL with VectorChord
		embDim := cfg.FactStore.EmbeddingDimensions
		if embDim <= 0 {
			embDim = 1024 // Default for qwen3-embedding
		}
		factsDBConfig := &factstore.FactStoreConfig{
			Type:                factstore.FactStoreTypePostgres,
			ConnectionString:    cfg.FactStore.ConnectionString,
			EmbeddingDimensions: embDim,
		}
		fmt.Printf("Using PostgreSQL factstore (VectorChord): %s\n", cfg.FactStore.ConnectionString)

		predicatoConfig = &predicato.Config{
			GroupID:         "default",
			TimeZone:        time.UTC,
			FactStoreConfig: factsDBConfig,
		}
	} else {
		// Default to embedded Dolt
		dataPath := cfg.FactStore.DataPath
		if dataPath == "" {
			dataPath, _ = factstore.DefaultDataPath()
		}
		if err := os.MkdirAll(dataPath, 0755); err != nil {
			return nil, fmt.Errorf("failed to create factstore data directory: %w", err)
		}
		connString := fmt.Sprintf(
			"file://%s?commitname=predicato&commitemail=predicato@localhost&database=facts",
			dataPath,
		)
		fmt.Printf("Using embedded Dolt factstore at: %s\n", dataPath)

		predicatoConfig = &predicato.Config{
			GroupID:    "default",
			TimeZone:   time.UTC,
			FactsDBURL: connString,
		}
	}

	// Create and return Predicato client
	client, err := predicato.NewClient(graphDriver, nlProcessor, embedderClient, predicatoConfig, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create Predicato client: %w", err)
	}

	fmt.Printf("Predicato initialized successfully with driver: %s\n", cfg.Database.Driver)
	if nlProcessor != nil {
		fmt.Printf("NLP provider: %s, model: %s\n", defaultModel.Provider, defaultModel.Model)
	}
	if embedderClient != nil {
		fmt.Printf("Embedding provider: %s, model: %s\n", cfg.Embedding.Provider, cfg.Embedding.Model)
	}

	return client, nil
}
