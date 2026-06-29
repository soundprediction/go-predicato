package predicato

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/soundprediction/predicato"
	candleAdapter "github.com/soundprediction/predicato/pkg/candle"
	"github.com/soundprediction/predicato/pkg/config"
	"github.com/soundprediction/predicato/pkg/crossencoder"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/embedder"
	"github.com/soundprediction/predicato/pkg/factstore"
	"github.com/soundprediction/predicato/pkg/gliner2"
	predicatoLogger "github.com/soundprediction/predicato/pkg/logger"
	"github.com/soundprediction/predicato/pkg/nlp"
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
	useGLiNER2 bool
)

func init() {
	rootCmd.AddCommand(serverCmd)

	// Server-specific flags
	serverCmd.Flags().StringVar(&serverHost, "host", "localhost", "Server host")
	serverCmd.Flags().IntVar(&serverPort, "port", 19898, "Server port")
	serverCmd.Flags().StringVar(&serverMode, "mode", "debug", "Server mode (debug, release, test)")

	// Database flags
	defaultDBPath := "./ladybug_db"
	if home, err := os.UserHomeDir(); err == nil {
		defaultDBPath = filepath.Join(home, ".predicato", "ladybug_db")
	}
	serverCmd.Flags().String("db-driver", "ladybug", "Database driver (ladybug, neo4j, falkordb)")
	serverCmd.Flags().String("db-uri", defaultDBPath, "Database URI/path")
	serverCmd.Flags().String("db-username", "", "Database username (not used for ladybug)")
	serverCmd.Flags().String("db-password", "", "Database password (not used for ladybug)")
	serverCmd.Flags().String("db-database", "", "Database name (not used for ladybug)")

	// GLiNER2 flags
	serverCmd.Flags().BoolVar(&useGLiNER2, "gliner2", false, "Use GLiNER2 as the NLP provider (requires local GLiNER2 service)")
	serverCmd.Flags().String("gliner2-endpoint", "http://localhost:11435", "GLiNER2 service endpoint")

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

	// Reranker flags
	serverCmd.Flags().String("reranker-provider", "reranker", "Cross-encoder reranker provider (reranker, local, mock)")
	serverCmd.Flags().String("reranker-model", "", "Cross-encoder reranker model")
	serverCmd.Flags().String("reranker-api-key", "", "Cross-encoder reranker API key")
	serverCmd.Flags().String("reranker-base-url", "", "Cross-encoder reranker base URL")

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
	predicatoInstance, embedderClient, nlpClient, err := initializePredicato(cmd, cfg)
	if err != nil {
		return fmt.Errorf("failed to initialize Predicato: %w", err)
	}

	// Create and setup server
	srv := server.New(cfg, predicatoInstance, embedderClient)
	if nlpClient != nil {
		srv.SetNLPClient(nlpClient)
	}
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
	} else if cfg.Database.URI == "./ladybug_db" {
		// If config still has the hardcoded relative default (e.g. from config.Load defaults),
		// fallback to the flag's default which is smarter (uses home dir).
		// We only do this if the value equals the old default, preserving explicit config file values.
		val, _ := cmd.Flags().GetString("db-uri")
		if val != "" {
			cfg.Database.URI = val
		}
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

	// Reranker flags
	if cmd.Flags().Changed("reranker-provider") {
		cfg.Reranker.Provider, _ = cmd.Flags().GetString("reranker-provider")
	}
	if cmd.Flags().Changed("reranker-model") {
		cfg.Reranker.Model, _ = cmd.Flags().GetString("reranker-model")
	}
	if cmd.Flags().Changed("reranker-api-key") {
		cfg.Reranker.APIKey, _ = cmd.Flags().GetString("reranker-api-key")
	}
	if cmd.Flags().Changed("reranker-base-url") {
		cfg.Reranker.BaseURL, _ = cmd.Flags().GetString("reranker-base-url")
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

func initializePredicato(cmd *cobra.Command, cfg *config.Config) (predicato.Predicato, embedder.Client, nlp.Client, error) {
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
			return nil, nil, nil, fmt.Errorf("failed to create ladybug driver: %w", err)
		}

	case "cozo":
		fmt.Printf("CozoDB path: %s\n", cfg.Database.URI)
		embDim := cfg.FactStore.EmbeddingDimensions
		if embDim == 0 {
			embDim = 1024
		}
		graphDriver, err = driver.NewCozoDriver(cfg.Database.URI, embDim)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to create cozo driver: %w", err)
		}

	case "duckpgq":
		fmt.Printf("DuckPGQ path: %s\n", cfg.Database.URI)
		embDim := cfg.FactStore.EmbeddingDimensions
		if embDim == 0 {
			embDim = 1024
		}
		graphDriver, err = driver.NewDuckPGQDriver(cfg.Database.URI, embDim)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to create duckpgq driver: %w", err)
		}

	case "falkordb":
		return nil, nil, nil, fmt.Errorf("FalkorDB driver not yet implemented")
	default:
		return nil, nil, nil, fmt.Errorf("unsupported database driver: %s (supported: ladybug, cozo, duckpgq)", cfg.Database.Driver)
	}

	// Initialize NLP client
	var nlProcessor nlp.Client

	if useGLiNER2 {
		endpoint, _ := cmd.Flags().GetString("gliner2-endpoint")
		glinerClient, err := createGLiNER2NLPClient(endpoint)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to create GLiNER2 client: %w", err)
		}

		nlProcessor = glinerClient
		fmt.Printf("Using GLiNER2 NLP provider at: %s\n", endpoint)

		// Update config to reflect GLiNER2 usage so logging and other components are aware
		defaultModel := cfg.NLP.Models["default"]
		defaultModel.Provider = "gliner2"
		defaultModel.Model = "gliner2-multi-v1" // or prompt for model?
		cfg.NLP.Models["default"] = defaultModel
	}

	defaultModel := cfg.NLP.Models["default"]
	if nlProcessor == nil && defaultModel.APIKey != "" {
		switch defaultModel.Provider {
		case "openai":
			nlpConfig := nlp.Config{
				Model:       defaultModel.Model,
				Temperature: &defaultModel.Temperature,
				BaseURL:     defaultModel.BaseURL,
			}
			baseNLPClient, err := nlp.NewOpenAIClient(defaultModel.APIKey, nlpConfig)
			if err != nil {
				return nil, nil, nil, fmt.Errorf("failed to create NLP client: %w", err)
			}
			// Wrap with retry client for automatic retry on errors
			retryClient, err := nlp.NewRetryClient(baseNLPClient, nlp.DefaultRetryConfig())
			if err != nil {
				return nil, nil, nil, fmt.Errorf("failed to create retry client: %w", err)
			}

			// Telemetry using Parquet
			trackingPath := cfg.Telemetry.ParquetPath
			if trackingPath == "" {
				homeDir, err := os.UserHomeDir()
				if err != nil {
					return nil, nil, nil, fmt.Errorf("failed to get user home directory: %w", err)
				}
				trackingPath = fmt.Sprintf("%s/.predicato/telemetry", homeDir)
			}

			// Ensure directory exists
			if err := os.MkdirAll(trackingPath, 0755); err != nil {
				return nil, nil, nil, fmt.Errorf("failed to create telemetry directory: %w", err)
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
			return nil, nil, nil, fmt.Errorf("unsupported NLP provider: %s", defaultModel.Provider)
		}
	} else if nlProcessor == nil {
		// Default to internal Candle NLP client (no external API required)
		fmt.Println("Initializing internal Candle NLP service...")
		candleClient := candleAdapter.NewClient(candleAdapter.CandleNLPConfig{
			TextGenModelID: "HuggingFaceTB/SmolLM2-360M-Instruct",
		})
		nlProcessor = candleAdapter.NewLLMAdapter(candleClient, "text_generation")
		fmt.Println("Candle NLP service initialized (internal, no API key required)")
	}

	// Initialize embedder client
	var embedderClient embedder.Client
	if cfg.Embedding.APIKey != "" || cfg.Embedding.BaseURL != "" {
		switch cfg.Embedding.Provider {
		case "openai":
			embedderConfig := embedder.Config{
				Model:   cfg.Embedding.Model,
				BaseURL: cfg.Embedding.BaseURL,
			}
			embedderClient = embedder.NewOpenAIEmbedder(cfg.Embedding.APIKey, embedderConfig)
			if fallback, err := createInternalEmbedder(); err != nil {
				fmt.Printf("Warning: Failed to initialize fallback Candle embedder: %v\n", err)
			} else {
				embedderClient = embedder.NewFallbackClient(embedderClient, fallback)
				fmt.Println("Embedding fallback enabled: internal Candle embedder")
			}
		default:
			return nil, nil, nil, fmt.Errorf("unsupported embedding provider: %s", cfg.Embedding.Provider)
		}
	} else {
		// Default to internal Candle embedder (no external API required)
		fmt.Println("Initializing internal Candle embedder service...")
		internalEmbedder, err := createInternalEmbedder()
		if err != nil {
			fmt.Printf("Warning: Failed to initialize Candle embedder: %v\n", err)
			fmt.Println("Continuing without embedder - semantic search will be unavailable")
		} else {
			embedderClient = internalEmbedder
			cfg.Embedding.Provider = "candle"
			cfg.Embedding.Model = "qwen/qwen3-embedding-0.6b"
			fmt.Println("Candle embedder initialized (internal, no API key required)")
		}
	}

	// Initialize FactStore (PostgreSQL with VectorChord required)
	var predicatoConfig *predicato.Config

	if cfg.FactStore.ConnectionString != "" {
		embDim := cfg.FactStore.EmbeddingDimensions
		if embDim <= 0 {
			embDim = 1024 // Default for qwen3-embedding
		}
		dbConfig := &factstore.DbConfig{
			Type:                factstore.FactStoreTypePostgres,
			ConnectionString:    cfg.FactStore.ConnectionString,
			EmbeddingDimensions: embDim,
		}
		fmt.Printf("Using PostgreSQL factstore (VectorChord): %s\n", cfg.FactStore.ConnectionString)

		predicatoConfig = &predicato.Config{
			GroupID:  "default",
			TimeZone: time.UTC,
			DbConfig: dbConfig,
		}
	} else {
		fmt.Println("No factstore connection string configured, running without factstore")
		predicatoConfig = &predicato.Config{
			GroupID:  "default",
			TimeZone: time.UTC,
		}
	}

	// Create and return Predicato client
	client, err := predicato.NewClient(graphDriver, nlProcessor, embedderClient, predicatoConfig, logger)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to create Predicato client: %w", err)
	}
	if ce, err := createRerankerClient(cfg.Reranker, embedderClient); err != nil {
		return nil, nil, nil, err
	} else if ce != nil {
		client.SetCrossEncoder(ce)
		fmt.Printf("Cross-encoder reranker enabled: %s (%s)\n", cfg.Reranker.Model, cfg.Reranker.BaseURL)
	}

	fmt.Printf("Predicato initialized successfully with driver: %s\n", cfg.Database.Driver)
	if nlProcessor != nil {
		fmt.Printf("NLP provider: %s, model: %s\n", defaultModel.Provider, defaultModel.Model)
	}
	if embedderClient != nil {
		fmt.Printf("Embedding provider: %s, model: %s\n", cfg.Embedding.Provider, cfg.Embedding.Model)
	}

	return client, embedderClient, nlProcessor, nil
}

func createInternalEmbedder() (embedder.Client, error) {
	return candleAdapter.NewCandleEmbedderClient(&candleAdapter.CandleEmbedderConfig{
		Model:      "qwen/qwen3-embedding-0.6b",
		Dimensions: 1024,
		Normalize:  true,
	})
}

func createRerankerClient(cfg config.RerankerConfig, fallbackEmbedder embedder.Client) (crossencoder.Client, error) {
	if cfg.BaseURL == "" && cfg.Provider == "" && cfg.Model == "" {
		return nil, nil
	}

	provider := cfg.Provider
	if provider == "" {
		provider = "reranker"
	}

	switch provider {
	case "reranker", "http", "jina":
		if cfg.BaseURL == "" {
			return nil, fmt.Errorf("reranker.base_url is required for provider %q", provider)
		}
		primary := crossencoder.NewRerankerClient(crossencoder.RerankerConfig{
			BaseURL: cfg.BaseURL,
			APIKey:  cfg.APIKey,
			Config:  crossencoder.Config{Model: cfg.Model},
		})
		if fallbackEmbedder == nil {
			return primary, nil
		}
		fallback := crossencoder.NewEmbeddingRerankerClient(fallbackEmbedder, crossencoder.EmbeddingConfig{
			Config:         crossencoder.Config{Model: "embedding-fallback"},
			BorrowEmbedder: true,
		})
		return crossencoder.NewFallbackClient(primary, fallback), nil
	case "local":
		return crossencoder.NewLocalRerankerClient(crossencoder.Config{Model: cfg.Model}), nil
	case "mock":
		return crossencoder.NewMockRerankerClient(crossencoder.Config{Model: cfg.Model}), nil
	default:
		return nil, fmt.Errorf("unsupported reranker provider: %s", provider)
	}
}

func createGLiNER2NLPClient(endpoint string) (nlp.Client, error) {
	if endpoint == "" {
		endpoint = "http://localhost:11435"
	}

	primary, err := newGLiNER2NLPClient(endpoint)
	if err != nil {
		return nil, err
	}

	if isLocalEndpoint(endpoint) {
		if err := ensureGLiNER2Server(endpoint); err != nil {
			return nil, fmt.Errorf("failed to ensure local GLiNER2 server: %w", err)
		}
		return primary, nil
	}

	fallbackEndpoint := "http://localhost:11435"
	fmt.Printf("GLiNER2 fallback configured: primary=%s fallback=%s\n", endpoint, fallbackEndpoint)
	return nlp.NewLazyFallbackClient(primary, func() (nlp.Client, error) {
		if err := ensureGLiNER2Server(fallbackEndpoint); err != nil {
			return nil, err
		}
		return newGLiNER2NLPClient(fallbackEndpoint)
	}), nil
}

func newGLiNER2NLPClient(endpoint string) (nlp.Client, error) {
	return gliner2.NewClient(gliner2.Config{
		Provider: gliner2.ProviderLocal,
		Local: &gliner2.LocalConfig{
			Endpoint: endpoint,
		},
	})
}

func isLocalEndpoint(endpoint string) bool {
	u, err := url.Parse(endpoint)
	if err != nil {
		return false
	}
	host := u.Hostname()
	return host == "" || host == "localhost" || host == "127.0.0.1" || host == "0.0.0.0"
}

func ensureGLiNER2Server(endpoint string) error {
	u, err := url.Parse(endpoint)
	if err != nil {
		return err
	}

	// Check if already running
	if isServerHealthy(endpoint) {
		return nil
	}

	// Only auto-start if localhost
	hostname := u.Hostname()
	if hostname != "localhost" && hostname != "127.0.0.1" && hostname != "0.0.0.0" {
		return fmt.Errorf("GLiNER2 server at %s is not available and cannot be auto-started (remote host)", endpoint)
	}

	fmt.Println("GLiNER2 server not ready, attempting to start local Python server...")

	// Check if uv is installed
	uvExec := "uv"
	if _, err := exec.LookPath("uv"); err != nil {
		fmt.Println("Warning: uv not found. Falling back to system python3 (auto-start may fail if dependencies are missing).")
		uvExec = ""
	}

	var cmd *exec.Cmd
	cwd, _ := os.Getwd()
	// Assuming workspace structure: root has predicato/python
	pythonDir := filepath.Join(cwd, "predicato", "python")

	// Double check python dir validity
	if _, err := os.Stat(filepath.Join(pythonDir, "pyproject.toml")); os.IsNotExist(err) {
		// Try submodule structure (if run from workspace root)
		pythonDir = filepath.Join(cwd, "python") // try alternate structure
		if _, err := os.Stat(filepath.Join(pythonDir, "pyproject.toml")); os.IsNotExist(err) {
			// Fallback to original guess if neither work (error will happen later)
			pythonDir = filepath.Join(cwd, "predicato", "python")
		}
	}

	if uvExec != "" {
		// Use uv run
		// We need to run inside the python directory where pyproject.toml and .venv are
		fmt.Printf("Starting GLiNER2 server using uv at %s\n", pythonDir)
		cmd = exec.Command(uvExec, "run", "python", "-m", "predicato.server")
		cmd.Dir = pythonDir
	} else {
		// Fallback to direct python execution (previous logic simplified)
		pythonExec := "python3"
		if _, err := exec.LookPath("python"); err == nil {
			pythonExec = "python"
		}

		// We still need to find paths for PYTHONPATH if not using uv
		glinerLibPath := filepath.Join(cwd, "GLiNER2")
		if _, err := os.Stat(glinerLibPath); os.IsNotExist(err) {
			glinerLibPath = filepath.Join(filepath.Dir(cwd), "GLiNER2")
		}

		cmd = exec.Command(pythonExec, "-m", "predicato.server")

		// Set PYTHONPATH
		env := os.Environ()
		newPythonPath := fmt.Sprintf("PYTHONPATH=%s:%s", pythonDir, glinerLibPath)
		if existingPP := os.Getenv("PYTHONPATH"); existingPP != "" {
			newPythonPath = fmt.Sprintf("%s:%s", newPythonPath, existingPP)
		}
		cmd.Env = append(env, newPythonPath)
	}

	// Pass port configuration
	port := u.Port()
	if port == "" {
		port = "11435"
	}
	// Add environment variables (cmd.Env is nil for uv branch normally, exec.Command uses os.Environ() by default for nil)
	if cmd.Env == nil {
		cmd.Env = os.Environ()
	}
	cmd.Env = append(cmd.Env, fmt.Sprintf("PORT=%s", port))
	cmd.Env = append(cmd.Env, fmt.Sprintf("HOST=%s", hostname))

	// Redirect output to stdout/stderr
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start server process: %w", err)
	}

	fmt.Printf("Started GLiNER2 server (PID: %d). Waiting for health check...\n", cmd.Process.Pid)

	// Wait for up to 60 seconds (model loading might take time)
	for i := 0; i < 60; i++ {
		if isServerHealthy(endpoint) {
			fmt.Println("GLiNER2 server is ready!")
			return nil
		}
		time.Sleep(1 * time.Second)
	}

	// Cleanup if failed
	cmd.Process.Kill()
	return fmt.Errorf("timed out waiting for GLiNER2 server to start")
}

func isServerHealthy(endpoint string) bool {
	// Simple HTTP health check
	healthURL := fmt.Sprintf("%s/health", endpoint)
	client := http.Client{Timeout: 1 * time.Second}
	resp, err := client.Get(healthURL)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}
