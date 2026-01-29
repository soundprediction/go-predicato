package predicato

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/soundprediction/predicato"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/embedder"
	predicatoLogger "github.com/soundprediction/predicato/pkg/logger"
	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/telemetry"
	"github.com/soundprediction/predicato/pkg/types"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

// Default configuration values for MCP server
const (
	DefaultMCPLLMModel       = "gpt-4o-mini"
	DefaultMCPSmallModel     = "gpt-4o-mini"
	DefaultMCPEmbedderModel  = "text-embedding-3-small"
	DefaultMCPSemaphoreLimit = 10
)

var mcpCmd = &cobra.Command{
	Use:   "mcp",
	Short: "Start the Model Context Protocol (MCP) server",
	Long: `Start the Model Context Protocol (MCP) server to provide MCP tool access to the knowledge graph.

The MCP server provides tools for:
- Adding episodes/memories to the knowledge graph
- Searching nodes and facts in the graph
- Managing entities and episodes
- Clearing graph data

The server can communicate over stdio or HTTP/SSE transport protocols and is designed
to work with MCP clients like Claude Desktop or other compatible applications.`,
	RunE: runMCPServer,
}

var (
	mcpGroupID           string
	mcpTransport         string
	mcpHost              string
	mcpPort              int
	mcpModel             string
	mcpSmallModel        string
	mcpTemperature       float64
	mcpUseCustomEntities bool
	mcpDestroyGraph      bool
	mcpSemaphoreLimit    int
)

func init() {
	rootCmd.AddCommand(mcpCmd)

	// Configure viper to automatically check for environment variables
	viper.AutomaticEnv()

	// Set up specific environment variable bindings to maintain compatibility
	// with existing environment variable names
	_ = viper.BindEnv("nlp.api_key", "OPENAI_API_KEY")
	_ = viper.BindEnv("nlp.base_url", "LLM_BASE_URL")
	_ = viper.BindEnv("embedder.api_key", "EMBEDDING_API_KEY", "OPENAI_API_KEY") // Fallback to OpenAI key
	_ = viper.BindEnv("embedder.base_url", "EMBEDDING_BASE_URL")
	_ = viper.BindEnv("embedder.model", "EMBEDDER_MODEL_NAME")
	_ = viper.BindEnv("database.uri", "NEO4J_URI")
	_ = viper.BindEnv("database.username", "NEO4J_USER")
	_ = viper.BindEnv("database.password", "NEO4J_PASSWORD")
	_ = viper.BindEnv("database.database", "NEO4J_DATABASE")
	_ = viper.BindEnv("mcp.group_id", "GROUP_ID")
	_ = viper.BindEnv("mcp.transport", "MCP_TRANSPORT")
	_ = viper.BindEnv("mcp.host", "MCP_HOST")
	_ = viper.BindEnv("mcp.port", "MCP_PORT")
	_ = viper.BindEnv("mcp.model", "MODEL_NAME")
	_ = viper.BindEnv("mcp.small_model", "SMALL_MODEL_NAME")
	_ = viper.BindEnv("mcp.temperature", "LLM_TEMPERATURE")
	_ = viper.BindEnv("mcp.use_custom_entities", "USE_CUSTOM_ENTITIES")
	_ = viper.BindEnv("mcp.destroy_graph", "DESTROY_GRAPH")
	_ = viper.BindEnv("mcp.semaphore_limit", "SEMAPHORE_LIMIT")

	// MCP Server specific flags
	mcpCmd.Flags().StringVar(&mcpGroupID, "group-id", "default", "Namespace for the graph")
	mcpCmd.Flags().StringVar(&mcpTransport, "transport", "stdio", "Transport to use (stdio or sse)")
	mcpCmd.Flags().StringVar(&mcpHost, "host", "localhost", "Host to bind the MCP server to")
	mcpCmd.Flags().IntVar(&mcpPort, "port", 3000, "Port to bind the MCP server to")
	mcpCmd.Flags().StringVar(&mcpModel, "model", DefaultMCPLLMModel, "LLM model name")
	mcpCmd.Flags().StringVar(&mcpSmallModel, "small-model", DefaultMCPSmallModel, "Small LLM model name")
	mcpCmd.Flags().Float64Var(&mcpTemperature, "temperature", 0.0, "Temperature setting for the LLM (0.0-2.0)")
	mcpCmd.Flags().BoolVar(&mcpUseCustomEntities, "use-custom-entities", false, "Enable entity extraction using predefined entity types")
	mcpCmd.Flags().BoolVar(&mcpDestroyGraph, "destroy-graph", false, "Destroy all Predicato graphs on startup")
	mcpCmd.Flags().IntVar(&mcpSemaphoreLimit, "semaphore-limit", DefaultMCPSemaphoreLimit, "Concurrency limit for operations")

	// Database flags
	mcpCmd.Flags().String("db-driver", "ladybug", "Database driver (ladybug, neo4j, falkordb)")
	mcpCmd.Flags().String("db-uri", "./ladybug_db", "Database URI/path")
	mcpCmd.Flags().String("db-username", "", "Database username (not used for ladybug)")
	mcpCmd.Flags().String("db-password", "", "Database password (not used for ladybug)")
	mcpCmd.Flags().String("db-database", "", "Database name (not used for ladybug)")

	// LLM flags
	mcpCmd.Flags().String("llm-api-key", "", "OpenAI API key")
	mcpCmd.Flags().String("llm-base-url", "", "LLM base URL (for OpenAI-compatible services)")

	// Embedding flags
	mcpCmd.Flags().String("embedder-model", DefaultMCPEmbedderModel, "Embedding model name")
	mcpCmd.Flags().String("embedding-api-key", "", "Embedding API key")
	mcpCmd.Flags().String("embedding-base-url", "", "Embedding base URL")

	// Telemetry flags
	mcpCmd.Flags().String("telemetry-parquet-path", "", "Path to directory for telemetry (errors and token usage)")

	// Bind flags to viper for configuration
	_ = viper.BindPFlag("mcp.group_id", mcpCmd.Flags().Lookup("group-id"))
	_ = viper.BindPFlag("mcp.transport", mcpCmd.Flags().Lookup("transport"))
	_ = viper.BindPFlag("mcp.host", mcpCmd.Flags().Lookup("host"))
	_ = viper.BindPFlag("mcp.port", mcpCmd.Flags().Lookup("port"))
	_ = viper.BindPFlag("mcp.model", mcpCmd.Flags().Lookup("model"))
	_ = viper.BindPFlag("mcp.small_model", mcpCmd.Flags().Lookup("small-model"))
	_ = viper.BindPFlag("mcp.temperature", mcpCmd.Flags().Lookup("temperature"))
	_ = viper.BindPFlag("mcp.use_custom_entities", mcpCmd.Flags().Lookup("use-custom-entities"))
	_ = viper.BindPFlag("mcp.destroy_graph", mcpCmd.Flags().Lookup("destroy-graph"))
	_ = viper.BindPFlag("mcp.semaphore_limit", mcpCmd.Flags().Lookup("semaphore-limit"))

	// Database configuration
	_ = viper.BindPFlag("database.uri", mcpCmd.Flags().Lookup("db-uri"))
	_ = viper.BindPFlag("database.username", mcpCmd.Flags().Lookup("db-username"))
	_ = viper.BindPFlag("database.password", mcpCmd.Flags().Lookup("db-password"))
	_ = viper.BindPFlag("database.database", mcpCmd.Flags().Lookup("db-database"))

	// LLM configuration
	_ = viper.BindPFlag("nlp.api_key", mcpCmd.Flags().Lookup("llm-api-key"))
	_ = viper.BindPFlag("nlp.base_url", mcpCmd.Flags().Lookup("llm-base-url"))

	// Embedder configuration
	_ = viper.BindPFlag("embedder.model", mcpCmd.Flags().Lookup("embedder-model"))
	_ = viper.BindPFlag("embedder.api_key", mcpCmd.Flags().Lookup("embedding-api-key"))
	_ = viper.BindPFlag("embedder.base_url", mcpCmd.Flags().Lookup("embedding-base-url"))

	// Telemetry configuration
	_ = viper.BindPFlag("telemetry.parquet_path", mcpCmd.Flags().Lookup("telemetry-parquet-path"))
}

// MCPConfig holds all configuration for the MCP server
type MCPConfig struct {
	// LLM Configuration
	LLMModel       string
	SmallLLMModel  string
	LLMTemperature float64
	OpenAIAPIKey   string
	LLMBaseURL     string

	// Embedder Configuration
	EmbedderModel    string
	EmbeddingAPIKey  string
	EmbeddingBaseURL string

	// Database Configuration
	DatabaseDriver   string
	DatabaseURI      string
	DatabaseUser     string
	DatabasePassword string
	DatabaseName     string

	// MCP Server Configuration
	GroupID           string
	UseCustomEntities bool
	DestroyGraph      bool
	Transport         string
	Host              string
	Port              int

	// Concurrency limits
	SemaphoreLimit int

	// Telemetry Configuration
	TelemetryParquetPath string
}

// MCPServer wraps the Predicato client for MCP operations
type MCPServer struct {
	config *MCPConfig
	client *predicato.Client
	logger *slog.Logger
}

// EntityTypes represents custom entity types for extraction
var EntityTypes = map[string]interface{}{
	"Requirement": struct {
		ProjectName string `json:"project_name" description:"The name of the project to which the requirement belongs."`
		Description string `json:"description" description:"Description of the requirement. Only use information mentioned in the context to write this description."`
	}{},
	"Preference": struct {
		Category    string `json:"category" description:"The category of the preference. (e.g., 'Brands', 'Food', 'Music')"`
		Description string `json:"description" description:"Brief description of the preference. Only use information mentioned in the context to write this description."`
	}{},
	"Procedure": struct {
		Description string `json:"description" description:"Brief description of the procedure. Only use information mentioned in the context to write this description."`
	}{},
}

// MCP Tool request/response types

// AddMemoryRequest represents the parameters for adding memory
type AddMemoryRequest struct {
	Name              string `json:"name"`
	EpisodeBody       string `json:"episode_body"`
	GroupID           string `json:"group_id,omitempty"`
	Source            string `json:"source,omitempty"`
	SourceDescription string `json:"source_description,omitempty"`
	UUID              string `json:"uuid,omitempty"`
}

// SearchRequest represents search parameters
type SearchRequest struct {
	Query string `json:"query"`
	Limit int    `json:"limit,omitempty"`
}

// GetEpisodesRequest represents parameters for retrieving episodes
type GetEpisodesRequest struct {
	GroupID string `json:"group_id,omitempty"`
	LastN   int    `json:"last_n,omitempty"`
}

// ClearGraphRequest represents parameters for clearing the graph
type ClearGraphRequest struct {
	GroupID string `json:"group_id,omitempty"`
	Confirm bool   `json:"confirm,omitempty"`
}

// UUIDRequest represents a simple UUID parameter
type UUIDRequest struct {
	UUID string `json:"uuid"`
}

// MCPToolResponse is a generic response wrapper
type MCPToolResponse struct {
	Success bool        `json:"success"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
}

// MCPTool represents a registered MCP tool
type MCPTool struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Schema      interface{} `json:"inputSchema"`
}

// MCPCapabilities represents the capabilities of the MCP server
type MCPCapabilities struct {
	Tools map[string]MCPTool `json:"tools"`
}

func runMCPServer(cmd *cobra.Command, args []string) error {
	// Create configuration using viper (supports config files, env vars, and flags)
	config := &MCPConfig{
		// MCP Server configuration
		GroupID:           getViperStringWithFallback("mcp.group_id", mcpGroupID),
		Transport:         getViperStringWithFallback("mcp.transport", mcpTransport),
		Host:              getViperStringWithFallback("mcp.host", mcpHost),
		Port:              getViperIntWithFallback("mcp.port", mcpPort),
		LLMModel:          getViperStringWithFallback("mcp.model", mcpModel),
		SmallLLMModel:     getViperStringWithFallback("mcp.small_model", mcpSmallModel),
		LLMTemperature:    getViperFloat64WithFallback("mcp.temperature", mcpTemperature),
		UseCustomEntities: getViperBoolWithFallback("mcp.use_custom_entities", mcpUseCustomEntities),
		DestroyGraph:      getViperBoolWithFallback("mcp.destroy_graph", mcpDestroyGraph),
		SemaphoreLimit:    getViperIntWithFallback("mcp.semaphore_limit", mcpSemaphoreLimit),

		// Database configuration - viper handles env vars automatically
		DatabaseDriver:   getViperStringWithFallback("database.driver", "ladybug"),
		DatabaseURI:      getViperStringWithFallback("database.uri", "./ladybug_db"),
		DatabaseUser:     getViperStringWithFallback("database.username", ""),
		DatabasePassword: getViperStringWithFallback("database.password", ""),
		DatabaseName:     getViperStringWithFallback("database.database", ""),

		// LLM configuration - now optional
		OpenAIAPIKey: viper.GetString("nlp.api_key"), // No fallback - truly optional
		LLMBaseURL:   viper.GetString("nlp.base_url"),

		// Embedder configuration
		EmbedderModel:    getViperStringWithFallback("embedder.model", DefaultMCPEmbedderModel),
		EmbeddingAPIKey:  viper.GetString("embedder.api_key"), // No fallback - truly optional
		EmbeddingBaseURL: viper.GetString("embedder.base_url"),

		// Telemetry configuration
		TelemetryParquetPath: viper.GetString("telemetry.parquet_path"),
	}

	// Use LLM API key for embeddings if embedding API key not provided
	if config.EmbeddingAPIKey == "" {
		config.EmbeddingAPIKey = config.OpenAIAPIKey
	}

	// Validate required configuration
	if err := validateMCPConfig(config); err != nil {
		return fmt.Errorf("invalid MCP configuration: %w", err)
	}

	// Create MCP server
	server, err := NewMCPServer(config)
	if err != nil {
		return fmt.Errorf("failed to create MCP server: %w", err)
	}

	// Initialize server
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := server.Initialize(ctx); err != nil {
		return fmt.Errorf("failed to initialize MCP server: %w", err)
	}

	// Setup graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Run server in a goroutine
	serverErrChan := make(chan error, 1)
	go func() {
		serverErrChan <- server.Run(ctx)
	}()

	// Wait for shutdown signal or server error
	select {
	case err := <-serverErrChan:
		if err != nil && err != context.Canceled {
			return fmt.Errorf("MCP server error: %w", err)
		}
		return nil
	case sig := <-sigChan:
		fmt.Printf("\nReceived signal: %v\n", sig)
		cancel()

		// Give server time to shutdown gracefully
		select {
		case <-time.After(10 * time.Second):
			return fmt.Errorf("server shutdown timeout")
		case <-serverErrChan:
			fmt.Println("MCP server stopped gracefully")
			return nil
		}
	}
}

// NewMCPServer creates a new MCP server instance
func NewMCPServer(config *MCPConfig) (*MCPServer, error) {
	logger := slog.New(predicatoLogger.NewColorHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	// Create database driver
	var graphDriver driver.GraphDriver
	var err error

	switch config.DatabaseDriver {
	case "ladybug":
		graphDriver, err = driver.NewLadybugDriver(config.DatabaseURI, 16)
		if err != nil {
			return nil, fmt.Errorf("failed to create ladybug driver: %w", err)
		}

	default:
		return nil, fmt.Errorf("unsupported database driver: %s", config.DatabaseDriver)
	}

	// Create LLM client - only if we have an API key or base URL
	var nlProcessor nlp.Client
	if config.OpenAIAPIKey != "" || config.LLMBaseURL != "" {
		llmConfig := nlp.Config{
			Model:       config.LLMModel,
			Temperature: &[]float32{float32(config.LLMTemperature)}[0],
			BaseURL:     config.LLMBaseURL,
		}
		// Use empty string as API key if only base URL is provided (for services that don't require auth)
		apiKey := config.OpenAIAPIKey
		if apiKey == "" && config.LLMBaseURL != "" {
			apiKey = "dummy" // Some OpenAI-compatible services require a non-empty key
		}
		baseLLMClient, err := nlp.NewOpenAIClient(apiKey, llmConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create LLM client: %w", err)
		}
		// Wrap with retry client for automatic retry on errors
		retryClient, err := nlp.NewRetryClient(baseLLMClient, nlp.DefaultRetryConfig())
		if err != nil {
			return nil, fmt.Errorf("failed to create retry client: %w", err)
		}

		// Telemetry using Parquet
		trackingPath := config.TelemetryParquetPath
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
			logger.Warn("Failed to initialize token tracker", "error", err)
			nlProcessor = retryClient
		} else {
			nlProcessor = nlp.NewTokenTrackingClient(retryClient, tracker)
			logger.Info("Token tracking enabled", "path", trackingPath)
		}

		// Initialize Error Tracking Logger
		// We wrap the existing logger's handler with our Parquet handler
		parquetHandler, err := telemetry.NewParquetHandler(logger.Handler(), trackingPath)
		if err != nil {
			logger.Warn("Failed to initialize error tracking", "error", err)
		} else {
			// Update the logger to use our new handler
			logger = slog.New(parquetHandler)
			logger.Info("Error tracking enabled")
		}
	} else {
		logger.Warn("No LLM configuration provided - LLM functionality will be disabled")
	}

	// Create embedder client - only if we have an API key or base URL
	var embedderClient embedder.Client
	if config.EmbeddingAPIKey != "" || config.EmbeddingBaseURL != "" {
		embedderConfig := embedder.Config{
			Model:   config.EmbedderModel,
			BaseURL: config.EmbeddingBaseURL,
		}
		// Use empty string as API key if only base URL is provided
		apiKey := config.EmbeddingAPIKey
		if apiKey == "" && config.EmbeddingBaseURL != "" {
			apiKey = "dummy"
		}
		embedderClient = embedder.NewOpenAIEmbedder(apiKey, embedderConfig)
	} else {
		logger.Warn("No embedder configuration provided - embedding functionality will be disabled")
	}

	// Create Predicato client
	predicatoConfig := &predicato.Config{
		GroupID:  config.GroupID,
		TimeZone: time.UTC,
	}

	client, err := predicato.NewClient(graphDriver, nlProcessor, embedderClient, predicatoConfig, logger)
	if err != nil {
		return nil, fmt.Errorf("failed to create Predicato client: %w", err)
	}

	return &MCPServer{
		config: config,
		client: client,
		logger: logger,
	}, nil
}

// Initialize sets up the MCP server and Predicato client
func (s *MCPServer) Initialize(ctx context.Context) error {
	s.logger.Info("Initializing Predicato MCP server...")

	// Verify the client is ready
	if s.client == nil {
		return fmt.Errorf("predicato client not initialized")
	}

	// Clear graph if requested
	if s.config.DestroyGraph {
		s.logger.Warn("Graph destruction requested - clearing all data for group", "group_id", s.config.GroupID)

		err := s.client.ClearGraph(ctx, s.config.GroupID)
		if err != nil {
			s.logger.Error("Failed to clear graph during initialization", "error", err)
			return fmt.Errorf("failed to clear graph: %w", err)
		}

		s.logger.Info("Graph cleared successfully during initialization")
	}

	s.logger.Info("Predicato client initialized successfully")
	s.logger.Info("MCP server configuration",
		"llm_model", s.config.LLMModel,
		"temperature", s.config.LLMTemperature,
		"group_id", s.config.GroupID,
		"transport", s.config.Transport,
		"custom_entities", s.config.UseCustomEntities,
		"semaphore_limit", s.config.SemaphoreLimit,
	)

	return nil
}

// RegisterTools registers all MCP tools
func (s *MCPServer) RegisterTools() error {
	s.logger.Info("Registering MCP tools...")

	// Create capabilities structure
	capabilities := &MCPCapabilities{
		Tools: make(map[string]MCPTool),
	}

	// Register add_memory tool
	capabilities.Tools["add_memory"] = MCPTool{
		Name:        "add_memory",
		Description: "Add an episode to memory. This is the primary way to add information to the graph.",
		Schema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"name": map[string]interface{}{
					"type":        "string",
					"description": "Name or title of the episode",
				},
				"episode_body": map[string]interface{}{
					"type":        "string",
					"description": "The content or body of the episode to be added",
				},
				"group_id": map[string]interface{}{
					"type":        "string",
					"description": "Group ID to associate the episode with (optional)",
				},
				"source": map[string]interface{}{
					"type":        "string",
					"description": "Source of the episode (optional)",
				},
				"source_description": map[string]interface{}{
					"type":        "string",
					"description": "Description of the source (optional)",
				},
				"uuid": map[string]interface{}{
					"type":        "string",
					"description": "Custom UUID for the episode (optional)",
				},
			},
			"required": []string{"name", "episode_body"},
		},
	}

	// Register search_memory_nodes tool
	capabilities.Tools["search_memory_nodes"] = MCPTool{
		Name:        "search_memory_nodes",
		Description: "Search the graph memory for relevant node summaries.",
		Schema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "Search query for finding relevant nodes",
				},
				"limit": map[string]interface{}{
					"type":        "integer",
					"description": "Maximum number of results to return (default: 10)",
					"minimum":     1,
					"maximum":     100,
				},
			},
			"required": []string{"query"},
		},
	}

	// Register search_memory_facts tool
	capabilities.Tools["search_memory_facts"] = MCPTool{
		Name:        "search_memory_facts",
		Description: "Search the graph memory for relevant facts (relationships).",
		Schema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"query": map[string]interface{}{
					"type":        "string",
					"description": "Search query for finding relevant facts",
				},
				"limit": map[string]interface{}{
					"type":        "integer",
					"description": "Maximum number of results to return (default: 10)",
					"minimum":     1,
					"maximum":     100,
				},
			},
			"required": []string{"query"},
		},
	}

	// Register get_episodes tool
	capabilities.Tools["get_episodes"] = MCPTool{
		Name:        "get_episodes",
		Description: "Get the most recent memory episodes for a specific group.",
		Schema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"group_id": map[string]interface{}{
					"type":        "string",
					"description": "Group ID to retrieve episodes from (optional)",
				},
				"last_n": map[string]interface{}{
					"type":        "integer",
					"description": "Number of recent episodes to retrieve (default: 10)",
					"minimum":     1,
					"maximum":     100,
				},
			},
		},
	}

	// Register clear_graph tool
	capabilities.Tools["clear_graph"] = MCPTool{
		Name:        "clear_graph",
		Description: "Clear all data from the graph memory. Requires confirmation.",
		Schema: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"group_id": map[string]interface{}{
					"type":        "string",
					"description": "Group ID to clear (optional, defaults to server group)",
				},
				"confirm": map[string]interface{}{
					"type":        "boolean",
					"description": "Confirmation flag - must be true to proceed with clearing",
				},
			},
			"required": []string{"confirm"},
		},
	}

	toolNames := make([]string, 0, len(capabilities.Tools))
	for name := range capabilities.Tools {
		toolNames = append(toolNames, name)
	}

	s.logger.Info("MCP tools registered successfully", "tools", toolNames, "count", len(capabilities.Tools))
	return nil
}

// Tool implementations

// AddMemoryTool handles adding episodes to memory
func (s *MCPServer) AddMemoryTool(ctx context.Context, input *AddMemoryRequest) (*MCPToolResponse, error) {
	// Validate required fields
	if input.Name == "" {
		return &MCPToolResponse{
			Success: false,
			Error:   "Name is required",
		}, nil
	}
	if input.EpisodeBody == "" {
		return &MCPToolResponse{
			Success: false,
			Error:   "EpisodeBody is required",
		}, nil
	}

	// Set defaults
	if input.Source == "" {
		input.Source = "text"
	}
	if input.GroupID == "" {
		input.GroupID = s.config.GroupID
	}

	// Create episode
	episode := types.Episode{
		ID:        input.UUID, // Will be generated if empty
		Name:      input.Name,
		Content:   input.EpisodeBody,
		Reference: time.Now(),
		CreatedAt: time.Now(),
		GroupID:   input.GroupID,
		Metadata: map[string]interface{}{
			"source":             input.Source,
			"source_description": input.SourceDescription,
		},
	}

	// Add episode using Predicato client
	_, err := s.client.Add(ctx, []types.Episode{episode}, nil)
	if err != nil {
		s.logger.Error("Failed to add episode", "error", err)
		return &MCPToolResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to add episode: %v", err),
		}, nil
	}

	s.logger.Info("Episode added successfully", "name", input.Name, "group_id", input.GroupID)
	return &MCPToolResponse{
		Success: true,
		Message: fmt.Sprintf("Episode '%s' added successfully", input.Name),
	}, nil
}

// SearchMemoryNodesTool handles searching for nodes
func (s *MCPServer) SearchMemoryNodesTool(ctx context.Context, input *SearchRequest) (*MCPToolResponse, error) {
	// Validate required fields
	if input.Query == "" {
		return &MCPToolResponse{
			Success: false,
			Error:   "Query is required",
		}, nil
	}

	// Set defaults
	if input.Limit <= 0 {
		input.Limit = 10
	}

	// Create search configuration
	searchConfig := &types.SearchConfig{
		Limit:              input.Limit,
		CenterNodeDistance: 2,
		MinScore:           0.0,
		IncludeEdges:       false,
		Rerank:             true,
		NodeConfig: &types.NodeSearchConfig{
			SearchMethods: []string{"bm25", "cosine_similarity"},
			Reranker:      "rrf",
			MinScore:      0.0,
		},
	}

	// Perform search
	results, err := s.client.Search(ctx, input.Query, searchConfig)
	if err != nil {
		s.logger.Error("Failed to search nodes", "error", err)
		return &MCPToolResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to search nodes: %v", err),
		}, nil
	}

	if len(results.Nodes) == 0 {
		return &MCPToolResponse{
			Success: true,
			Message: "No relevant nodes found",
			Data:    []interface{}{},
		}, nil
	}

	// Format results
	nodeResults := make([]map[string]interface{}, len(results.Nodes))
	for i, node := range results.Nodes {
		nodeResults[i] = map[string]interface{}{
			"uuid":       node.Uuid,
			"name":       node.Name,
			"summary":    node.Summary,
			"type":       string(node.Type),
			"group_id":   node.GroupID,
			"created_at": node.CreatedAt.Format(time.RFC3339),
			"metadata":   node.Metadata,
		}
	}

	return &MCPToolResponse{
		Success: true,
		Message: "Nodes retrieved successfully",
		Data:    nodeResults,
	}, nil
}

// SearchMemoryFactsTool handles searching for facts (edges)
func (s *MCPServer) SearchMemoryFactsTool(ctx context.Context, input *SearchRequest) (*MCPToolResponse, error) {
	// Validate required fields
	if input.Query == "" {
		return &MCPToolResponse{
			Success: false,
			Error:   "Query is required",
		}, nil
	}

	// Set defaults
	if input.Limit <= 0 {
		input.Limit = 10
	}

	// Create search configuration focused on edges
	searchConfig := &types.SearchConfig{
		Limit:              input.Limit,
		CenterNodeDistance: 2,
		MinScore:           0.0,
		IncludeEdges:       true,
		Rerank:             true,
		EdgeConfig: &types.EdgeSearchConfig{
			SearchMethods: []string{"bm25", "cosine_similarity"},
			Reranker:      "rrf",
			MinScore:      0.0,
		},
	}

	// Perform search
	results, err := s.client.Search(ctx, input.Query, searchConfig)
	if err != nil {
		s.logger.Error("Failed to search facts", "error", err)
		return &MCPToolResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to search facts: %v", err),
		}, nil
	}

	if len(results.Edges) == 0 {
		return &MCPToolResponse{
			Success: true,
			Message: "No relevant facts found",
			Data:    []interface{}{},
		}, nil
	}

	// Format results
	facts := make([]map[string]interface{}, len(results.Edges))
	for i, edge := range results.Edges {
		facts[i] = map[string]interface{}{
			"uuid":       edge.Uuid,
			"type":       string(edge.Type),
			"source_id":  edge.SourceID,
			"target_id":  edge.TargetID,
			"name":       edge.Name,
			"summary":    edge.Summary,
			"strength":   edge.Strength,
			"group_id":   edge.GroupID,
			"created_at": edge.CreatedAt.Format(time.RFC3339),
			"updated_at": edge.UpdatedAt.Format(time.RFC3339),
			"valid_from": edge.ValidFrom.Format(time.RFC3339),
			"metadata":   edge.Metadata,
		}
		if edge.ValidTo != nil {
			facts[i]["valid_to"] = edge.ValidTo.Format(time.RFC3339)
		}
	}

	return &MCPToolResponse{
		Success: true,
		Message: "Facts retrieved successfully",
		Data:    facts,
	}, nil
}

// GetEpisodesTool handles getting recent episodes
func (s *MCPServer) GetEpisodesTool(ctx context.Context, input *GetEpisodesRequest) (*MCPToolResponse, error) {
	s.logger.Info("Get episodes requested", "group_id", input.GroupID, "last_n", input.LastN)

	// Set default values
	groupID := input.GroupID
	if groupID == "" {
		groupID = s.config.GroupID // Use server's default group ID
	}

	limit := input.LastN
	if limit <= 0 {
		limit = 10 // Default to 10 episodes
	}

	// Use the Predicato client to retrieve episodes
	episodeNodes, err := s.client.GetEpisodes(ctx, groupID, limit)
	if err != nil {
		s.logger.Error("Failed to retrieve episodes", "error", err)
		return &MCPToolResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to retrieve episodes: %v", err),
		}, nil
	}

	// Convert nodes to episode format
	var episodes []map[string]interface{}
	for _, node := range episodeNodes {
		episode := map[string]interface{}{
			"uuid":       node.Uuid,
			"name":       node.Name,
			"content":    node.Content,
			"group_id":   node.GroupID,
			"created_at": node.CreatedAt.Format(time.RFC3339),
		}

		// Add episode type if available
		if node.EpisodeType != "" {
			episode["episode_type"] = string(node.EpisodeType)
		}

		// Add reference time if available
		if !node.Reference.IsZero() {
			episode["reference"] = node.Reference.Format(time.RFC3339)
		}

		// Add metadata if available
		if node.Metadata != nil {
			episode["metadata"] = node.Metadata
		}

		episodes = append(episodes, episode)
	}

	s.logger.Info("Retrieved episodes", "count", len(episodes))

	return &MCPToolResponse{
		Success: true,
		Message: fmt.Sprintf("Retrieved %d episodes", len(episodes)),
		Data: map[string]interface{}{
			"episodes": episodes,
			"total":    len(episodes),
			"group_id": groupID,
		},
	}, nil
}

// ClearGraphTool handles clearing the entire graph
func (s *MCPServer) ClearGraphTool(ctx context.Context, input *ClearGraphRequest) (*MCPToolResponse, error) {
	s.logger.Info("Clear graph requested", "group_id", input.GroupID, "confirm", input.Confirm)

	// Safety check - require explicit confirmation
	if !input.Confirm {
		return &MCPToolResponse{
			Success: false,
			Error:   "Graph clearing requires explicit confirmation. Set 'confirm' to true to proceed.",
		}, nil
	}

	// Set default group ID
	groupID := input.GroupID
	if groupID == "" {
		groupID = s.config.GroupID // Use server's default group ID
	}

	// Warn about the destructive operation
	s.logger.Warn("Clearing all data from graph", "group_id", groupID)

	// Use the Predicato client to clear the graph
	err := s.client.ClearGraph(ctx, groupID)
	if err != nil {
		s.logger.Error("Failed to clear graph", "error", err, "group_id", groupID)
		return &MCPToolResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to clear graph: %v", err),
		}, nil
	}

	s.logger.Info("Graph cleared successfully", "group_id", groupID)

	return &MCPToolResponse{
		Success: true,
		Message: fmt.Sprintf("Graph cleared successfully for group '%s'", groupID),
		Data: map[string]interface{}{
			"group_id": groupID,
			"cleared":  true,
		},
	}, nil
}

// Transport handlers

// handleStdioTransport handles MCP protocol over stdio
func (s *MCPServer) handleStdioTransport(ctx context.Context) error {
	s.logger.Info("MCP server handling stdio transport")

	// For stdio transport, we read from stdin and write to stdout
	// This is a simplified implementation - a full MCP implementation would:
	// 1. Parse JSON-RPC messages from stdin
	// 2. Route them to appropriate tool handlers
	// 3. Send responses via stdout

	s.logger.Info("Stdio transport handler started. Waiting for MCP protocol messages...")

	// Simple message loop - in a real implementation this would be a JSON-RPC message handler
	for {
		select {
		case <-ctx.Done():
			s.logger.Info("Stdio transport handler shutting down")
			return ctx.Err()
		default:
			// In a real implementation, this would read and parse JSON-RPC messages
			// and call the appropriate tool methods based on the message content
			time.Sleep(100 * time.Millisecond)
		}
	}
}

// handleSSETransport handles MCP protocol over Server-Sent Events (HTTP)
func (s *MCPServer) handleSSETransport(ctx context.Context) error {
	s.logger.Info("MCP server handling SSE transport", "host", s.config.Host, "port", s.config.Port)

	// For SSE transport, we start an HTTP server
	// This is a simplified implementation - a full MCP implementation would:
	// 1. Serve MCP protocol endpoints
	// 2. Handle tool invocation requests
	// 3. Stream responses back via SSE

	address := fmt.Sprintf("%s:%d", s.config.Host, s.config.Port)
	s.logger.Info("Starting HTTP server for SSE transport", "address", address)

	// Simple HTTP server setup - in a real implementation this would have proper MCP endpoints
	// and handle the MCP protocol over HTTP/SSE
	server := &http.Server{
		Addr: address,
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`{"status": "MCP server running", "transport": "sse"}`))
		}),
	}

	// Start server in goroutine
	go func() {
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			s.logger.Error("HTTP server error", "error", err)
		}
	}()

	s.logger.Info("SSE transport handler started")

	// Wait for context cancellation
	<-ctx.Done()

	// Shutdown server gracefully
	s.logger.Info("SSE transport handler shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	return server.Shutdown(shutdownCtx)
}

// Run starts the MCP server
func (s *MCPServer) Run(ctx context.Context) error {
	s.logger.Info("Starting MCP server", "transport", s.config.Transport)

	// Register all tools
	if err := s.RegisterTools(); err != nil {
		return fmt.Errorf("failed to register MCP tools: %w", err)
	}

	s.logger.Info("MCP server is ready to accept requests")

	// Handle different transport protocols
	switch s.config.Transport {
	case "stdio":
		s.logger.Info("Starting MCP server with stdio transport")
		return s.handleStdioTransport(ctx)
	case "sse":
		s.logger.Info("Starting MCP server with SSE transport", "host", s.config.Host, "port", s.config.Port)
		return s.handleSSETransport(ctx)
	default:
		return fmt.Errorf("unsupported transport: %s", s.config.Transport)
	}
}

// Helper functions for configuration
func validateMCPConfig(config *MCPConfig) error {
	if config.GroupID == "" {
		return fmt.Errorf("group ID is required")
	}

	if config.DatabaseURI == "" {
		return fmt.Errorf("Database URI is required")
	}

	// Only require API key if custom entities are enabled AND no base URL is provided
	// This allows for OpenAI-compatible services that might not need API keys or use different auth
	if config.UseCustomEntities && config.OpenAIAPIKey == "" && config.LLMBaseURL == "" {
		return fmt.Errorf("LLM API key is required when custom entities are enabled, unless using a custom base URL")
	}

	if config.Port <= 0 || config.Port > 65535 {
		return fmt.Errorf("invalid port: %d", config.Port)
	}

	return nil
}

// Viper helper functions with fallback support
func getViperStringWithFallback(key, fallback string) string {
	if viper.IsSet(key) {
		return viper.GetString(key)
	}
	return fallback
}

func getViperIntWithFallback(key string, fallback int) int {
	if viper.IsSet(key) {
		return viper.GetInt(key)
	}
	return fallback
}

func getViperFloat64WithFallback(key string, fallback float64) float64 {
	if viper.IsSet(key) {
		return viper.GetFloat64(key)
	}
	return fallback
}

func getViperBoolWithFallback(key string, fallback bool) bool {
	if viper.IsSet(key) {
		return viper.GetBool(key)
	}
	return fallback
}
