package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"log/slog"
	"os"
	"strconv"
	"time"

	"github.com/firebase/genkit/go/genkit"
	"github.com/soundprediction/predicato"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/embedder"
	predicatoLogger "github.com/soundprediction/predicato/pkg/logger"
	"github.com/soundprediction/predicato/pkg/nlp"
)

// Default configuration values
const (
	DefaultNLPModel       = "gpt-4o-mini"
	DefaultSmallModel     = "gpt-4o-mini"
	DefaultEmbedderModel  = "text-embedding-3-small"
	DefaultSemaphoreLimit = 10
)

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

// Config holds all configuration for the MCP server
type Config struct {
	// NLP Configuration
	NLPModel       string
	SmallNLPModel  string
	NLPTemperature float64
	OpenAIAPIKey   string

	// Embedder Configuration
	EmbedderModel string

	// Database Configuration
	DatabaseDriver   string
	DatabaseURI      string
	DatabaseUser     string
	DatabasePassword string

	// MCP Server Configuration
	GroupID           string
	UseCustomEntities bool
	DestroyGraph      bool
	Transport         string
	Host              string
	Port              int

	// Concurrency limits
	SemaphoreLimit int
}

// MCPServer wraps the Predicato client for MCP operations
type MCPServer struct {
	config *Config
	client *predicato.Client
	logger *slog.Logger
}

// NewConfig creates a new configuration from environment variables and command line flags
func NewConfig() *Config {
	config := &Config{
		NLPModel:          getEnv("MODEL_NAME", DefaultNLPModel),
		SmallNLPModel:     getEnv("SMALL_MODEL_NAME", DefaultSmallModel),
		NLPTemperature:    getEnvFloat("LLM_TEMPERATURE", 0.0),
		OpenAIAPIKey:      getEnv("OPENAI_API_KEY", ""),
		EmbedderModel:     getEnv("EMBEDDER_MODEL_NAME", DefaultEmbedderModel),
		DatabaseDriver:    getEnv("DB_DRIVER", "ladybug"),
		DatabaseURI:       getEnv("DB_URI", getEnv("ladybug_DB_PATH", "./ladybug_db")),
		DatabaseUser:      getEnv("NEO4J_USER", ""),
		DatabasePassword:  getEnv("NEO4J_PASSWORD", ""),
		GroupID:           getEnv("GROUP_ID", "default"),
		UseCustomEntities: getEnvBool("USE_CUSTOM_ENTITIES", false),
		DestroyGraph:      getEnvBool("DESTROY_GRAPH", false),
		Transport:         getEnv("MCP_TRANSPORT", "stdio"),
		Host:              getEnv("MCP_HOST", "localhost"),
		Port:              getEnvInt("MCP_PORT", 3000),
		SemaphoreLimit:    getEnvInt("SEMAPHORE_LIMIT", DefaultSemaphoreLimit),
	}

	return config
}

// Helper functions for environment variable parsing
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvFloat(key string, defaultValue float64) float64 {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.ParseFloat(value, 64); err == nil {
			return parsed
		}
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.ParseBool(value); err == nil {
			return parsed
		}
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.Atoi(value); err == nil {
			return parsed
		}
	}
	return defaultValue
}

// NewMCPServer creates a new MCP server instance
func NewMCPServer(config *Config) (*MCPServer, error) {
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

	// Create NLP client
	var nlProcessor nlp.Client
	if config.OpenAIAPIKey != "" {
		nlpConfig := nlp.Config{
			Model:       config.NLPModel,
			Temperature: &[]float32{float32(config.NLPTemperature)}[0],
		}
		baseNLPClient, err := nlp.NewOpenAIClient(config.OpenAIAPIKey, nlpConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create NLP client: %w", err)
		}
		// Wrap with retry client for automatic retry on errors
		nlProcessor, err = nlp.NewRetryClient(baseNLPClient, nlp.DefaultRetryConfig())
		if err != nil {
			return nil, fmt.Errorf("failed to create retry client: %w", err)
		}
	}

	// Create embedder client
	var embedderClient embedder.Client
	if config.OpenAIAPIKey != "" {
		embedderConfig := embedder.Config{
			Model: config.EmbedderModel,
		}
		embedderClient = embedder.NewOpenAIEmbedder(config.OpenAIAPIKey, embedderConfig)
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

	// Initialize graph indices and constraints
	s.logger.Info("Initializing graph indices and constraints...")
	err := s.client.CreateIndices(ctx)
	if err != nil {
		s.logger.Error("Failed to initialize graph indices", "error", err)
		return fmt.Errorf("failed to initialize graph indices: %w", err)
	}
	s.logger.Info("Graph indices and constraints initialized successfully")

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
		"llm_model", s.config.NLPModel,
		"temperature", s.config.NLPTemperature,
		"group_id", s.config.GroupID,
		"custom_entities", s.config.UseCustomEntities,
		"semaphore_limit", s.config.SemaphoreLimit,
	)

	return nil
}

// RegisterTools registers all MCP tools with Genkit
func (s *MCPServer) RegisterTools(g *genkit.Genkit) {
	// Register add_memory tool
	genkit.DefineTool(g, "add_memory",
		"Add an episode to memory. This is the primary way to add information to the graph.",
		s.AddMemoryTool)

	// Register search_memory_nodes tool
	genkit.DefineTool(g, "search_memory_nodes",
		"Search the graph memory for relevant node summaries.",
		s.SearchMemoryNodesTool)

	// Register search_memory_facts tool
	genkit.DefineTool(g, "search_memory_facts",
		"Search the graph memory for relevant facts.",
		s.SearchMemoryFactsTool)

	// Register delete_entity_edge tool
	genkit.DefineTool(g, "delete_entity_edge",
		"Delete an entity edge from the graph memory.",
		s.DeleteEntityEdgeTool)

	// Register delete_episode tool
	genkit.DefineTool(g, "delete_episode",
		"Delete an episode from the graph memory.",
		s.DeleteEpisodeTool)

	// Register get_entity_edge tool
	genkit.DefineTool(g, "get_entity_edge",
		"Get an entity edge from the graph memory by its UUID.",
		s.GetEntityEdgeTool)

	// Register get_episodes tool
	genkit.DefineTool(g, "get_episodes",
		"Get the most recent memory episodes for a specific group.",
		s.GetEpisodesTool)

	// Register clear_graph tool
	genkit.DefineTool(g, "clear_graph",
		"Clear all data from the graph memory.",
		s.ClearGraphTool)
}

// Run starts the MCP server
func (s *MCPServer) Run(ctx context.Context) error {
	s.logger.Info("Starting Genkit MCP server", "transport", s.config.Transport)

	// Initialize Genkit
	g := genkit.Init(ctx)

	// Register all tools
	s.RegisterTools(g)

	// Start the server (this would typically be handled by Genkit's runtime)
	s.logger.Info("MCP server is ready to accept requests")

	// Keep the server running
	<-ctx.Done()
	return ctx.Err()
}

func main() {
	// Parse command line flags
	var (
		groupID           = flag.String("group-id", "", "Namespace for the graph")
		transport         = flag.String("transport", "stdio", "Transport to use (stdio or sse)")
		model             = flag.String("model", "", fmt.Sprintf("Model name to use (default: %s)", DefaultNLPModel))
		smallModel        = flag.String("small-model", "", fmt.Sprintf("Small model name to use (default: %s)", DefaultSmallModel))
		temperature       = flag.Float64("temperature", -1, "Temperature setting for the LLM (0.0-2.0)")
		destroyGraph      = flag.Bool("destroy-graph", false, "Destroy all Predicato graphs")
		useCustomEntities = flag.Bool("use-custom-entities", false, "Enable entity extraction using predefined entity types")
		host              = flag.String("host", "", "Host to bind the MCP server to")
		port              = flag.Int("port", 0, "Port to bind the MCP server to")
	)
	flag.Parse()

	// Create configuration
	config := NewConfig()

	// Apply command line overrides
	if *groupID != "" {
		config.GroupID = *groupID
	}
	if *transport != "" {
		config.Transport = *transport
	}
	if *model != "" {
		config.NLPModel = *model
	}
	if *smallModel != "" {
		config.SmallNLPModel = *smallModel
	}
	if *temperature >= 0 {
		config.NLPTemperature = *temperature
	}
	if *destroyGraph {
		config.DestroyGraph = true
	}
	if *useCustomEntities {
		config.UseCustomEntities = true
	}
	if *host != "" {
		config.Host = *host
	}
	if *port != 0 {
		config.Port = *port
	}

	// Validate required configuration
	if config.OpenAIAPIKey == "" && config.UseCustomEntities {
		log.Fatal("OPENAI_API_KEY must be set when custom entities are enabled")
	}

	// Validate database configuration based on driver type
	if config.DatabaseURI == "" {
		log.Fatal("Database URI/path must be set")
	}

	// Only Neo4j requires username and password
	if config.DatabaseDriver == "neo4j" && (config.DatabaseUser == "" || config.DatabasePassword == "") {
		log.Fatal("NEO4J_USER and NEO4J_PASSWORD must be set when using Neo4j driver")
	}

	// Create and initialize server
	server, err := NewMCPServer(config)
	if err != nil {
		log.Fatalf("Failed to create MCP server: %v", err)
	}

	ctx := context.Background()
	if err := server.Initialize(ctx); err != nil {
		log.Fatalf("Failed to initialize MCP server: %v", err)
	}

	// Run the server
	if err := server.Run(ctx); err != nil {
		log.Fatalf("MCP server error: %v", err)
	}
}
