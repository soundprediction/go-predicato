package predicato

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/soundprediction/predicato/pkg/community"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/embedder"
	"github.com/soundprediction/predicato/pkg/factstore"
	"github.com/soundprediction/predicato/pkg/modeler"
	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/search"
	"github.com/soundprediction/predicato/pkg/types"
	"github.com/soundprediction/predicato/pkg/utils/maintenance"
)

// driverWrapper wraps driver.GraphDriver to implement types.EdgeOperations
type driverWrapper struct {
	driver.GraphDriver
}

// Provider returns the graph provider type.
// Since driver.GraphProvider is now an alias to types.GraphProvider,
// no conversion is needed.
func (w *driverWrapper) Provider() types.GraphProvider {
	return w.GraphDriver.Provider()
}

// nodeOpsWrapper adapts maintenance.NodeOperations to utils.NodeOperations interface
type nodeOpsWrapper struct {
	*maintenance.NodeOperations
}

// ResolveExtractedNodes wraps maintenance.NodeOperations.ResolveExtractedNodes to match the interface
func (w *nodeOpsWrapper) ResolveExtractedNodes(ctx context.Context, extractedNodes []*types.Node, episode *types.Node, previousEpisodes []*types.Node, entityTypes map[string]interface{}) ([]*types.Node, map[string]string, interface{}, error) {
	nodes, uuidMap, pairs, err := w.NodeOperations.ResolveExtractedNodes(ctx, extractedNodes, episode, previousEpisodes, entityTypes)
	// Return pairs as interface{} to satisfy the interface
	return nodes, uuidMap, pairs, err
}

// Predicato is the main interface for interacting with temporal knowledge graphs.
// It provides methods for building, querying, and maintaining temporally-aware
// knowledge graphs designed for AI agents.
type Predicato interface {
	// Add processes and adds new episodes to the knowledge graph.
	// Episodes can be text, conversations, or any temporal data.
	// Options parameter is optional and can be nil for default behavior.
	Add(ctx context.Context, episodes []types.Episode, options *AddEpisodeOptions) (*types.AddBulkEpisodeResults, error)

	// AddEpisode processes and adds a single episode to the knowledge graph.
	// This is equivalent to the Python add_episode method.
	AddEpisode(ctx context.Context, episode types.Episode, options *AddEpisodeOptions) (*types.AddEpisodeResults, error)

	// Search performs hybrid search across the knowledge graph combining
	// semantic embeddings, keyword search, and graph traversal.
	Search(ctx context.Context, query string, config *types.SearchConfig) (*types.SearchResults, error)

	// GetNode retrieves a specific node from the knowledge graph.
	GetNode(ctx context.Context, nodeID string) (*types.Node, error)

	// GetEdge retrieves a specific edge from the knowledge graph.
	GetEdge(ctx context.Context, edgeID string) (*types.Edge, error)

	// GetEpisodes retrieves recent episodes from the knowledge graph.
	GetEpisodes(ctx context.Context, groupID string, limit int) ([]*types.Node, error)

	// ClearGraph removes all nodes and edges from the knowledge graph for a specific group.
	ClearGraph(ctx context.Context, groupID string) error

	// CreateIndices creates database indices and constraints for optimal performance.
	CreateIndices(ctx context.Context) error

	// AddTriplet adds a triplet (subject-predicate-object) directly to the knowledge graph.
	AddTriplet(ctx context.Context, sourceNode *types.Node, edge *types.Edge, targetNode *types.Node, createEmbeddings bool) (*types.AddTripletResults, error)

	// RemoveEpisode removes an episode and its associated nodes and edges from the knowledge graph.
	RemoveEpisode(ctx context.Context, episodeUUID string) error

	// GetNodesAndEdgesByEpisode retrieves all nodes and edges associated with a specific episode.
	GetNodesAndEdgesByEpisode(ctx context.Context, episodeUUID string) ([]*types.Node, []*types.Edge, error)

	// Close closes all connections and cleans up resources.
	Close(ctx context.Context) error

	UpdateCommunities(ctx context.Context, episodeUUID string, groupID string) ([]*types.Node, []*types.Edge, error)

	// GetFactStore returns the underlying fact store
	GetFactStore() factstore.FactsDB

	// SearchFacts performs RAG search directly on the factstore without graph queries.
	// This is useful for simpler RAG use cases that don't need relationship traversal.
	// The query is embedded using the configured embedder, then hybrid search is performed.
	SearchFacts(ctx context.Context, query string, config *types.SearchConfig) (*factstore.FactSearchResults, error)

	// ExtractToFacts extracts entities and relationships from an episode and stores them
	// in the fact store without promoting to the graph. Use PromoteToGraph to later
	// promote facts to the graph with custom GraphModeler logic.
	//
	// This is the first step of the two-phase ingestion pipeline when you want to
	// decouple extraction from graph modeling.
	//
	// Requires DbConfig to be set in Config.
	ExtractToFacts(ctx context.Context, episode types.Episode, options *AddEpisodeOptions) (*types.ExtractionResults, error)

	// PromoteToGraph takes previously extracted facts from the fact store and promotes
	// them to the knowledge graph using the configured GraphModeler.
	//
	// This is the second step of the two-phase ingestion pipeline. It reads extracted
	// entities and relationships from the fact store and applies entity resolution,
	// relationship resolution, and community detection according to the GraphModeler.
	//
	// The sourceID parameter is the episode UUID returned from ExtractToFacts.
	PromoteToGraph(ctx context.Context, sourceID string, options *AddEpisodeOptions) (*types.AddEpisodeResults, error)

	// ValidateModeler tests a GraphModeler implementation with sample data to verify
	// it works correctly before using it in production.
	//
	// Returns validation results including pass/fail status for each method,
	// latency measurements, and any warnings.
	ValidateModeler(ctx context.Context, gm modeler.GraphModeler) (*modeler.ModelerValidationResult, error)

	// AnalyzeContentRelevance uses the configured LLM to determine if content is
	// relevant to the specified topics. Returns true if the content is relevant.
	// This is useful for filtering content before ingestion.
	AnalyzeContentRelevance(ctx context.Context, content string, topics []string) (bool, error)

	// ExtractSourceInfo uses the configured LLM to extract source metadata from
	// web content. Returns the source name and type (e.g., "CDC", "government_health_agency").
	ExtractSourceInfo(ctx context.Context, content string, url string) (sourceName string, sourceType string, err error)
}

// Client is the main implementation of the Predicato interface.
type Client struct {
	driver      driver.GraphDriver
	nlProcessor nlp.Client
	embedder    embedder.Client
	searcher    *search.Searcher
	community   *community.Builder
	config      *Config
	logger      *slog.Logger
	factStore   factstore.FactsDB

	// Specialized NLP clients for different steps
	nlpModels NlpModels
}

// NlpModels holds specialized NLP clients for different pipeline steps.
//
// NOTE: This type is intentionally duplicated in pkg/modeler/default.go to avoid
// an import cycle (modeler cannot import root package). Both types should be
// kept in sync. If adding fields here, also add them to modeler.NlpModels.
type NlpModels struct {
	NodeExtraction nlp.Client
	NodeReflexion  nlp.Client
	NodeResolution nlp.Client
	NodeAttribute  nlp.Client
	EdgeExtraction nlp.Client
	EdgeResolution nlp.Client
	Summarization  nlp.Client
	TextGeneration nlp.Client
}

// Config holds configuration for the Predicato client.
type Config struct {
	// NlpModels holds specialized NLP clients for different steps
	NlpModels NlpModels

	// DefaultGraphModeler is the default GraphModeler used for graph promotion.
	// If nil, a DefaultModeler is created using the client's NLP/embedder config.
	DefaultGraphModeler modeler.GraphModeler
	// TimeZone for temporal operations
	TimeZone *time.Location
	// Search configuration
	SearchConfig *types.SearchConfig
	// DefaultEntityTypes defines the default entity types to use when AddEpisodeOptions.EntityTypes is nil
	EntityTypes map[string]interface{}
	EdgeTypes   map[string]interface{}

	// DbConfig configures the structured store backend.
	// Set DbConfig.DB to pass a pre-built FactsDB, or set Type and
	// ConnectionString to have one created automatically.
	DbConfig *factstore.DbConfig

	EdgeMap map[string]map[string][]interface{}
	// GroupID is used to isolate data for multi-tenant scenarios
	GroupID string

	// FactsDBURL is the connection string for the Dolt facts database
	// Deprecated: Use DbConfig instead
	FactsDBURL string
}

// AddEpisodeOptions holds options for adding a single episode.
// This matches the optional parameters from the Python add_episode method.
type AddEpisodeOptions struct {

	// GraphModeler overrides the default graph modeler for this episode.
	// If nil, uses Config.DefaultGraphModeler or creates a DefaultModeler.
	GraphModeler modeler.GraphModeler

	// EntityTypes custom entity type definitions
	EntityTypes map[string]interface{}
	// EdgeTypes custom edge type definitions
	EdgeTypes map[string]interface{}
	// EdgeTypeMap mapping of entity pairs to edge types
	EdgeTypeMap map[string]map[string][]interface{}
	// ExcludedEntityTypes entity types to exclude from extraction
	ExcludedEntityTypes []string
	// PreviousEpisodeUUIDs UUIDs of previous episodes for context
	PreviousEpisodeUUIDs []string
	MaxCharacters        int

	// ModelerErrorHandling controls how errors from GraphModeler are handled.
	// Default is FailOnError.
	ModelerErrorHandling modeler.ModelerErrorHandling

	// OverwriteExisting whether to overwrite an existing episode with the same UUID
	// Default behavior is false (skip if exists)
	OverwriteExisting  bool
	GenerateEmbeddings bool

	// Skip options for faster ingestion or debugging
	SkipReflexion      bool
	SkipResolution     bool
	SkipAttributes     bool
	SkipEdgeResolution bool

	// UseYAML toggles between CSV/TSV (default) and YAML for LLM interchange
	UseYAML bool

	// ExtractOnly stops after extraction, storing facts in the factstore but not
	// promoting to the graph. Use PromoteToGraph to complete ingestion later.
	// Requires DbConfig to be configured.
	ExtractOnly bool

	// Verbose enables detailed logging of extracted entities, relations, and summaries
	// during ingestion. Useful for debugging and monitoring ingestion pipelines.
	Verbose bool
}

// NewClient creates a new Predicato client with the provided configuration.
func NewClient(driver driver.GraphDriver, nlProcessor nlp.Client, embedderClient embedder.Client, config *Config, logger *slog.Logger) (*Client, error) {
	if config == nil {
		config = &Config{
			GroupID:  "default",
			TimeZone: time.UTC,
		}
	}
	if config.SearchConfig == nil {
		config.SearchConfig = NewDefaultSearchConfig()
	}
	if logger == nil {
		logger = slog.Default()
	}

	searcher := search.NewSearcher(driver, embedderClient, nlProcessor)
	communityBuilder, err := community.NewBuilder(driver, nlProcessor, config.NlpModels.Summarization, embedderClient)
	if err != nil {
		return nil, fmt.Errorf("failed to create community builder: %w", err)
	}

	var factStore factstore.FactsDB

	// Create or reuse the structured store
	if config.DbConfig != nil {
		factStore, err = factstore.NewFactsDB(config.DbConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create factstore: %w", err)
		}
		// Only auto-initialize when the factory created a new connection
		// (i.e. no pre-built DB was provided)
		if config.DbConfig.DB == nil {
			if err := factStore.Initialize(context.Background()); err != nil {
				return nil, fmt.Errorf("failed to initialize factstore: %w", err)
			}
		}
	} else if config.FactsDBURL != "" {
		// Legacy Dolt support (deprecated)
		fdb, err := factstore.NewDoltDB(config.FactsDBURL)
		if err != nil {
			return nil, err
		}
		if err := fdb.Initialize(context.Background()); err != nil {
			return nil, err
		}
		factStore = fdb
	}

	return &Client{
		driver:      driver,
		nlProcessor: nlProcessor,
		embedder:    embedderClient,
		searcher:    searcher,
		community:   communityBuilder,
		config:      config,
		logger:      logger,
		factStore:   factStore,
		nlpModels:   config.NlpModels,
	}, nil
}

// GetDriver returns the underlying graph driver
func (c *Client) GetDriver() driver.GraphDriver {
	return c.driver
}

// GetNLProcessor returns the NLP client
func (c *Client) GetNLProcessor() nlp.Client {
	return c.nlProcessor
}

// GetEmbedder returns the embedder client
func (c *Client) GetEmbedder() embedder.Client {
	return c.embedder
}

// GetCommunityBuilder returns the community builder
func (c *Client) GetCommunityBuilder() *community.Builder {
	return c.community
}

// GetFactStore returns the underlying fact store
func (c *Client) GetFactStore() factstore.FactsDB {
	return c.factStore
}

var (
	// ErrNodeNotFound is returned when a node is not found.
	ErrNodeNotFound = errors.New("node not found")
	// ErrEdgeNotFound is returned when an edge is not found.
	ErrEdgeNotFound = errors.New("edge not found")
	// ErrInvalidEpisode is returned when an episode is invalid.
	ErrInvalidEpisode = errors.New("invalid episode")
)
