package factstore

import (
	"context"
	"time"

	"github.com/soundprediction/predicato/pkg/types"
)

// SearchMethod defines the type of search to perform
type SearchMethod string

const (
	// VectorSearch performs similarity search using embeddings
	VectorSearch SearchMethod = "vector"
	// KeywordSearch performs full-text keyword search
	KeywordSearch SearchMethod = "keyword"
)

// TimeRange defines a time window for filtering
type TimeRange struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

// FactSearchConfig configures a factstore search query
type FactSearchConfig struct {

	// TimeRange filters by created_at timestamp
	TimeRange *TimeRange `json:"time_range,omitempty"`

	// GroupID filters results to a specific group (multi-tenant)
	GroupID string `json:"group_id,omitempty"`

	// NodeTypes filters to specific entity types (e.g., ["person", "organization"])
	NodeTypes []string `json:"node_types,omitempty"`

	// SearchMethods specifies which search methods to use
	// Default: [VectorSearch, KeywordSearch] (hybrid)
	SearchMethods []SearchMethod `json:"search_methods,omitempty"`

	// Limit is the maximum number of results to return
	Limit int `json:"limit,omitempty"`

	// MinScore filters out results below this similarity threshold (0.0-1.0)
	MinScore float64 `json:"min_score,omitempty"`
}

// FactSearchResults contains the results of a factstore search
type FactSearchResults struct {

	// Query is the original query string
	Query string `json:"query"`

	// Nodes are the matching extracted nodes
	Nodes []*ExtractedNode `json:"nodes"`

	// Triples are the matching extracted triples
	Triples []*ExtractedTriple `json:"triples,omitempty"`

	// NodeScores are the relevance scores for each node (same order as Nodes)
	NodeScores []float64 `json:"node_scores"`

	// TripleScores are the relevance scores for each triple (same order as Triples)
	TripleScores []float64 `json:"triple_scores,omitempty"`

	// Total is the total number of results
	Total int `json:"total"`
}

// DbType defines the backend database type
type DbType string

const (
	// DbTypePostgres uses external PostgreSQL with VectorChord
	DbTypePostgres DbType = "postgres"
	// DbTypeDoltGres uses embedded DoltGres (PostgreSQL-compatible)
	DbTypeDoltGres DbType = "doltgres"
	// DbTypeMySQL uses MySQL-compatible databases
	DbTypeMySQL DbType = "mysql"
	// DbTypeSQLite uses SQLite for local/embedded storage
	DbTypeSQLite DbType = "sqlite"
	// DbTypeDuckDB uses DuckDB for analytical workloads
	DbTypeDuckDB DbType = "duckdb"
)

// Deprecated aliases for backward compatibility.
type FactStoreType = DbType

const (
	FactStoreTypePostgres = DbTypePostgres
	FactStoreTypeDoltGres = DbTypeDoltGres
)

// DbConfig configures the structured store backend.
// Either set DB to provide a pre-built FactsDB instance (e.g. to share
// a connection pool with the host application), or set Type and
// ConnectionString to have one created automatically.
type DbConfig struct {
	// DB is an optional pre-built FactsDB instance.
	// When set, Type/ConnectionString/DataDir are ignored and no new
	// connection is opened.
	DB FactsDB `json:"-"`

	// Type is the backend type: "postgres", "doltgres", "mysql", "sqlite", "duckdb"
	Type DbType `json:"type,omitempty"`

	// ConnectionString for the database
	// Postgres: "postgres://user:pass@host:5432/database?sslmode=disable"
	// MySQL:    "user:pass@tcp(host:3306)/database"
	// SQLite:   "file:path/to/db.sqlite"
	// DuckDB:   "path/to/db.duckdb"
	ConnectionString string `json:"connection_string,omitempty"`

	// DataDir is the directory for embedded DoltGres data (only for doltgres type)
	DataDir string `json:"data_dir,omitempty"`

	// EmbeddingDimensions is the vector dimension (e.g., 1024 for qwen3-embedding)
	EmbeddingDimensions int `json:"embedding_dimensions,omitempty"`

	// MaxConnections for connection pooling (postgres/mysql only)
	MaxConnections int `json:"max_connections,omitempty"`
}

// Deprecated: use DbConfig instead.
type FactStoreConfig = DbConfig

// Source represents the origin of extracted information.
type Source struct {
	CreatedAt time.Time              `json:"created_at"`
	Metadata  map[string]interface{} `json:"metadata"`
	ID        string                 `json:"id"`
	Name      string                 `json:"name"`
	Content   string                 `json:"content"`
	GroupID   string                 `json:"group_id"`
}

// ExtractedNode is an alias to types.ExtractedNode for backward compatibility.
// The canonical definition is in pkg/types/extraction.go.
type ExtractedNode = types.ExtractedNode

// ExtractedTriple is an alias to types.ExtractedTriple for backward compatibility.
// The canonical definition is in pkg/types/extraction.go.
type ExtractedTriple = types.ExtractedTriple

// ExtractedRule is an alias to types.ExtractedRule for backward compatibility.
// The canonical definition is in pkg/types/extraction.go.
type ExtractedRule = types.ExtractedRule

// FactsDB defines the interface for the intermediate knowledge storage.
type FactsDB interface {
	// Initialize ensures the database schema exists.
	Initialize(ctx context.Context) error

	// SaveSource saves the source metadata.
	SaveSource(ctx context.Context, source *Source) error

	// SaveExtractedKnowledge saves the raw extraction results (nodes and triples).
	SaveExtractedKnowledge(ctx context.Context, sourceID string, nodes []*ExtractedNode, triples []*ExtractedTriple) error

	// GetSource retrieves a source by ID.
	GetSource(ctx context.Context, sourceID string) (*Source, error)

	// GetExtractedNodes retrieves extracted nodes for a source.
	GetExtractedNodes(ctx context.Context, sourceID string) ([]*ExtractedNode, error)

	// GetExtractedTriples retrieves extracted triples for a source.
	GetExtractedTriples(ctx context.Context, sourceID string) ([]*ExtractedTriple, error)

	// GetAllSources retrieves all sources.
	GetAllSources(ctx context.Context, limit int) ([]*Source, error)

	// GetAllNodes retrieves all nodes (with optional limit).
	GetAllNodes(ctx context.Context, limit int) ([]*ExtractedNode, error)

	// GetAllTriples retrieves all triples (with optional limit).
	GetAllTriples(ctx context.Context, limit int) ([]*ExtractedTriple, error)

	// GetStats retrieves statistics about the fact store.
	GetStats(ctx context.Context) (*Stats, error)

	// Close closes the database connection.
	Close() error

	// --- Search Methods ---

	// SearchNodes performs similarity and/or keyword search on extracted nodes.
	// If embedding is nil, only keyword search is performed.
	// If query is empty, only vector search is performed.
	SearchNodes(ctx context.Context, query string, embedding []float32, config *FactSearchConfig) ([]*ExtractedNode, []float64, error)

	// SearchTriples performs similarity and/or keyword search on extracted triples.
	SearchTriples(ctx context.Context, query string, embedding []float32, config *FactSearchConfig) ([]*ExtractedTriple, []float64, error)

	// SearchSources performs keyword search on source content.
	SearchSources(ctx context.Context, query string, config *FactSearchConfig) ([]*Source, []float64, error)

	// HybridSearch performs combined vector and keyword search with RRF fusion.
	HybridSearch(ctx context.Context, query string, embedding []float32, config *FactSearchConfig) (*FactSearchResults, error)

	// --- Extended Extraction Methods ---

	// SaveExtractedRules saves conditional rules for a source.
	SaveExtractedRules(ctx context.Context, sourceID string, rules []*ExtractedRule) error

	// GetExtractedRules retrieves extracted rules for a source.
	GetExtractedRules(ctx context.Context, sourceID string) ([]*ExtractedRule, error)

	// --- Paginated Methods (for large sources) ---

	// GetExtractedNodesPaginated retrieves a batch of extracted nodes for a source.
	GetExtractedNodesPaginated(ctx context.Context, sourceID string, offset, limit int) ([]*ExtractedNode, error)

	// GetExtractedTriplesPaginated retrieves a batch of extracted triples for a source.
	GetExtractedTriplesPaginated(ctx context.Context, sourceID string, offset, limit int) ([]*ExtractedTriple, error)

	// GetExtractedRulesPaginated retrieves a batch of extracted rules for a source.
	GetExtractedRulesPaginated(ctx context.Context, sourceID string, offset, limit int) ([]*ExtractedRule, error)

	// CountExtractedNodes returns the number of extracted nodes for a source.
	CountExtractedNodes(ctx context.Context, sourceID string) (int64, error)

	// CountExtractedTriples returns the number of extracted triples for a source.
	CountExtractedTriples(ctx context.Context, sourceID string) (int64, error)

	// CountExtractedRules returns the number of extracted rules for a source.
	CountExtractedRules(ctx context.Context, sourceID string) (int64, error)
}

type Stats struct {
	SourceCount int64
	NodeCount   int64
	TripleCount int64
	RuleCount   int64
}
