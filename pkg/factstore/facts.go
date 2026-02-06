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

	// Edges are the matching extracted edges
	Edges []*ExtractedEdge `json:"edges,omitempty"`

	// NodeScores are the relevance scores for each node (same order as Nodes)
	NodeScores []float64 `json:"node_scores"`

	// EdgeScores are the relevance scores for each edge (same order as Edges)
	EdgeScores []float64 `json:"edge_scores,omitempty"`

	// Total is the total number of results
	Total int `json:"total"`
}

// FactStoreType defines the backend database type
type FactStoreType string

const (
	// FactStoreTypePostgres uses external PostgreSQL with VectorChord
	FactStoreTypePostgres FactStoreType = "postgres"
	// FactStoreTypeDoltGres uses embedded DoltGres (PostgreSQL-compatible)
	FactStoreTypeDoltGres FactStoreType = "doltgres"
)

// FactStoreConfig configures the factstore backend
type FactStoreConfig struct {
	// Type is the backend type: "postgres" or "doltgres" (default)
	Type FactStoreType `json:"type,omitempty"`

	// ConnectionString for the database
	// Postgres: "postgres://user:pass@host:5432/database?sslmode=disable"
	// DoltGres: "postgres://user:pass@localhost:5432/database" (embedded)
	ConnectionString string `json:"connection_string,omitempty"`

	// DataDir is the directory for embedded DoltGres data (only for doltgres type)
	DataDir string `json:"data_dir,omitempty"`

	// EmbeddingDimensions is the vector dimension (e.g., 1024 for qwen3-embedding)
	EmbeddingDimensions int `json:"embedding_dimensions,omitempty"`

	// MaxConnections for connection pooling (postgres only)
	MaxConnections int `json:"max_connections,omitempty"`
}

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

// ExtractedEdge is an alias to types.ExtractedEdge for backward compatibility.
// The canonical definition is in pkg/types/extraction.go.
type ExtractedEdge = types.ExtractedEdge

// FactsDB defines the interface for the intermediate knowledge storage.
type FactsDB interface {
	// Initialize ensures the database schema exists.
	Initialize(ctx context.Context) error

	// SaveSource saves the source metadata.
	SaveSource(ctx context.Context, source *Source) error

	// SaveExtractedKnowledge saves the raw extraction results.
	SaveExtractedKnowledge(ctx context.Context, sourceID string, nodes []*ExtractedNode, edges []*ExtractedEdge) error

	// GetSource retrieves a source by ID.
	GetSource(ctx context.Context, sourceID string) (*Source, error)

	// GetExtractedNodes retrieves extracted nodes for a source.
	GetExtractedNodes(ctx context.Context, sourceID string) ([]*ExtractedNode, error)

	// GetExtractedEdges retrieves extracted edges for a source.
	GetExtractedEdges(ctx context.Context, sourceID string) ([]*ExtractedEdge, error)

	// GetAllSources retrieves all sources.
	GetAllSources(ctx context.Context, limit int) ([]*Source, error)

	// GetAllNodes retrieves all nodes (with optional limit).
	GetAllNodes(ctx context.Context, limit int) ([]*ExtractedNode, error)

	// GetAllEdges retrieves all edges (with optional limit).
	GetAllEdges(ctx context.Context, limit int) ([]*ExtractedEdge, error)

	// GetStats retrieves statistics about the fact store.
	GetStats(ctx context.Context) (*Stats, error)

	// Close closes the database connection.
	Close() error

	// --- Search Methods ---

	// SearchNodes performs similarity and/or keyword search on extracted nodes.
	// If embedding is nil, only keyword search is performed.
	// If query is empty, only vector search is performed.
	SearchNodes(ctx context.Context, query string, embedding []float32, config *FactSearchConfig) ([]*ExtractedNode, []float64, error)

	// SearchEdges performs similarity and/or keyword search on extracted edges.
	SearchEdges(ctx context.Context, query string, embedding []float32, config *FactSearchConfig) ([]*ExtractedEdge, []float64, error)

	// SearchSources performs keyword search on source content.
	SearchSources(ctx context.Context, query string, config *FactSearchConfig) ([]*Source, []float64, error)

	// HybridSearch performs combined vector and keyword search with RRF fusion.
	HybridSearch(ctx context.Context, query string, embedding []float32, config *FactSearchConfig) (*FactSearchResults, error)
}

type Stats struct {
	SourceCount int64
	NodeCount   int64
	EdgeCount   int64
}
