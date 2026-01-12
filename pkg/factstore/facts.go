package factstore

import (
	"context"
	"time"
)

// Source represents the origin of extracted information.
type Source struct {
	ID        string                 `json:"id"`
	Name      string                 `json:"name"`
	Content   string                 `json:"content"`
	GroupID   string                 `json:"group_id"`
	Metadata  map[string]interface{} `json:"metadata"`
	CreatedAt time.Time              `json:"created_at"`
}

// ExtractedNode represents a raw entity extracted from a source.
type ExtractedNode struct {
	ID          string    `json:"id"`
	SourceID    string    `json:"source_id"`
	Name        string    `json:"name"`
	Type        string    `json:"type"`
	EntityType  string    `json:"entity_type"`
	Description string    `json:"description"`
	Embedding   []float32 `json:"embedding"`
	ChunkIndex  int       `json:"chunk_index"`
}

// ExtractedEdge represents a raw relationship extracted from a source.
type ExtractedEdge struct {
	ID             string  `json:"id"`
	SourceID       string  `json:"source_id"`
	SourceNodeName string  `json:"source_node_name"`
	TargetNodeName string  `json:"target_node_name"`
	Relation       string  `json:"relation"`
	Description    string  `json:"description"`
	Weight         float64 `json:"weight"`
	ChunkIndex     int     `json:"chunk_index"`
}

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
}

type Stats struct {
	SourceCount int64
	NodeCount   int64
	EdgeCount   int64
}
