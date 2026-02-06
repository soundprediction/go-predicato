package types

import (
	"time"
)

// ExtractedNode represents a raw entity extracted from a source.
// This type is used by both the factstore (for persistence) and the extraction
// pipeline (for intermediate results).
type ExtractedNode struct {
	CreatedAt      time.Time `json:"created_at"`
	ID             string    `json:"id"`
	SourceID       string    `json:"source_id"`
	GroupID        string    `json:"group_id"`
	Name           string    `json:"name"`
	NormalizedName string    `json:"normalized_name"`
	Type           string    `json:"type"`
	Description    string    `json:"description"`
	Embedding      []float32 `json:"embedding"`
	ChunkIndex     int       `json:"chunk_index"`
}

// ExtractedEdge represents a raw relationship extracted from a source.
// This type is used by both the factstore (for persistence) and the extraction
// pipeline (for intermediate results).
type ExtractedEdge struct {
	CreatedAt      time.Time `json:"created_at"`
	ID             string    `json:"id"`
	SourceID       string    `json:"source_id"`
	GroupID        string    `json:"group_id"`
	SourceNodeName string    `json:"source_node_name"`
	SourceNodeType string    `json:"source_node_type"`
	TargetNodeName string    `json:"target_node_name"`
	TargetNodeType string    `json:"target_node_type"`
	Relation       string    `json:"relation"`
	Description    string    `json:"description"`
	Embedding      []float32 `json:"embedding,omitempty"`
	Weight         float64   `json:"weight"`
	ChunkIndex     int       `json:"chunk_index"`
}

// ExtractionResults is returned when ExtractOnly=true in AddEpisodeOptions.
// It contains the raw extractions before graph modeling, allowing for custom
// processing, filtering, or alternative graph modeling logic.
type ExtractionResults struct {

	// Metadata contains additional extraction information
	Metadata *ExtractionMetadata `json:"metadata,omitempty"`
	// SourceID is the episode/document ID in the fact store
	SourceID string `json:"source_id"`

	// ExtractedNodes are raw entities before resolution/deduplication
	ExtractedNodes []*ExtractedNode `json:"extracted_nodes"`

	// ExtractedEdges are raw relationships before resolution
	ExtractedEdges []*ExtractedEdge `json:"extracted_edges"`

	// ChunkCount is the number of chunks the episode was split into
	ChunkCount int `json:"chunk_count"`

	// ExtractionTime is how long the extraction took
	ExtractionTime time.Duration `json:"extraction_time"`
}

// ExtractionMetadata contains additional information about the extraction process.
type ExtractionMetadata struct {
	// ModelUsed is the NLP model used for extraction
	ModelUsed string `json:"model_used,omitempty"`

	// EmbeddingModel is the model used for generating embeddings
	EmbeddingModel string `json:"embedding_model,omitempty"`

	// EmbeddingDimensions is the dimension of the embeddings
	EmbeddingDimensions int `json:"embedding_dimensions,omitempty"`

	// TotalTokens is the estimated token count processed
	TotalTokens int `json:"total_tokens,omitempty"`
}

// NodeCount returns the number of extracted nodes.
func (r *ExtractionResults) NodeCount() int {
	if r == nil {
		return 0
	}
	return len(r.ExtractedNodes)
}

// EdgeCount returns the number of extracted edges.
func (r *ExtractionResults) EdgeCount() int {
	if r == nil {
		return 0
	}
	return len(r.ExtractedEdges)
}

// IsEmpty returns true if no entities or relationships were extracted.
func (r *ExtractionResults) IsEmpty() bool {
	return r.NodeCount() == 0 && r.EdgeCount() == 0
}
