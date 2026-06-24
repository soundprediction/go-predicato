package types

import (
	"time"

	"github.com/soundprediction/predicato/pkg/ruleschema"
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
	Model          string    `json:"model,omitempty"`
	Embedding      []float32 `json:"embedding"`
	ChunkIndex     int       `json:"chunk_index"`
}

// ExtractedTriple represents a knowledge triple extracted from a source.
// Combines entity-relationship data (subject, predicate, object with types)
// with contextual fields (condition, temporal, location, certainty, scope).
// All fields are flat columns — no nesting.
type ExtractedTriple struct {
	CreatedAt         time.Time `json:"created_at"`
	ID                string    `json:"id"`
	SourceID          string    `json:"source_id"`
	GroupID           string    `json:"group_id,omitempty"`
	Subject           string    `json:"subject"`
	SubjectType       string    `json:"subject_type,omitempty"`
	Predicate         string    `json:"predicate"`
	Object            string    `json:"object"`
	ObjectType        string    `json:"object_type,omitempty"`
	Description       string    `json:"description,omitempty"`
	Condition         string    `json:"condition,omitempty"`
	Temporal          string    `json:"temporal,omitempty"`
	Location          string    `json:"location,omitempty"`
	Certainty         string    `json:"certainty,omitempty"`
	Scope             string    `json:"scope,omitempty"`
	SourceAttribution string    `json:"source_attribution,omitempty"`
	Model             string    `json:"model,omitempty"`
	Embedding         []float32 `json:"embedding,omitempty"`
	Confidence        float64   `json:"confidence,omitempty"`
	ChunkIndex        int       `json:"chunk_index"`
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

	// ExtractedTriples are knowledge triples (subject-predicate-object with context)
	ExtractedTriples []*ExtractedTriple `json:"extracted_triples"`

	// ExtractedRules are conditional rules (extended extraction)
	ExtractedRules []*ExtractedRule `json:"extracted_rules,omitempty"`

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

// TripleCount returns the number of extracted triples.
func (r *ExtractionResults) TripleCount() int {
	if r == nil {
		return 0
	}
	return len(r.ExtractedTriples)
}

// IsEmpty returns true if no entities or triples were extracted.
func (r *ExtractionResults) IsEmpty() bool {
	return r.NodeCount() == 0 && r.TripleCount() == 0
}

// RuleCount returns the number of extracted rules.
func (r *ExtractionResults) RuleCount() int {
	if r == nil {
		return 0
	}
	return len(r.ExtractedRules)
}

// ExtractedRule represents a conditional rule extracted from a source:
// IF antecedent THEN consequent [UNLESS exception].
type ExtractedRule struct {
	CreatedAt         time.Time        `json:"created_at"`
	ID                string           `json:"id"`
	SourceID          string           `json:"source_id"`
	Antecedent        string           `json:"antecedent"`
	Consequent        string           `json:"consequent"`
	Exception         string           `json:"exception,omitempty"`
	RuleType          string           `json:"rule_type,omitempty"`
	Scope             string           `json:"scope,omitempty"`
	SourceAttribution string           `json:"source_attribution,omitempty"`
	Model             string           `json:"model,omitempty"`
	Embedding         []float32        `json:"embedding,omitempty"`
	Confidence        float64          `json:"confidence,omitempty"`
	ChunkIndex        int              `json:"chunk_index"`
	Structured        *ruleschema.Rule `json:"structured,omitempty"`
	StructureStatus   string           `json:"structure_status,omitempty"`
	StructureError    string           `json:"structure_error,omitempty"`
}
