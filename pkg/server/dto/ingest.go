package dto

import (
	"errors"
	"fmt"
	"strings"
	"time"
)

// Validation errors
var (
	ErrEmptyGroupID      = errors.New("group_id cannot be empty")
	ErrEmptyMessages     = errors.New("messages cannot be empty")
	ErrEmptyName         = errors.New("name cannot be empty")
	ErrGroupIDTooLong    = errors.New("group_id exceeds maximum length (256)")
	ErrNameTooLong       = errors.New("name exceeds maximum length (1024)")
	ErrContentTooLong    = errors.New("content exceeds maximum length (1MB)")
	ErrInvalidCharacters = errors.New("field contains invalid characters")
)

// MaxFieldLengths defines maximum lengths for fields to prevent abuse
const (
	MaxGroupIDLength  = 256
	MaxNameLength     = 1024
	MaxContentLength  = 1024 * 1024 // 1MB
	MaxMessagesCount  = 1000
	MaxEntityType     = 256
	MaxAttributeCount = 100
)

// AddMessagesRequest represents a request to add messages to the knowledge graph
type AddMessagesRequest struct {
	Reference *time.Time `json:"reference,omitempty"`
	GroupID   string     `json:"group_id" binding:"required"`
	Messages  []Message  `json:"messages" binding:"required,dive"`
}

// Validate performs validation on AddMessagesRequest
func (r *AddMessagesRequest) Validate() error {
	if strings.TrimSpace(r.GroupID) == "" {
		return ErrEmptyGroupID
	}
	if len(r.GroupID) > MaxGroupIDLength {
		return ErrGroupIDTooLong
	}
	if len(r.Messages) == 0 {
		return ErrEmptyMessages
	}
	if len(r.Messages) > MaxMessagesCount {
		return errors.New("messages count exceeds maximum (1000)")
	}
	for i, msg := range r.Messages {
		if err := msg.Validate(); err != nil {
			return fmt.Errorf("message %d: %w", i, err)
		}
	}
	return nil
}

// AddEntityNodeRequest represents a request to add an entity node
type AddEntityNodeRequest struct {
	Attributes map[string]interface{} `json:"attributes,omitempty"`
	GroupID    string                 `json:"group_id" binding:"required"`
	Name       string                 `json:"name" binding:"required"`
	EntityType string                 `json:"entity_type,omitempty"`
}

// Validate performs validation on AddEntityNodeRequest
func (r *AddEntityNodeRequest) Validate() error {
	if strings.TrimSpace(r.GroupID) == "" {
		return ErrEmptyGroupID
	}
	if len(r.GroupID) > MaxGroupIDLength {
		return ErrGroupIDTooLong
	}
	if strings.TrimSpace(r.Name) == "" {
		return ErrEmptyName
	}
	if len(r.Name) > MaxNameLength {
		return ErrNameTooLong
	}
	if len(r.EntityType) > MaxEntityType {
		return errors.New("entity_type exceeds maximum length (256)")
	}
	if len(r.Attributes) > MaxAttributeCount {
		return errors.New("attributes count exceeds maximum (100)")
	}
	return nil
}

// ClearDataRequest represents a request to clear graph data
type ClearDataRequest struct {
	GroupIDs []string `json:"group_ids,omitempty"`
}

// Validate performs validation on ClearDataRequest
func (r *ClearDataRequest) Validate() error {
	for _, groupID := range r.GroupIDs {
		if len(groupID) > MaxGroupIDLength {
			return ErrGroupIDTooLong
		}
	}
	return nil
}

// IngestResponse represents a response from ingest operations
type IngestResponse struct {
	Message   string `json:"message,omitempty"`
	ProcessID string `json:"process_id,omitempty"`
	Success   bool   `json:"success"`
}

// ExtractEpisodeRequest represents a request to extract entities from an episode
// without promoting to the graph (first stage of two-stage ingestion)
type ExtractEpisodeRequest struct {
	// Episode fields
	ID        string                 `json:"id,omitempty"`
	Name      string                 `json:"name" binding:"required"`
	Content   string                 `json:"content" binding:"required"`
	Source    string                 `json:"source,omitempty"`
	GroupID   string                 `json:"group_id" binding:"required"`
	Reference *time.Time             `json:"reference,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`

	// Extraction options
	EntityTypes          map[string]interface{}            `json:"entity_types,omitempty"`
	ExcludedEntityTypes  []string                          `json:"excluded_entity_types,omitempty"`
	EdgeTypes            map[string]interface{}            `json:"edge_types,omitempty"`
	EdgeTypeMap          map[string]map[string]interface{} `json:"edge_type_map,omitempty"`
	PreviousEpisodeUUIDs []string                          `json:"previous_episode_uuids,omitempty"`
	MaxCharacters        int                               `json:"max_characters,omitempty"`
	UseYAML              bool                              `json:"use_yaml,omitempty"`
}

// Validate performs validation on ExtractEpisodeRequest
func (r *ExtractEpisodeRequest) Validate() error {
	if strings.TrimSpace(r.GroupID) == "" {
		return ErrEmptyGroupID
	}
	if len(r.GroupID) > MaxGroupIDLength {
		return ErrGroupIDTooLong
	}
	if strings.TrimSpace(r.Name) == "" {
		return ErrEmptyName
	}
	if len(r.Name) > MaxNameLength {
		return ErrNameTooLong
	}
	if strings.TrimSpace(r.Content) == "" {
		return errors.New("content cannot be empty")
	}
	if len(r.Content) > MaxContentLength {
		return ErrContentTooLong
	}
	return nil
}

// ExtractedNodeDTO represents an extracted entity node in API responses
type ExtractedNodeDTO struct {
	CreatedAt   time.Time `json:"created_at"`
	ID          string    `json:"id"`
	SourceID    string    `json:"source_id"`
	GroupID     string    `json:"group_id"`
	Name        string    `json:"name"`
	Type        string    `json:"type"`
	Description string    `json:"description"`
	ChunkIndex  int       `json:"chunk_index"`
}

// ExtractedTripleDTO represents an extracted knowledge triple in API responses
type ExtractedTripleDTO struct {
	CreatedAt         time.Time `json:"created_at"`
	ID                string    `json:"id"`
	SourceID          string    `json:"source_id"`
	GroupID           string    `json:"group_id"`
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
	Confidence        float64   `json:"confidence,omitempty"`
	ChunkIndex        int       `json:"chunk_index"`
}

// ExtractionMetadataDTO contains metadata about the extraction process
type ExtractionMetadataDTO struct {
	ModelUsed           string `json:"model_used,omitempty"`
	EmbeddingModel      string `json:"embedding_model,omitempty"`
	EmbeddingDimensions int    `json:"embedding_dimensions,omitempty"`
	TotalTokens         int    `json:"total_tokens,omitempty"`
}

// ExtractEpisodeResponse represents the response from extraction
type ExtractEpisodeResponse struct {
	Metadata       *ExtractionMetadataDTO `json:"metadata,omitempty"`
	SourceID       string                 `json:"source_id"`
	ExtractionTime string                 `json:"extraction_time"`
	ExtractedNodes []ExtractedNodeDTO     `json:"extracted_nodes"`
	ExtractedTriples []ExtractedTripleDTO   `json:"extracted_triples"`
	ChunkCount     int                    `json:"chunk_count"`
	Success        bool                   `json:"success"`
}

// PromoteToGraphRequest represents a request to promote extracted knowledge to the graph
// (second stage of two-stage ingestion)
type PromoteToGraphRequest struct {
	SourceID string `json:"source_id" binding:"required"`

	// Resolution options
	EntityTypes          map[string]interface{}            `json:"entity_types,omitempty"`
	EdgeTypes            map[string]interface{}            `json:"edge_types,omitempty"`
	EdgeTypeMap          map[string]map[string]interface{} `json:"edge_type_map,omitempty"`
	PreviousEpisodeUUIDs []string                          `json:"previous_episode_uuids,omitempty"`

	// Skip options for faster processing
	SkipReflexion      bool `json:"skip_reflexion,omitempty"`
	SkipResolution     bool `json:"skip_resolution,omitempty"`
	SkipAttributes     bool `json:"skip_attributes,omitempty"`
	SkipEdgeResolution bool `json:"skip_edge_resolution,omitempty"`
	UseYAML            bool `json:"use_yaml,omitempty"`
}

// Validate performs validation on PromoteToGraphRequest
func (r *PromoteToGraphRequest) Validate() error {
	if strings.TrimSpace(r.SourceID) == "" {
		return errors.New("source_id cannot be empty")
	}
	return nil
}

// NodeDTO represents a node in API responses
type NodeDTO struct {
	CreatedAt  time.Time              `json:"created_at"`
	UpdatedAt  *time.Time             `json:"updated_at,omitempty"`
	ValidFrom  *time.Time             `json:"valid_from,omitempty"`
	ValidTo    *time.Time             `json:"valid_to,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	UUID       string                 `json:"uuid"`
	GroupID    string                 `json:"group_id"`
	Name       string                 `json:"name"`
	Type       string                 `json:"type"`
	EntityType string                 `json:"entity_type,omitempty"`
	Summary    string                 `json:"summary,omitempty"`
	SourceIDs  []string               `json:"source_ids,omitempty"`
}

// EdgeDTO represents an edge in API responses
type EdgeDTO struct {
	CreatedAt      time.Time              `json:"created_at"`
	ExpiredAt      *time.Time             `json:"expired_at,omitempty"`
	ValidAt        *time.Time             `json:"valid_at,omitempty"`
	InvalidAt      *time.Time             `json:"invalid_at,omitempty"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
	UUID           string                 `json:"uuid"`
	GroupID        string                 `json:"group_id"`
	SourceNodeUUID string                 `json:"source_node_uuid"`
	TargetNodeUUID string                 `json:"target_node_uuid"`
	Type           string                 `json:"type"`
	Name           string                 `json:"name,omitempty"`
	Fact           string                 `json:"fact,omitempty"`
	Episodes       []string               `json:"episodes,omitempty"`
}

// PromoteToGraphResponse represents the response from promoting to graph
type PromoteToGraphResponse struct {
	Episode        *NodeDTO  `json:"episode,omitempty"`
	EpisodicEdges  []EdgeDTO `json:"episodic_edges"`
	Nodes          []NodeDTO `json:"nodes"`
	Edges          []EdgeDTO `json:"edges"`
	Communities    []NodeDTO `json:"communities"`
	CommunityEdges []EdgeDTO `json:"community_edges"`
	Success        bool      `json:"success"`
}
