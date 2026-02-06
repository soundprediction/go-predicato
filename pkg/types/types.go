package types

import (
	"errors"
	"time"
)

// Validation errors
var (
	ErrEmptyName    = errors.New("name cannot be empty")
	ErrEmptyGroupID = errors.New("group_id cannot be empty")
	ErrEmptyUUID    = errors.New("uuid cannot be empty")
	ErrEmptyID      = errors.New("id cannot be empty")
	ErrEmptyContent = errors.New("content cannot be empty")
	ErrInvalidLimit = errors.New("limit must be positive")
)

// Node represents a node in the knowledge graph.
type Node struct {
	CreatedAt time.Time `json:"created_at" mapstructure:"created_at"`
	UpdatedAt time.Time `json:"updated_at" mapstructure:"updated_at"`

	Reference time.Time `json:"reference,omitempty" mapstructure:"reference"`

	// Temporal fields
	ValidFrom time.Time              `json:"valid_from" mapstructure:"valid_from"`
	Metadata  map[string]interface{} `json:"metadata,omitempty" mapstructure:"metadata"`

	ValidTo *time.Time `json:"valid_to,omitempty" mapstructure:"valid_to"`

	Uuid    string   `json:"uuid" mapstructure:"uuid"`
	Name    string   `json:"name" mapstructure:"name"`
	Type    NodeType `json:"type" mapstructure:"type"`
	GroupID string   `json:"group_id" mapstructure:"group_id"`

	// Entity-specific fields
	EntityType string `json:"entity_type,omitempty" mapstructure:"entity_type"`
	Summary    string `json:"summary,omitempty" mapstructure:"summary"`

	// Episode-specific fields
	EpisodeType EpisodeType `json:"episode_type,omitempty" mapstructure:"episode_type"`
	Content     string      `json:"content,omitempty" mapstructure:"content"`
	EntityEdges []string    `json:"entity_edges,omitempty" mapstructure:"entity_edges"`

	// Common fields
	Embedding     []float32 `json:"embedding,omitempty" mapstructure:"embedding"`
	NameEmbedding []float32 `json:"name_embedding,omitempty" mapstructure:"name_embedding"`

	// Source tracking
	SourceIDs []string `json:"source_ids,omitempty" mapstructure:"source_ids"`

	// Community-specific fields
	Level int `json:"level,omitempty"`
}

// Validate checks if the Node has all required fields set.
func (n *Node) Validate() error {
	if n.Name == "" {
		return ErrEmptyName
	}
	if n.GroupID == "" {
		return ErrEmptyGroupID
	}
	return nil
}

// ValidateForCreate checks if the Node has all required fields for creation.
func (n *Node) ValidateForCreate() error {
	if n.Uuid == "" {
		return ErrEmptyUUID
	}
	return n.Validate()
}

// Edge is an alias for EntityEdge to maintain backward compatibility
// Use EntityEdge, EpisodicEdge, or CommunityEdge directly for type-safe operations
type Edge = EntityEdge

// NodeType represents the type of a node.
type NodeType string

const (
	// EntityNodeType represents entities extracted from content.
	EntityNodeType NodeType = "entity"
	// EpisodicNodeType represents episodic memories or events.
	EpisodicNodeType NodeType = "episodic"
	// CommunityNodeType represents communities of related entities.
	CommunityNodeType NodeType = "community"
	// SourceNodeType represents source nodes where content originates.
	SourceNodeType NodeType = "source"
)

// EdgeType and related constants are now defined in edge.go

// EpisodeType represents the type of an episode.
type EpisodeType string

const (
	// ConversationEpisodeType for conversational data.
	ConversationEpisodeType EpisodeType = "conversation"
	// DocumentEpisodeType for document content.
	DocumentEpisodeType EpisodeType = "document"
	// EventEpisodeType for events or actions.
	EventEpisodeType EpisodeType = "event"
)

// Episode represents a temporal data unit to be processed.
type Episode struct {
	ID               string
	Name             string
	Content          string
	Source           string // Source where the content originates (e.g., URL, file path, or identifier)
	Reference        time.Time
	CreatedAt        time.Time
	GroupID          string
	Metadata         map[string]interface{}
	ContentEmbedding []float32
}

// Validate checks if the Episode has all required fields set.
func (e *Episode) Validate() error {
	if e.Content == "" {
		return ErrEmptyContent
	}
	if e.GroupID == "" {
		return ErrEmptyGroupID
	}
	return nil
}

// ValidateForCreate checks if the Episode has all required fields for creation.
func (e *Episode) ValidateForCreate() error {
	if e.ID == "" {
		return ErrEmptyID
	}
	return e.Validate()
}

// SearchConfig holds configuration for search operations.
type SearchConfig struct {
	// Filters for constraining search results.
	Filters *SearchFilters
	// NodeConfig holds configuration for node search.
	NodeConfig *NodeSearchConfig
	// EdgeConfig holds configuration for edge search.
	EdgeConfig *EdgeSearchConfig
	// Limit is the maximum number of results to return.
	Limit int
	// CenterNodeDistance is the maximum distance from center nodes.
	CenterNodeDistance int
	// MinScore is the minimum relevance score for results.
	MinScore float64
	// IncludeEdges determines if edges should be included in results.
	IncludeEdges bool
	// Rerank determines if results should be reranked.
	Rerank bool
}

// Validate checks if the SearchConfig has valid values.
func (c *SearchConfig) Validate() error {
	if c.Limit < 0 {
		return ErrInvalidLimit
	}
	return nil
}

// WithDefaults returns a copy of the config with default values applied.
func (c *SearchConfig) WithDefaults() *SearchConfig {
	if c == nil {
		return &SearchConfig{
			Limit:              10,
			CenterNodeDistance: 2,
			MinScore:           0.0,
			IncludeEdges:       true,
		}
	}
	result := *c
	if result.Limit == 0 {
		result.Limit = 10
	}
	if result.CenterNodeDistance == 0 {
		result.CenterNodeDistance = 2
	}
	return &result
}

// NodeSearchConfig holds configuration for node search operations.
type NodeSearchConfig struct {
	// Reranker defines which reranking method to use.
	Reranker string
	// SearchMethods defines which search methods to use.
	SearchMethods []string
	// MinScore is the minimum score for results.
	MinScore float64
}

// EdgeSearchConfig holds configuration for edge search operations.
type EdgeSearchConfig struct {
	// Reranker defines which reranking method to use.
	Reranker string
	// SearchMethods defines which search methods to use.
	SearchMethods []string
	// MinScore is the minimum score for results.
	MinScore float64
}

// SearchFilters holds filters for search operations.
type SearchFilters struct {
	// TimeRange for temporal filtering.
	TimeRange *TimeRange
	// GroupIDs to include in search.
	GroupIDs []string
	// NodeTypes to include.
	NodeTypes []NodeType
	// EdgeTypes to include.
	EdgeTypes []EdgeType
	// EntityTypes to include.
	EntityTypes []string
}

// TimeRange represents a time range for filtering.
type TimeRange struct {
	Start time.Time
	End   time.Time
}

// SearchResults holds the results of a search operation.
type SearchResults struct {
	// Query used for the search.
	Query string
	// Nodes found in the search.
	Nodes []*Node
	// Edges found in the search.
	Edges []*Edge
	// Total number of results found (before limit).
	Total int
}

// ExtractedEntity represents an entity extracted from content.
type ExtractedEntity struct {
	Metadata map[string]string `json:"metadata"`
	Name     string            `json:"name"`
	Type     string            `json:"type"`
	Summary  string            `json:"summary"`
}

// ExtractedRelationship represents a relationship extracted from content.
type ExtractedRelationship struct {
	Metadata     map[string]string `json:"metadata"`
	SourceEntity string            `json:"source_entity"`
	TargetEntity string            `json:"target_entity"`
	Name         string            `json:"name"`
	Summary      string            `json:"summary"`
	Strength     float64           `json:"strength"`
}

// AddEpisodeResults represents the result of adding a single episode to the knowledge graph.
type AddEpisodeResults struct {
	// Episode is the episodic node that was created.
	Episode *Node `json:"episode"`
	// EpisodicEdges are the edges connecting the episode to entities.
	EpisodicEdges []*Edge `json:"episodic_edges"`
	// Nodes are the entity nodes that were extracted or updated.
	Nodes []*Node `json:"nodes"`
	// Edges are the entity relationships that were extracted or updated.
	Edges []*Edge `json:"edges"`
	// Communities are the community nodes that were created or updated.
	Communities []*Node `json:"communities"`
	// CommunityEdges are the edges connecting communities to entities.
	CommunityEdges []*Edge `json:"community_edges"`
}

// AddBulkEpisodeResults represents the result of adding multiple episodes to the knowledge graph.
type AddBulkEpisodeResults struct {
	// Episodes are the episodic nodes that were created.
	Episodes []*Node `json:"episodes"`
	// EpisodicEdges are the edges connecting episodes to entities.
	EpisodicEdges []*Edge `json:"episodic_edges"`
	// Nodes are the entity nodes that were extracted or updated.
	Nodes []*Node `json:"nodes"`
	// Edges are the entity relationships that were extracted or updated.
	Edges []*Edge `json:"edges"`
	// Communities are the community nodes that were created or updated.
	Communities []*Node `json:"communities"`
	// CommunityEdges are the edges connecting communities to entities.
	CommunityEdges []*Edge `json:"community_edges"`
}

// AddTripletResults represents the result of adding a triplet (subject-predicate-object) to the knowledge graph.
type AddTripletResults struct {
	// Nodes are the entity nodes that were created or updated (subject and object).
	Nodes []*Node `json:"nodes"`
	// Edges are the relationship edges that were created (predicate).
	Edges []*Edge `json:"edges"`
}

// EpisodeProcessingResult represents the result of processing a single episode.
// This is used internally during episode processing.
type EpisodeProcessingResult struct {
	// Episode is the processed episode node.
	Episode *Node `json:"episode"`
	// ExtractedEntities are the entities found in the episode.
	ExtractedEntities []*Node `json:"extracted_entities"`
	// ExtractedRelationships are the relationships found in the episode.
	ExtractedRelationships []*Edge `json:"extracted_relationships"`
	// EpisodicEdges connect the episode to the extracted entities.
	EpisodicEdges []*Edge `json:"episodic_edges"`
	// Errors encountered during processing.
	Errors []string `json:"errors,omitempty"`
	// ProcessingTime is the time taken to process this episode.
	ProcessingTime time.Duration `json:"processing_time"`
}

// BulkEpisodeResults represents the result of bulk episode processing.
type BulkEpisodeResults struct {
	// ProcessedEpisodes contains results for each processed episode.
	ProcessedEpisodes []*EpisodeProcessingResult `json:"processed_episodes"`
	// TotalProcessingTime is the total time for all episodes.
	TotalProcessingTime time.Duration `json:"total_processing_time"`
	// SuccessCount is the number of successfully processed episodes.
	SuccessCount int `json:"success_count"`
	// ErrorCount is the number of episodes that failed processing.
	ErrorCount int `json:"error_count"`
}

// TripletResults represents the result of triplet operations.
type TripletResults struct {
	// SubjectNode is the subject entity node.
	SubjectNode *Node `json:"subject_node"`
	// ObjectNode is the object entity node.
	ObjectNode *Node `json:"object_node"`
	// PredicateEdge is the relationship edge between subject and object.
	PredicateEdge *Edge `json:"predicate_edge"`
	// Created indicates if new nodes/edges were created (vs updated).
	Created bool `json:"created"`
}
