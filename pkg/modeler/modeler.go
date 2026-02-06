// Package modeler provides interfaces and implementations for graph modeling logic.
// It controls how raw extracted facts (entities and relationships) are transformed
// into a knowledge graph representation.
package modeler

import (
	"context"

	"github.com/soundprediction/predicato/pkg/types"
)

// GraphModeler defines the interface for custom graph modeling logic.
// Implementations control how raw extracted facts become graph nodes and edges.
//
// The default implementation (DefaultModeler) wraps the existing NodeOperations
// and EdgeOperations logic. Custom implementations can provide alternative
// entity resolution, relationship handling, and community detection strategies.
type GraphModeler interface {
	// ResolveEntities performs entity resolution on extracted nodes.
	// This includes deduplication, merging similar entities, and linking
	// to existing entities in the graph.
	//
	// Returns:
	//   - EntityResolutionOutput with resolved nodes and UUID mapping
	//   - error if resolution fails
	ResolveEntities(ctx context.Context, input *EntityResolutionInput) (*EntityResolutionOutput, error)

	// ResolveRelationships performs edge resolution using the resolved nodes.
	// This includes deduplication, merging similar relationships, and
	// updating edge endpoints to use resolved node UUIDs.
	//
	// Returns:
	//   - RelationshipResolutionOutput with resolved edges
	//   - error if resolution fails
	ResolveRelationships(ctx context.Context, input *RelationshipResolutionInput) (*RelationshipResolutionOutput, error)

	// BuildCommunities detects communities among resolved entities.
	// Return nil output to skip community detection.
	//
	// Returns:
	//   - CommunityOutput with detected communities and edges (or nil to skip)
	//   - error if community detection fails
	BuildCommunities(ctx context.Context, input *CommunityInput) (*CommunityOutput, error)
}

// EntityResolutionInput contains input data for entity resolution.
type EntityResolutionInput struct {

	// Episode is the main episode node being processed
	Episode *types.Node

	// EntityTypes defines the entity type schema
	EntityTypes map[string]interface{}

	// Options contains resolution configuration
	Options *EntityResolutionOptions

	// GroupID is the tenant/group identifier
	GroupID string

	// ExtractedNodes are the raw entities extracted from the episode
	ExtractedNodes []*types.Node

	// PreviousEpisodes provides context from prior episodes
	PreviousEpisodes []*types.Node
}

// EntityResolutionOptions configures entity resolution behavior.
type EntityResolutionOptions struct {
	// SkipResolution disables entity deduplication/merging
	SkipResolution bool

	// SkipReflexion disables NLP-based entity refinement
	SkipReflexion bool

	// SkipAttributes disables attribute extraction
	SkipAttributes bool

	// SimilarityThreshold for entity matching (default: 0.85)
	SimilarityThreshold float64
}

// EntityResolutionOutput contains the results of entity resolution.
type EntityResolutionOutput struct {

	// UUIDMap maps original extraction UUID -> resolved node UUID
	// Used to update edge endpoints after entity resolution
	UUIDMap map[string]string

	// ResolvedNodes are the deduplicated/merged nodes ready for the graph
	ResolvedNodes []*types.Node

	// MergedCount is the number of entities that were merged into existing ones
	MergedCount int

	// NewCount is the number of genuinely new entities
	NewCount int
}

// RelationshipResolutionInput contains input data for relationship resolution.
type RelationshipResolutionInput struct {

	// UUIDMap maps original extraction UUID -> resolved node UUID
	UUIDMap map[string]string

	// Episode is the main episode node being processed
	Episode *types.Node

	// EdgeTypes defines the relationship type schema
	EdgeTypes map[string]interface{}

	// Options contains resolution configuration
	Options *RelationshipResolutionOptions

	// GroupID is the tenant/group identifier
	GroupID string

	// ExtractedEdges are the raw relationships extracted from the episode
	ExtractedEdges []*types.Edge

	// ResolvedNodes are the entities after resolution
	ResolvedNodes []*types.Node
}

// RelationshipResolutionOptions configures relationship resolution behavior.
type RelationshipResolutionOptions struct {
	// SkipEdgeResolution disables edge deduplication
	SkipEdgeResolution bool
}

// RelationshipResolutionOutput contains the results of relationship resolution.
type RelationshipResolutionOutput struct {
	// ResolvedEdges are the deduplicated edges ready for the graph
	ResolvedEdges []*types.Edge

	// EpisodicEdges connect entities to the episode
	EpisodicEdges []*types.Edge

	// NewCount is the number of genuinely new relationships
	NewCount int

	// UpdatedCount is the number of existing relationships that were updated
	UpdatedCount int
}

// CommunityInput contains input data for community detection.
type CommunityInput struct {

	// GroupID is the tenant/group identifier
	GroupID string

	// EpisodeID is the episode that triggered this update
	EpisodeID string
	// Nodes are the resolved entities to cluster
	Nodes []*types.Node

	// Edges are the relationships between entities
	Edges []*types.Edge
}

// CommunityOutput contains the results of community detection.
type CommunityOutput struct {
	// Communities are the detected community nodes
	Communities []*types.Node

	// CommunityEdges connect entities to their communities
	CommunityEdges []*types.Edge
}

// ModelerValidationResult contains the results of validating a GraphModeler.
type ModelerValidationResult struct {

	// EntityResolution contains the result of testing ResolveEntities
	EntityResolution *ValidationStepResult `json:"entity_resolution"`

	// RelationshipResolution contains the result of testing ResolveRelationships
	RelationshipResolution *ValidationStepResult `json:"relationship_resolution"`

	// CommunityDetection contains the result of testing BuildCommunities
	CommunityDetection *ValidationStepResult `json:"community_detection"`

	// Warnings contains non-fatal issues found during validation
	Warnings []string `json:"warnings,omitempty"`
	// Valid is true if all required methods work correctly
	Valid bool `json:"valid"`
}

// ValidationStepResult contains the result of validating a single modeler method.
type ValidationStepResult struct {

	// Error contains any error returned by the method
	Error error `json:"error,omitempty"`

	// Latency is how long the method took to execute
	Latency int64 `json:"latency_ms"`
	// Passed is true if the method executed without error
	Passed bool `json:"passed"`
}
