package predicato

import (
	"context"

	"github.com/soundprediction/predicato/pkg/factstore"
	"github.com/soundprediction/predicato/pkg/modeler"
	"github.com/soundprediction/predicato/pkg/types"
)

// This file defines focused interfaces that follow the Interface Segregation Principle (ISP).
// The main Predicato interface is composed from these smaller interfaces for backward compatibility.
//
// Benefits of Interface Segregation:
//
// 1. Easier Testing: Mock implementations only need to implement the methods being tested.
//    Example: A test for search functionality can mock only GraphQuerier.
//
// 2. Clear Dependencies: Code that accepts a focused interface documents its actual requirements.
//    Example: func HandleSearch(q GraphQuerier) clearly shows only query capabilities are needed.
//
// 3. Flexibility: New implementations can satisfy only the interfaces they need.
//    Example: A read-only client could implement GraphQuerier without episode management.
//
// Usage Pattern:
//
//	// Accept the focused interface when possible
//	func SearchGraph(querier GraphQuerier, query string) (*types.SearchResults, error) {
//	    return querier.Search(ctx, query, config)
//	}
//
//	// Existing code using Predicato continues to work
//	func ProcessKnowledgeGraph(p Predicato) {
//	    // p implements all focused interfaces
//	    SearchGraph(p, "query")  // Works because Predicato embeds GraphQuerier
//	}

// EpisodeManager provides operations for managing episodes in the knowledge graph.
// Use this interface when you only need to add, remove, or retrieve episodes.
type EpisodeManager interface {
	// Add processes and adds new episodes to the knowledge graph.
	// Episodes can be text, conversations, or any temporal data.
	// Options parameter is optional and can be nil for default behavior.
	Add(ctx context.Context, episodes []types.Episode, options *AddEpisodeOptions) (*types.AddBulkEpisodeResults, error)

	// AddEpisode processes and adds a single episode to the knowledge graph.
	// This is equivalent to the Python add_episode method.
	AddEpisode(ctx context.Context, episode types.Episode, options *AddEpisodeOptions) (*types.AddEpisodeResults, error)

	// RemoveEpisode removes an episode and its associated nodes and edges from the knowledge graph.
	RemoveEpisode(ctx context.Context, episodeUUID string) error

	// GetEpisodes retrieves recent episodes from the knowledge graph.
	GetEpisodes(ctx context.Context, groupID string, limit int) ([]*types.Node, error)

	// GetNodesAndEdgesByEpisode retrieves all nodes and edges associated with a specific episode.
	GetNodesAndEdgesByEpisode(ctx context.Context, episodeUUID string) ([]*types.Node, []*types.Edge, error)
}

// GraphQuerier provides read-only query operations on the knowledge graph.
// Use this interface when you only need to search or retrieve data without modifications.
type GraphQuerier interface {
	// Search performs hybrid search across the knowledge graph combining
	// semantic embeddings, keyword search, and graph traversal.
	Search(ctx context.Context, query string, config *types.SearchConfig) (*types.SearchResults, error)

	// GetNode retrieves a specific node from the knowledge graph.
	GetNode(ctx context.Context, nodeID string) (*types.Node, error)

	// GetEdge retrieves a specific edge from the knowledge graph.
	GetEdge(ctx context.Context, edgeID string) (*types.Edge, error)
}

// GraphMutator provides write operations on the knowledge graph.
// Use this interface when you need direct manipulation of the graph structure.
type GraphMutator interface {
	// AddTriplet adds a triplet (subject-predicate-object) directly to the knowledge graph.
	AddTriplet(ctx context.Context, sourceNode *types.Node, edge *types.Edge, targetNode *types.Node) (*types.AddTripletResults, error)

	// ClearGraph removes all nodes and edges from the knowledge graph for a specific group.
	ClearGraph(ctx context.Context, groupID string) error

	// UpdateCommunities updates community assignments after changes to the graph.
	UpdateCommunities(ctx context.Context, episodeUUID string, groupID string) ([]*types.Node, []*types.Edge, error)
}

// FactsManager provides operations for the two-phase ingestion pipeline.
// Use this interface when you want to decouple extraction from graph modeling.
type FactsManager interface {
	// GetFactStore returns the underlying fact store.
	GetFactStore() factstore.FactsDB

	// SearchFacts performs RAG search directly on the factstore without graph queries.
	// This is useful for simpler RAG use cases that don't need relationship traversal.
	SearchFacts(ctx context.Context, query string, config *types.SearchConfig) (*factstore.FactSearchResults, error)

	// ExtractToFacts extracts entities and relationships from an episode and stores them
	// in the fact store without promoting to the graph.
	ExtractToFacts(ctx context.Context, episode types.Episode, options *AddEpisodeOptions) (*types.ExtractionResults, error)

	// PromoteToGraph takes previously extracted facts from the fact store and promotes
	// them to the knowledge graph using the configured GraphModeler.
	PromoteToGraph(ctx context.Context, sourceID string, options *AddEpisodeOptions) (*types.AddEpisodeResults, error)
}

// GraphAdmin provides administrative operations for the knowledge graph.
// Use this interface for maintenance and configuration tasks.
type GraphAdmin interface {
	// CreateIndices creates database indices and constraints for optimal performance.
	CreateIndices(ctx context.Context) error

	// ValidateModeler tests a GraphModeler implementation with sample data.
	ValidateModeler(ctx context.Context, gm modeler.GraphModeler) (*modeler.ModelerValidationResult, error)

	// Close closes all connections and cleans up resources.
	Close(ctx context.Context) error
}

// Ensure Predicato interface composes all focused interfaces.
// This compile-time check ensures backward compatibility.
var _ interface {
	EpisodeManager
	GraphQuerier
	GraphMutator
	FactsManager
	GraphAdmin
} = (Predicato)(nil)
