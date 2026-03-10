package driver

import (
	"context"
	"time"

	"github.com/soundprediction/predicato/pkg/types"
)

// This file defines focused interfaces that follow the Interface Segregation Principle (ISP).
// The main GraphDriver interface is composed from these smaller interfaces for backward compatibility.
//
// Benefits of Interface Segregation:
//
// 1. Easier Testing: Mock implementations only need to implement the methods being tested.
//    Example: A test for search functionality can mock only GraphSearcher.
//
// 2. Clear Dependencies: Code that accepts a focused interface documents its actual requirements.
//    Example: func ProcessSearch(searcher GraphSearcher) clearly shows only search is needed.
//
// 3. Flexibility: New implementations can satisfy only the interfaces they need.
//    Example: A read-only cache could implement GraphSearcher without write operations.
//
// Usage Pattern:
//
//	// Accept the focused interface when possible
//	func SearchNodes(searcher GraphSearcher, query string) ([]*types.Node, error) {
//	    return searcher.SearchNodes(ctx, query, groupID, opts)
//	}
//
//	// Existing code using GraphDriver continues to work
//	func ProcessGraph(driver GraphDriver) {
//	    // driver implements all focused interfaces
//	    SearchNodes(driver, "query")  // Works because GraphDriver embeds GraphSearcher
//	}

// GraphCore provides core database operations that all graph drivers must implement.
// This includes session management, query execution, and lifecycle management.
type GraphCore interface {
	// ExecuteQuery executes a Cypher query with parameters.
	// Returns (results, summary, keys, error) matching the Python interface.
	ExecuteQuery(ctx context.Context, cypherQuery string, kwargs map[string]any) (any, any, any, error)

	// Session returns a database session for transaction management.
	Session(database *string) GraphDriverSession

	// Close releases all resources held by the driver.
	Close() error

	// DeleteAllIndexes removes all indexes from the database.
	DeleteAllIndexes(database string)

	// Provider returns the type of graph database provider.
	Provider() GraphProvider

	// GetAossClient returns the AOSS client if available, nil otherwise.
	GetAossClient() any
}

// NodeStore provides operations for managing nodes in the graph.
type NodeStore interface {
	// GetNode retrieves a single node by ID.
	GetNode(ctx context.Context, nodeID, groupID string) (*types.Node, error)

	// GetNodes retrieves multiple nodes by their IDs.
	GetNodes(ctx context.Context, nodeIDs []string, groupID string) ([]*types.Node, error)

	// UpsertNode creates or updates a node.
	UpsertNode(ctx context.Context, node *types.Node) error

	// UpsertNodes creates or updates multiple nodes.
	UpsertNodes(ctx context.Context, nodes []*types.Node) error

	// DeleteNode removes a node from the graph.
	DeleteNode(ctx context.Context, nodeID, groupID string) error

	// GetEntityNodesByGroup retrieves all entity nodes for a group.
	GetEntityNodesByGroup(ctx context.Context, groupID string) ([]*types.Node, error)

	// ParseNodesFromRecords parses database records into Node objects.
	ParseNodesFromRecords(records any) ([]*types.Node, error)
}

// EdgeStore provides operations for managing edges in the graph.
type EdgeStore interface {
	// GetEdge retrieves a single edge by ID.
	GetEdge(ctx context.Context, edgeID, groupID string) (*types.Edge, error)

	// GetEdges retrieves multiple edges by their IDs.
	GetEdges(ctx context.Context, edgeIDs []string, groupID string) ([]*types.Edge, error)

	// UpsertEdge creates or updates an edge.
	UpsertEdge(ctx context.Context, edge *types.Edge) error

	// UpsertEdges creates or updates multiple edges.
	UpsertEdges(ctx context.Context, edges []*types.Edge) error

	// UpsertEpisodicEdge creates an edge between an episode and an entity.
	UpsertEpisodicEdge(ctx context.Context, episodeUUID, entityUUID, groupID string) error

	// UpsertCommunityEdge creates an edge between a community and a node.
	UpsertCommunityEdge(ctx context.Context, communityUUID, nodeUUID, uuid, groupID string) error

	// DeleteEdge removes an edge from the graph.
	DeleteEdge(ctx context.Context, edgeID, groupID string) error

	// GetBetweenNodes retrieves edges between two specific nodes.
	GetBetweenNodes(ctx context.Context, sourceNodeID, targetNodeID string) ([]*types.Edge, error)
}

// GraphTraversal provides operations for navigating the graph structure.
type GraphTraversal interface {
	// GetNeighbors retrieves neighboring nodes within a specified distance.
	GetNeighbors(ctx context.Context, nodeID, groupID string, maxDistance int) ([]*types.Node, error)

	// GetRelatedNodes retrieves nodes connected by specific edge types.
	GetRelatedNodes(ctx context.Context, nodeID, groupID string, edgeTypes []types.EdgeType) ([]*types.Node, error)

	// GetNodeNeighbors retrieves immediate neighbors with relationship info.
	GetNodeNeighbors(ctx context.Context, nodeUUID, groupID string) ([]types.Neighbor, error)
}

// GraphSearcher provides search operations across the graph.
// Use this interface when you only need search capabilities.
type GraphSearcher interface {
	// SearchNodes performs text-based search on nodes.
	SearchNodes(ctx context.Context, query, groupID string, options *SearchOptions) ([]*types.Node, error)

	// SearchEdges performs text-based search on edges.
	SearchEdges(ctx context.Context, query, groupID string, options *SearchOptions) ([]*types.Edge, error)

	// SearchNodesByVector performs vector similarity search on nodes.
	SearchNodesByVector(ctx context.Context, vector []float32, groupID string, options *VectorSearchOptions) ([]*types.Node, error)

	// SearchEdgesByVector performs vector similarity search on edges.
	SearchEdgesByVector(ctx context.Context, vector []float32, groupID string, options *VectorSearchOptions) ([]*types.Edge, error)

	// SearchNodesByEmbedding performs embedding-based search on nodes.
	SearchNodesByEmbedding(ctx context.Context, embedding []float32, groupID string, limit int) ([]*types.Node, error)

	// SearchEdgesByEmbedding performs embedding-based search on edges.
	SearchEdgesByEmbedding(ctx context.Context, embedding []float32, groupID string, limit int) ([]*types.Edge, error)
}

// TemporalOperations provides operations for time-based queries.
type TemporalOperations interface {
	// GetNodesInTimeRange retrieves nodes within a time range.
	GetNodesInTimeRange(ctx context.Context, start, end time.Time, groupID string) ([]*types.Node, error)

	// GetEdgesInTimeRange retrieves edges within a time range.
	GetEdgesInTimeRange(ctx context.Context, start, end time.Time, groupID string) ([]*types.Edge, error)

	// RetrieveEpisodes retrieves episodes based on time and type filters.
	RetrieveEpisodes(ctx context.Context, referenceTime time.Time, groupIDs []string, limit int, episodeType *types.EpisodeType) ([]*types.Node, error)
}

// CommunityOperations provides operations for community detection and management.
type CommunityOperations interface {
	// GetCommunities retrieves all communities at a specific level.
	GetCommunities(ctx context.Context, groupID string, level int) ([]*types.Node, error)

	// BuildCommunities triggers community detection for a group.
	BuildCommunities(ctx context.Context, groupID string) error

	// GetExistingCommunity retrieves the community containing an entity.
	GetExistingCommunity(ctx context.Context, entityUUID string) (*types.Node, error)

	// FindModalCommunity finds the most relevant community for an entity.
	FindModalCommunity(ctx context.Context, entityUUID string) (*types.Node, error)

	// RemoveCommunities removes all community nodes and edges.
	RemoveCommunities(ctx context.Context) error
}

// DatabaseAdmin provides administrative operations for database maintenance.
type DatabaseAdmin interface {
	// CreateIndices creates database indices for optimal performance.
	CreateIndices(ctx context.Context) error

	// GetStats retrieves statistics about the graph.
	GetStats(ctx context.Context, groupID string) (*GraphStats, error)

	// GetAllGroupIDs retrieves all unique group IDs in the database.
	GetAllGroupIDs(ctx context.Context) ([]string, error)
}

// Ensure GraphDriver implements all focused interfaces.
// This compile-time check ensures backward compatibility.
var _ interface {
	GraphCore
	NodeStore
	EdgeStore
	GraphTraversal
	GraphSearcher
	TemporalOperations
	CommunityOperations
	DatabaseAdmin
} = (GraphDriver)(nil)
