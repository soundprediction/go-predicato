package driver

import (
	"context"
	"fmt"
	"time"

	"github.com/soundprediction/predicato/pkg/types"
)

// GraphProvider is an alias to types.GraphProvider for backward compatibility.
// The canonical definition is in pkg/types/edge.go.
type GraphProvider = types.GraphProvider

// GraphProvider constants are re-exported from types for backward compatibility.
const (
	GraphProviderNeo4j    = types.GraphProviderNeo4j
	GraphProviderMemgraph = types.GraphProviderMemgraph
	GraphProviderFalkorDB = types.GraphProviderFalkorDB
	GraphProviderLadybug  = types.GraphProviderLadybug
	GraphProviderNeptune  = types.GraphProviderNeptune
	GraphProviderCozo     = types.GraphProviderCozo
	GraphProviderDuckPGQ  = types.GraphProviderDuckPGQ
)

// GraphDriverSession defines the interface for database sessions (matching Python GraphDriverSession)
type GraphDriverSession interface {
	// Session management
	Enter(ctx context.Context) (GraphDriverSession, error)
	Exit(ctx context.Context, excType, excVal, excTb interface{}) error
	Close() error

	// Query execution
	Run(ctx context.Context, query interface{}, kwargs map[string]interface{}) error
	ExecuteWrite(ctx context.Context, fn func(context.Context, GraphDriverSession, ...interface{}) (interface{}, error), args ...interface{}) (interface{}, error)

	// Provider info
	Provider() GraphProvider
}

// GraphDriver defines the interface for graph database operations (matching Python GraphDriver)
type GraphDriver interface {
	// Core methods matching Python interface
	ExecuteQuery(ctx context.Context, cypherQuery string, kwargs map[string]interface{}) (interface{}, interface{}, interface{}, error)
	Session(database *string) GraphDriverSession
	Close() error
	DeleteAllIndexes(database string)
	Provider() GraphProvider
	GetAossClient() interface{}

	// Database-specific extensions (these can remain for compatibility)
	// Node operations
	GetNode(ctx context.Context, nodeID, groupID string) (*types.Node, error)
	UpsertNode(ctx context.Context, node *types.Node) error
	DeleteNode(ctx context.Context, nodeID, groupID string) error
	GetNodes(ctx context.Context, nodeIDs []string, groupID string) ([]*types.Node, error)

	// Edge operations
	GetEdge(ctx context.Context, edgeID, groupID string) (*types.Edge, error)
	UpsertEdge(ctx context.Context, edge *types.Edge) error
	UpsertEpisodicEdge(ctx context.Context, episodeUUID, entityUUID, groupID string) error
	UpsertCommunityEdge(ctx context.Context, communityUUID, nodeUUID, uuid, groupID string) error
	DeleteEdge(ctx context.Context, edgeID, groupID string) error
	GetEdges(ctx context.Context, edgeIDs []string, groupID string) ([]*types.Edge, error)

	// Graph traversal operations
	GetNeighbors(ctx context.Context, nodeID, groupID string, maxDistance int) ([]*types.Node, error)
	GetRelatedNodes(ctx context.Context, nodeID, groupID string, edgeTypes []types.EdgeType) ([]*types.Node, error)
	GetNodeNeighbors(ctx context.Context, nodeUUID, groupID string) ([]types.Neighbor, error)
	// GetBetweenNodes retrieves edges between two specific nodes using the proper query pattern
	GetBetweenNodes(ctx context.Context, sourceNodeID, targetNodeID string) ([]*types.Edge, error)

	// Search operations
	SearchNodesByEmbedding(ctx context.Context, embedding []float32, groupID string, limit int) ([]*types.Node, error)
	SearchEdgesByEmbedding(ctx context.Context, embedding []float32, groupID string, limit int) ([]*types.Edge, error)
	SearchNodes(ctx context.Context, query, groupID string, options *SearchOptions) ([]*types.Node, error)
	SearchEdges(ctx context.Context, query, groupID string, options *SearchOptions) ([]*types.Edge, error)
	SearchNodesByVector(ctx context.Context, vector []float32, groupID string, options *VectorSearchOptions) ([]*types.Node, error)
	SearchEdgesByVector(ctx context.Context, vector []float32, groupID string, options *VectorSearchOptions) ([]*types.Edge, error)

	// Bulk operations
	UpsertNodes(ctx context.Context, nodes []*types.Node) error
	UpsertEdges(ctx context.Context, edges []*types.Edge) error

	// Temporal operations
	GetNodesInTimeRange(ctx context.Context, start, end time.Time, groupID string) ([]*types.Node, error)
	GetEdgesInTimeRange(ctx context.Context, start, end time.Time, groupID string) ([]*types.Edge, error)
	RetrieveEpisodes(ctx context.Context, referenceTime time.Time, groupIDs []string, limit int, episodeType *types.EpisodeType) ([]*types.Node, error)

	// Community operations
	GetCommunities(ctx context.Context, groupID string, level int) ([]*types.Node, error)
	BuildCommunities(ctx context.Context, groupID string) error
	GetExistingCommunity(ctx context.Context, entityUUID string) (*types.Node, error)
	FindModalCommunity(ctx context.Context, entityUUID string) (*types.Node, error)
	RemoveCommunities(ctx context.Context) error

	// Database maintenance
	CreateIndices(ctx context.Context) error
	GetStats(ctx context.Context, groupID string) (*GraphStats, error)

	// Parsing
	ParseNodesFromRecords(records any) ([]*types.Node, error)
	// ParseEdgesFromRecords(records any) ([]*types.Edge, error)

	// Getters by group
	GetEntityNodesByGroup(ctx context.Context, groupID string) ([]*types.Node, error)
	GetAllGroupIDs(ctx context.Context) ([]string, error)
}

// GraphStats holds statistics about the graph.
type GraphStats struct {
	LastUpdated    time.Time        `json:"last_updated"`
	NodesByType    map[string]int64 `json:"nodes_by_type"`
	EdgesByType    map[string]int64 `json:"edges_by_type"`
	NodeCount      int64            `json:"node_count"`
	EdgeCount      int64            `json:"edge_count"`
	CommunityCount int64            `json:"community_count"`
}

// QueryOptions holds options for database queries.
type QueryOptions struct {
	Filters   map[string]interface{}
	SortBy    string
	SortOrder string
	Limit     int
	Offset    int
}

// SearchOptions holds options for text-based search operations.
type SearchOptions struct {
	TimeRange   *types.TimeRange `json:"time_range,omitempty"`
	NodeTypes   []types.NodeType `json:"node_types,omitempty"`
	EdgeTypes   []types.EdgeType `json:"edge_types,omitempty"`
	Limit       int              `json:"limit"`
	UseFullText bool             `json:"use_fulltext"`
	ExactMatch  bool             `json:"exact_match"`
}

// VectorSearchOptions holds options for vector similarity search operations.
type VectorSearchOptions struct {
	TimeRange *types.TimeRange `json:"time_range,omitempty"`
	NodeTypes []types.NodeType `json:"node_types,omitempty"`
	EdgeTypes []types.EdgeType `json:"edge_types,omitempty"`
	Limit     int              `json:"limit"`
	MinScore  float64          `json:"min_score"`
}

// ParquetTopicFilter restricts graph loading to triples/rules semantically similar
// to a pre-computed topic embedding vector.
type ParquetTopicFilter struct {
	// Embedding is the topic vector. Length must equal the driver's embeddingDim.
	Embedding []float32

	// PostgresConnStr, when non-empty, reads source data from PostgreSQL instead
	// of parquet files. The connection string is a libpq-style DSN
	// (e.g. "host=localhost port=5432 dbname=glancedb user=admin password=pass").
	PostgresConnStr string

	// Threshold is the minimum cosine similarity [0, 1] for node embedding inclusion.
	// A value of 0.0 includes all nodes with non-null embeddings.
	Threshold float64

	// TripleThreshold is the minimum cosine similarity for triple/rule embedding inclusion.
	// When 0, falls back to Threshold.
	TripleThreshold float64

	// MaxNodes caps the number of nodes inserted (0 = unlimited).
	MaxNodes int64

	// MaxEdges caps the number of edges (triples + rules combined) inserted (0 = unlimited).
	MaxEdges int64

	// EdgeThreshold is the minimum cosine similarity for expansion edges
	// (edges collected during neighbor expansion that were not in the filtered set).
	// When 0, defaults to 0.4.
	EdgeThreshold float64

	// EdgeBatchSize is the number of triples to INSERT per batch when joining
	// against the entities table. Smaller batches reduce peak memory.
	// When 0, defaults to 10000.
	EdgeBatchSize int64
}

// FilteredParquetImporter is implemented by drivers that support topic-filtered parquet import.
type FilteredParquetImporter interface {
	BulkLoadFromParquetWithFilter(ctx context.Context, inputDir, groupID string, filter *ParquetTopicFilter) (int64, int64, int64, error)
}

// convertRecordToEdge converts a database record to an Edge object
func convertRecordToEdge(record map[string]interface{}) (*types.Edge, error) {
	edge := &types.Edge{}

	// Extract basic fields
	if uuid, ok := record["uuid"].(string); ok {
		edge.Uuid = uuid
	} else {
		return nil, fmt.Errorf("missing or invalid uuid field")
	}

	if name, ok := record["name"].(string); ok {
		edge.Name = name
	}

	if fact, ok := record["fact"].(string); ok {
		edge.Summary = fact
	}

	if groupID, ok := record["group_id"].(string); ok {
		edge.GroupID = groupID
	}

	// Extract source and target IDs
	if sourceID, ok := record["source_id"].(string); ok {
		edge.SourceID = sourceID
	}
	if targetID, ok := record["target_id"].(string); ok {
		edge.TargetID = targetID
	}

	// Extract timestamps
	if createdAt, ok := record["created_at"].(time.Time); ok {
		edge.CreatedAt = createdAt
	}
	if updatedAt, ok := record["updated_at"].(time.Time); ok {
		edge.UpdatedAt = updatedAt
	}
	if validFrom, ok := record["valid_from"].(time.Time); ok {
		edge.ValidFrom = validFrom
	}
	if validTo, ok := record["valid_to"].(time.Time); ok {
		edge.ValidTo = &validTo
	}

	// Set edge type - assume EntityEdge for relationships from RelatesToNode_
	edge.Type = types.EntityEdgeType

	// Extract source IDs if present
	if sourceIDs, ok := record["source_ids"].([]interface{}); ok {
		strSourceIDs := make([]string, len(sourceIDs))
		for i, id := range sourceIDs {
			if strID, ok := id.(string); ok {
				strSourceIDs[i] = strID
			}
		}
		edge.SourceIDs = strSourceIDs
	}

	return edge, nil
}
