//go:build !cgo || !system_ladybug

package driver

import (
	"context"
	"errors"
	"time"

	"github.com/soundprediction/predicato/pkg/types"
)

// ErrCGORequired is returned when Ladybug operations are called without CGO support
var ErrCGORequired = errors.New("ladybug driver requires CGO; build with CGO_ENABLED=1")

// LadybugDriver is a stub implementation when CGO is disabled.
// All methods return ErrCGORequired.
type LadybugDriver struct{}

// NewLadybugDriver returns an error when CGO is disabled
func NewLadybugDriver(dbPath string, maxConcurrentQueries int) (*LadybugDriver, error) {
	return nil, ErrCGORequired
}

// ExecuteQuery returns ErrCGORequired
func (k *LadybugDriver) ExecuteQuery(ctx context.Context, cypherQuery string, kwargs map[string]interface{}) (interface{}, interface{}, interface{}, error) {
	return nil, nil, nil, ErrCGORequired
}

// Session returns nil
func (k *LadybugDriver) Session(database *string) GraphDriverSession {
	return nil
}

// Close returns nil
func (k *LadybugDriver) Close() error {
	return nil
}

// DeleteAllIndexes is a no-op
func (k *LadybugDriver) DeleteAllIndexes(database string) {}

// Provider returns GraphProviderLadybug
func (k *LadybugDriver) Provider() GraphProvider {
	return GraphProviderLadybug
}

// GetAossClient returns nil
func (k *LadybugDriver) GetAossClient() interface{} {
	return nil
}

// GetNode returns ErrCGORequired
func (k *LadybugDriver) GetNode(ctx context.Context, nodeID, groupID string) (*types.Node, error) {
	return nil, ErrCGORequired
}

// UpsertNode returns ErrCGORequired
func (k *LadybugDriver) UpsertNode(ctx context.Context, node *types.Node) error {
	return ErrCGORequired
}

// DeleteNode returns ErrCGORequired
func (k *LadybugDriver) DeleteNode(ctx context.Context, nodeID, groupID string) error {
	return ErrCGORequired
}

// GetNodes returns ErrCGORequired
func (k *LadybugDriver) GetNodes(ctx context.Context, nodeIDs []string, groupID string) ([]*types.Node, error) {
	return nil, ErrCGORequired
}

// GetEdge returns ErrCGORequired
func (k *LadybugDriver) GetEdge(ctx context.Context, edgeID, groupID string) (*types.Edge, error) {
	return nil, ErrCGORequired
}

// UpsertEdge returns ErrCGORequired
func (k *LadybugDriver) UpsertEdge(ctx context.Context, edge *types.Edge) error {
	return ErrCGORequired
}

// UpsertEpisodicEdge returns ErrCGORequired
func (k *LadybugDriver) UpsertEpisodicEdge(ctx context.Context, episodeUUID, entityUUID, groupID string) error {
	return ErrCGORequired
}

// UpsertCommunityEdge returns ErrCGORequired
func (k *LadybugDriver) UpsertCommunityEdge(ctx context.Context, communityUUID, nodeUUID, uuid, groupID string) error {
	return ErrCGORequired
}

// DeleteEdge returns ErrCGORequired
func (k *LadybugDriver) DeleteEdge(ctx context.Context, edgeID, groupID string) error {
	return ErrCGORequired
}

// GetEdges returns ErrCGORequired
func (k *LadybugDriver) GetEdges(ctx context.Context, edgeIDs []string, groupID string) ([]*types.Edge, error) {
	return nil, ErrCGORequired
}

// GetNeighbors returns ErrCGORequired
func (k *LadybugDriver) GetNeighbors(ctx context.Context, nodeID, groupID string, maxDistance int) ([]*types.Node, error) {
	return nil, ErrCGORequired
}

// GetRelatedNodes returns ErrCGORequired
func (k *LadybugDriver) GetRelatedNodes(ctx context.Context, nodeID, groupID string, edgeTypes []types.EdgeType) ([]*types.Node, error) {
	return nil, ErrCGORequired
}

// GetNodeNeighbors returns ErrCGORequired
func (k *LadybugDriver) GetNodeNeighbors(ctx context.Context, nodeUUID, groupID string) ([]types.Neighbor, error) {
	return nil, ErrCGORequired
}

// GetBetweenNodes returns ErrCGORequired
func (k *LadybugDriver) GetBetweenNodes(ctx context.Context, sourceNodeID, targetNodeID string) ([]*types.Edge, error) {
	return nil, ErrCGORequired
}

// SearchNodesByEmbedding returns ErrCGORequired
func (k *LadybugDriver) SearchNodesByEmbedding(ctx context.Context, embedding []float32, groupID string, limit int) ([]*types.Node, error) {
	return nil, ErrCGORequired
}

// SearchEdgesByEmbedding returns ErrCGORequired
func (k *LadybugDriver) SearchEdgesByEmbedding(ctx context.Context, embedding []float32, groupID string, limit int) ([]*types.Edge, error) {
	return nil, ErrCGORequired
}

// SearchNodes returns ErrCGORequired
func (k *LadybugDriver) SearchNodes(ctx context.Context, query, groupID string, options *SearchOptions) ([]*types.Node, error) {
	return nil, ErrCGORequired
}

// SearchEdges returns ErrCGORequired
func (k *LadybugDriver) SearchEdges(ctx context.Context, query, groupID string, options *SearchOptions) ([]*types.Edge, error) {
	return nil, ErrCGORequired
}

// SearchNodesByVector returns ErrCGORequired
func (k *LadybugDriver) SearchNodesByVector(ctx context.Context, vector []float32, groupID string, options *VectorSearchOptions) ([]*types.Node, error) {
	return nil, ErrCGORequired
}

// SearchEdgesByVector returns ErrCGORequired
func (k *LadybugDriver) SearchEdgesByVector(ctx context.Context, vector []float32, groupID string, options *VectorSearchOptions) ([]*types.Edge, error) {
	return nil, ErrCGORequired
}

// UpsertNodes returns ErrCGORequired
func (k *LadybugDriver) UpsertNodes(ctx context.Context, nodes []*types.Node) error {
	return ErrCGORequired
}

// UpsertEdges returns ErrCGORequired
func (k *LadybugDriver) UpsertEdges(ctx context.Context, edges []*types.Edge) error {
	return ErrCGORequired
}

// GetNodesInTimeRange returns ErrCGORequired
func (k *LadybugDriver) GetNodesInTimeRange(ctx context.Context, start, end time.Time, groupID string) ([]*types.Node, error) {
	return nil, ErrCGORequired
}

// GetEdgesInTimeRange returns ErrCGORequired
func (k *LadybugDriver) GetEdgesInTimeRange(ctx context.Context, start, end time.Time, groupID string) ([]*types.Edge, error) {
	return nil, ErrCGORequired
}

// RetrieveEpisodes returns ErrCGORequired
func (k *LadybugDriver) RetrieveEpisodes(ctx context.Context, referenceTime time.Time, groupIDs []string, limit int, episodeType *types.EpisodeType) ([]*types.Node, error) {
	return nil, ErrCGORequired
}

// GetCommunities returns ErrCGORequired
func (k *LadybugDriver) GetCommunities(ctx context.Context, groupID string, level int) ([]*types.Node, error) {
	return nil, ErrCGORequired
}

// BuildCommunities returns ErrCGORequired
func (k *LadybugDriver) BuildCommunities(ctx context.Context, groupID string) error {
	return ErrCGORequired
}

// GetExistingCommunity returns ErrCGORequired
func (k *LadybugDriver) GetExistingCommunity(ctx context.Context, entityUUID string) (*types.Node, error) {
	return nil, ErrCGORequired
}

// FindModalCommunity returns ErrCGORequired
func (k *LadybugDriver) FindModalCommunity(ctx context.Context, entityUUID string) (*types.Node, error) {
	return nil, ErrCGORequired
}

// RemoveCommunities returns ErrCGORequired
func (k *LadybugDriver) RemoveCommunities(ctx context.Context) error {
	return ErrCGORequired
}

// CreateIndices returns ErrCGORequired
func (k *LadybugDriver) CreateIndices(ctx context.Context) error {
	return ErrCGORequired
}

// GetStats returns ErrCGORequired
func (k *LadybugDriver) GetStats(ctx context.Context, groupID string) (*GraphStats, error) {
	return nil, ErrCGORequired
}

// ParseNodesFromRecords returns ErrCGORequired
func (k *LadybugDriver) ParseNodesFromRecords(records any) ([]*types.Node, error) {
	return nil, ErrCGORequired
}

// GetEntityNodesByGroup returns ErrCGORequired
func (k *LadybugDriver) GetEntityNodesByGroup(ctx context.Context, groupID string) ([]*types.Node, error) {
	return nil, ErrCGORequired
}

// GetAllGroupIDs returns ErrCGORequired
func (k *LadybugDriver) GetAllGroupIDs(ctx context.Context) ([]string, error) {
	return nil, ErrCGORequired
}
