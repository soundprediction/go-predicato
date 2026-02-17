//go:build !system_duckpgq

package driver

import (
	"context"
	"errors"
	"time"

	"github.com/soundprediction/predicato/pkg/types"
)

// ErrDuckPGQCGORequired is returned when DuckPGQ operations are called without CGO support
var ErrDuckPGQCGORequired = errors.New("DuckPGQ driver requires CGO; build with CGO_ENABLED=1")

// DuckPGQDriver is a stub implementation when CGO is disabled.
type DuckPGQDriver struct{}

// NewDuckPGQDriver returns an error when CGO is disabled
func NewDuckPGQDriver(uri string, embeddingDim int) (*DuckPGQDriver, error) {
	return nil, ErrDuckPGQCGORequired
}

func (d *DuckPGQDriver) ExecuteQuery(ctx context.Context, query string, kwargs map[string]interface{}) (interface{}, interface{}, interface{}, error) {
	return nil, nil, nil, ErrDuckPGQCGORequired
}

func (d *DuckPGQDriver) Session(database *string) GraphDriverSession { return nil }
func (d *DuckPGQDriver) Close() error                                { return nil }
func (d *DuckPGQDriver) DeleteAllIndexes(database string)            {}
func (d *DuckPGQDriver) Provider() GraphProvider                     { return GraphProviderDuckPGQ }
func (d *DuckPGQDriver) GetAossClient() interface{}                  { return nil }

func (d *DuckPGQDriver) GetNode(ctx context.Context, nodeID, groupID string) (*types.Node, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) UpsertNode(ctx context.Context, node *types.Node) error {
	return ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) DeleteNode(ctx context.Context, nodeID, groupID string) error {
	return ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) GetNodes(ctx context.Context, nodeIDs []string, groupID string) ([]*types.Node, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) UpsertNodes(ctx context.Context, nodes []*types.Node) error {
	return ErrDuckPGQCGORequired
}

func (d *DuckPGQDriver) GetEdge(ctx context.Context, edgeID, groupID string) (*types.Edge, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) UpsertEdge(ctx context.Context, edge *types.Edge) error {
	return ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) UpsertEpisodicEdge(ctx context.Context, episodeUUID, entityUUID, groupID string) error {
	return ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) UpsertCommunityEdge(ctx context.Context, communityUUID, nodeUUID, uuid, groupID string) error {
	return ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) DeleteEdge(ctx context.Context, edgeID, groupID string) error {
	return ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) GetEdges(ctx context.Context, edgeIDs []string, groupID string) ([]*types.Edge, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) UpsertEdges(ctx context.Context, edges []*types.Edge) error {
	return ErrDuckPGQCGORequired
}

func (d *DuckPGQDriver) GetNeighbors(ctx context.Context, nodeID, groupID string, maxDistance int) ([]*types.Node, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) GetRelatedNodes(ctx context.Context, nodeID, groupID string, edgeTypes []types.EdgeType) ([]*types.Node, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) GetNodeNeighbors(ctx context.Context, nodeUUID, groupID string) ([]types.Neighbor, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) GetBetweenNodes(ctx context.Context, sourceNodeID, targetNodeID string) ([]*types.Edge, error) {
	return nil, ErrDuckPGQCGORequired
}

func (d *DuckPGQDriver) SearchNodes(ctx context.Context, query, groupID string, options *SearchOptions) ([]*types.Node, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) SearchEdges(ctx context.Context, query, groupID string, options *SearchOptions) ([]*types.Edge, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) SearchNodesByVector(ctx context.Context, vector []float32, groupID string, options *VectorSearchOptions) ([]*types.Node, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) SearchEdgesByVector(ctx context.Context, vector []float32, groupID string, options *VectorSearchOptions) ([]*types.Edge, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) SearchNodesByEmbedding(ctx context.Context, embedding []float32, groupID string, limit int) ([]*types.Node, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) SearchEdgesByEmbedding(ctx context.Context, embedding []float32, groupID string, limit int) ([]*types.Edge, error) {
	return nil, ErrDuckPGQCGORequired
}

func (d *DuckPGQDriver) GetNodesInTimeRange(ctx context.Context, start, end time.Time, groupID string) ([]*types.Node, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) GetEdgesInTimeRange(ctx context.Context, start, end time.Time, groupID string) ([]*types.Edge, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) RetrieveEpisodes(ctx context.Context, referenceTime time.Time, groupIDs []string, limit int, episodeType *types.EpisodeType) ([]*types.Node, error) {
	return nil, ErrDuckPGQCGORequired
}

func (d *DuckPGQDriver) GetCommunities(ctx context.Context, groupID string, level int) ([]*types.Node, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) BuildCommunities(ctx context.Context, groupID string) error {
	return ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) GetExistingCommunity(ctx context.Context, entityUUID string) (*types.Node, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) FindModalCommunity(ctx context.Context, entityUUID string) (*types.Node, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) RemoveCommunities(ctx context.Context) error {
	return ErrDuckPGQCGORequired
}

func (d *DuckPGQDriver) CreateIndices(ctx context.Context) error {
	return ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) GetStats(ctx context.Context, groupID string) (*GraphStats, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) ParseNodesFromRecords(records any) ([]*types.Node, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) GetEntityNodesByGroup(ctx context.Context, groupID string) ([]*types.Node, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) GetAllGroupIDs(ctx context.Context) ([]string, error) {
	return nil, ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) IterateNodes(ctx context.Context, groupID string, fn func(*types.Node) error) error {
	return ErrDuckPGQCGORequired
}
func (d *DuckPGQDriver) IterateEdges(ctx context.Context, groupID string, fn func(*types.Edge) error) error {
	return ErrDuckPGQCGORequired
}
