//go:build !system_cozo

package driver

import (
	"context"
	"errors"
	"time"

	"github.com/soundprediction/predicato/pkg/types"
)

// ErrCozoCGORequired is returned when CozoDB operations are called without CGO support
var ErrCozoCGORequired = errors.New("CozoDB driver requires CGO; build with CGO_ENABLED=1")

// CozoDriver is a stub implementation when CGO is disabled.
type CozoDriver struct{}

// NewCozoDriver returns an error when CGO is disabled
func NewCozoDriver(uri string, embeddingDim int) (*CozoDriver, error) {
	return nil, ErrCozoCGORequired
}

func (d *CozoDriver) ExecuteQuery(ctx context.Context, query string, kwargs map[string]interface{}) (interface{}, interface{}, interface{}, error) {
	return nil, nil, nil, ErrCozoCGORequired
}

func (d *CozoDriver) Session(database *string) GraphDriverSession { return nil }
func (d *CozoDriver) Close() error                                { return nil }
func (d *CozoDriver) DeleteAllIndexes(database string)            {}
func (d *CozoDriver) Provider() GraphProvider                     { return GraphProviderCozo }
func (d *CozoDriver) GetAossClient() interface{}                  { return nil }

func (d *CozoDriver) GetNode(ctx context.Context, nodeID, groupID string) (*types.Node, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) UpsertNode(ctx context.Context, node *types.Node) error {
	return ErrCozoCGORequired
}
func (d *CozoDriver) DeleteNode(ctx context.Context, nodeID, groupID string) error {
	return ErrCozoCGORequired
}
func (d *CozoDriver) GetNodes(ctx context.Context, nodeIDs []string, groupID string) ([]*types.Node, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) UpsertNodes(ctx context.Context, nodes []*types.Node) error {
	return ErrCozoCGORequired
}

func (d *CozoDriver) GetEdge(ctx context.Context, edgeID, groupID string) (*types.Edge, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) UpsertEdge(ctx context.Context, edge *types.Edge) error {
	return ErrCozoCGORequired
}
func (d *CozoDriver) UpsertEpisodicEdge(ctx context.Context, episodeUUID, entityUUID, groupID string) error {
	return ErrCozoCGORequired
}
func (d *CozoDriver) UpsertCommunityEdge(ctx context.Context, communityUUID, nodeUUID, uuid, groupID string) error {
	return ErrCozoCGORequired
}
func (d *CozoDriver) DeleteEdge(ctx context.Context, edgeID, groupID string) error {
	return ErrCozoCGORequired
}
func (d *CozoDriver) GetEdges(ctx context.Context, edgeIDs []string, groupID string) ([]*types.Edge, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) UpsertEdges(ctx context.Context, edges []*types.Edge) error {
	return ErrCozoCGORequired
}

func (d *CozoDriver) GetNeighbors(ctx context.Context, nodeID, groupID string, maxDistance int) ([]*types.Node, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) GetRelatedNodes(ctx context.Context, nodeID, groupID string, edgeTypes []types.EdgeType) ([]*types.Node, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) GetNodeNeighbors(ctx context.Context, nodeUUID, groupID string) ([]types.Neighbor, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) GetBetweenNodes(ctx context.Context, sourceNodeID, targetNodeID string) ([]*types.Edge, error) {
	return nil, ErrCozoCGORequired
}

func (d *CozoDriver) SearchNodes(ctx context.Context, query, groupID string, options *SearchOptions) ([]*types.Node, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) SearchEdges(ctx context.Context, query, groupID string, options *SearchOptions) ([]*types.Edge, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) SearchNodesByVector(ctx context.Context, vector []float32, groupID string, options *VectorSearchOptions) ([]*types.Node, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) SearchEdgesByVector(ctx context.Context, vector []float32, groupID string, options *VectorSearchOptions) ([]*types.Edge, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) SearchNodesByEmbedding(ctx context.Context, embedding []float32, groupID string, limit int) ([]*types.Node, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) SearchEdgesByEmbedding(ctx context.Context, embedding []float32, groupID string, limit int) ([]*types.Edge, error) {
	return nil, ErrCozoCGORequired
}

func (d *CozoDriver) GetNodesInTimeRange(ctx context.Context, start, end time.Time, groupID string) ([]*types.Node, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) GetEdgesInTimeRange(ctx context.Context, start, end time.Time, groupID string) ([]*types.Edge, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) RetrieveEpisodes(ctx context.Context, referenceTime time.Time, groupIDs []string, limit int, episodeType *types.EpisodeType) ([]*types.Node, error) {
	return nil, ErrCozoCGORequired
}

func (d *CozoDriver) GetCommunities(ctx context.Context, groupID string, level int) ([]*types.Node, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) BuildCommunities(ctx context.Context, groupID string) error {
	return ErrCozoCGORequired
}
func (d *CozoDriver) GetExistingCommunity(ctx context.Context, entityUUID string) (*types.Node, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) FindModalCommunity(ctx context.Context, entityUUID string) (*types.Node, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) RemoveCommunities(ctx context.Context) error {
	return ErrCozoCGORequired
}

func (d *CozoDriver) CreateIndices(ctx context.Context) error {
	return ErrCozoCGORequired
}
func (d *CozoDriver) GetStats(ctx context.Context, groupID string) (*GraphStats, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) ParseNodesFromRecords(records any) ([]*types.Node, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) GetEntityNodesByGroup(ctx context.Context, groupID string) ([]*types.Node, error) {
	return nil, ErrCozoCGORequired
}
func (d *CozoDriver) GetAllGroupIDs(ctx context.Context) ([]string, error) {
	return nil, ErrCozoCGORequired
}
