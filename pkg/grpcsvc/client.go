package grpcsvc

import (
	"context"

	"github.com/soundprediction/predicato"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/grpcsvc/pb"
	"github.com/soundprediction/predicato/pkg/types"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Client calls a remote predicato gRPC graph server, presenting predicato's Go
// types (not the generated protos). It covers the core query + ingest surface
// callers use (e.g. humn's RAG layer) — not the full predicato.Predicato
// interface (methods returning non-serializable handles cannot be remoted).
type Client struct {
	cc  *grpc.ClientConn
	gc  pb.GraphServiceClient
	own bool // true when Dial created cc (so Close shuts it down)
}

// Dial connects to a predicato gRPC server at target (host:port). Defaults to an
// insecure transport; pass opts to add TLS, keepalive, retries, load-balancing.
// The connection is lazy (grpc.NewClient).
func Dial(target string, opts ...grpc.DialOption) (*Client, error) {
	base := []grpc.DialOption{grpc.WithTransportCredentials(insecure.NewCredentials())}
	cc, err := grpc.NewClient(target, append(base, opts...)...)
	if err != nil {
		return nil, err
	}
	return &Client{cc: cc, gc: pb.NewGraphServiceClient(cc), own: true}, nil
}

// NewClientConn wraps an already-built ClientConn (custom creds/interceptors);
// the caller owns the conn's lifecycle.
func NewClientConn(cc *grpc.ClientConn) *Client {
	return &Client{cc: cc, gc: pb.NewGraphServiceClient(cc)}
}

// Search runs hybrid search on the remote graph.
func (c *Client) Search(ctx context.Context, query string, config *types.SearchConfig) (*types.SearchResults, error) {
	res, err := c.gc.Search(ctx, &pb.SearchRequest{Query: query, Config: searchConfigToProto(config)})
	if err != nil {
		return nil, err
	}
	return searchResultsFromProto(res), nil
}

// AddEpisode ingests one episode into the remote graph (serializable options only).
func (c *Client) AddEpisode(ctx context.Context, episode types.Episode, options *predicato.AddEpisodeOptions) (*types.AddEpisodeResults, error) {
	res, err := c.gc.AddEpisode(ctx, &pb.AddEpisodeRequest{Episode: episodeToProto(episode), Options: addOptionsToProto(options)})
	if err != nil {
		return nil, err
	}
	return addResultsFromProto(res), nil
}

// GetNode / GetEdge fetch a single graph element.
func (c *Client) GetNode(ctx context.Context, uuid string) (*types.Node, error) {
	n, err := c.gc.GetNode(ctx, &pb.GetNodeRequest{Uuid: uuid})
	if err != nil {
		return nil, err
	}
	return nodeFromProto(n), nil
}

func (c *Client) GetEdge(ctx context.Context, uuid string) (*types.Edge, error) {
	e, err := c.gc.GetEdge(ctx, &pb.GetEdgeRequest{Uuid: uuid})
	if err != nil {
		return nil, err
	}
	return edgeFromProto(e), nil
}

// GetStats returns graph statistics from the remote engine.
func (c *Client) GetStats(ctx context.Context) (*driver.GraphStats, error) {
	st, err := c.gc.GetStats(ctx, &pb.StatsRequest{})
	if err != nil {
		return nil, err
	}
	return &driver.GraphStats{NodeCount: st.GetNodeCount(), EdgeCount: st.GetEdgeCount(), CommunityCount: st.GetCommunityCount()}, nil
}

// Health pings the server.
func (c *Client) Health(ctx context.Context) (string, error) {
	h, err := c.gc.Health(ctx, &pb.HealthRequest{})
	if err != nil {
		return "", err
	}
	return h.GetStatus(), nil
}

// Close closes the underlying connection when this Client owns it.
func (c *Client) Close() error {
	if c.own {
		return c.cc.Close()
	}
	return nil
}

// addOptionsToProto maps the serializable subset of add options for the client.
func addOptionsToProto(o *predicato.AddEpisodeOptions) *pb.AddEpisodeOptions {
	if o == nil {
		return nil
	}
	return &pb.AddEpisodeOptions{
		ExcludedEntityTypes:  o.ExcludedEntityTypes,
		PreviousEpisodeUuids: o.PreviousEpisodeUUIDs,
		MaxCharacters:        int32(o.MaxCharacters),
		EntityTypes:          structToProto(o.EntityTypes),
		EdgeTypes:            structToProto(o.EdgeTypes),
	}
}
