package grpcsvc

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"

	"github.com/soundprediction/predicato"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/grpcsvc/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// statsCapable is the optional GetStats method; *predicato.Client provides it but
// it is not on the Predicato interface, so the server detects it at runtime.
type statsCapable interface {
	GetStats(ctx context.Context) (*driver.GraphStats, error)
}

// Server adapts a predicato.Predicato to the generated GraphServiceServer.
type Server struct {
	pb.UnimplementedGraphServiceServer
	p predicato.Predicato
}

// NewServer wraps a predicato engine for gRPC serving.
func NewServer(p predicato.Predicato) *Server { return &Server{p: p} }

func (s *Server) Search(ctx context.Context, req *pb.SearchRequest) (*pb.SearchResults, error) {
	res, err := s.p.Search(ctx, req.GetQuery(), searchConfigFromProto(req.GetConfig()))
	if err != nil {
		return nil, err
	}
	return searchResultsToProto(res), nil
}

func (s *Server) AddEpisode(ctx context.Context, req *pb.AddEpisodeRequest) (*pb.AddEpisodeResults, error) {
	res, err := s.p.AddEpisode(ctx, episodeFromProto(req.GetEpisode()), addOptionsFromProto(req.GetOptions()))
	if err != nil {
		return nil, err
	}
	return addResultsToProto(res), nil
}

func (s *Server) GetNode(ctx context.Context, req *pb.GetNodeRequest) (*pb.Node, error) {
	n, err := s.p.GetNode(ctx, req.GetUuid())
	if err != nil {
		return nil, err
	}
	return nodeToProto(n), nil
}

func (s *Server) GetEdge(ctx context.Context, req *pb.GetEdgeRequest) (*pb.Edge, error) {
	e, err := s.p.GetEdge(ctx, req.GetUuid())
	if err != nil {
		return nil, err
	}
	return edgeToProto(e), nil
}

func (s *Server) GetStats(ctx context.Context, _ *pb.StatsRequest) (*pb.GraphStats, error) {
	sc, ok := s.p.(statsCapable)
	if !ok {
		return nil, status.Error(codes.Unimplemented, "graph engine does not support GetStats")
	}
	st, err := sc.GetStats(ctx)
	if err != nil {
		return nil, err
	}
	return &pb.GraphStats{NodeCount: st.NodeCount, EdgeCount: st.EdgeCount, CommunityCount: st.CommunityCount}, nil
}

func (s *Server) Health(_ context.Context, _ *pb.HealthRequest) (*pb.HealthResponse, error) {
	return &pb.HealthResponse{Status: "ok"}, nil
}

// addOptionsFromProto maps the serializable subset of add options (the
// GraphModeler / error-handling policy cannot be remoted).
func addOptionsFromProto(p *pb.AddEpisodeOptions) *predicato.AddEpisodeOptions {
	if p == nil {
		return nil
	}
	return &predicato.AddEpisodeOptions{
		ExcludedEntityTypes:  p.GetExcludedEntityTypes(),
		PreviousEpisodeUUIDs: p.GetPreviousEpisodeUuids(),
		MaxCharacters:        int(p.GetMaxCharacters()),
		EntityTypes:          structFromProto(p.GetEntityTypes()),
		EdgeTypes:            structFromProto(p.GetEdgeTypes()),
	}
}

// Register installs the GraphService on a grpc.Server.
func (s *Server) Register(gs *grpc.Server) { pb.RegisterGraphServiceServer(gs, s) }

// Serve builds a grpc.Server, registers the service, and blocks serving on addr
// (e.g. ":50071") until the listener closes. opts allows TLS, interceptors, etc.
func Serve(p predicato.Predicato, addr string, opts ...grpc.ServerOption) error {
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		return fmt.Errorf("grpcsvc listen %s: %w", addr, err)
	}
	if key := os.Getenv("PREDICATO_API_KEY"); key != "" {
		log.Printf("grpcsvc: api-key auth ON (x-api-key required; Health exempt)")
		opts = append([]grpc.ServerOption{
			grpc.ChainUnaryInterceptor(APIKeyUnaryInterceptor(key)),
			grpc.ChainStreamInterceptor(APIKeyStreamInterceptor(key)),
		}, opts...)
	} else {
		log.Printf("grpcsvc: api-key auth OFF (PREDICATO_API_KEY unset; fail-open)")
	}
	gs := grpc.NewServer(opts...)
	NewServer(p).Register(gs)
	return gs.Serve(lis)
}
