package grpcsvc

import (
	"context"
	"net"
	"testing"

	"github.com/soundprediction/predicato"
	"github.com/soundprediction/predicato/pkg/types"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// fakePredicato implements just enough of predicato.Predicato for the roundtrip.
type fakePredicato struct {
	predicato.Predicato // embed to satisfy the interface; unused methods panic if called
	gotQuery            string
}

func (f *fakePredicato) Search(_ context.Context, query string, _ *types.SearchConfig) (*types.SearchResults, error) {
	f.gotQuery = query
	return &types.SearchResults{Query: query, Total: 2, Nodes: []*types.Node{{Name: "Hypothyroidism"}}}, nil
}

// TestSearchRoundtrip starts the gRPC server with the JSON codec + hand-written
// ServiceDesc, dials a client, and verifies a Search call serializes the request,
// reaches the server, and returns the typed result — exercising the whole
// transport without protoc.
func TestSearchRoundtrip(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	fake := &fakePredicato{}
	gs := grpc.NewServer()
	NewServer(fake).Register(gs)
	go func() { _ = gs.Serve(lis) }()
	defer gs.Stop()

	cc, err := grpc.NewClient(lis.Addr().String(),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = cc.Close() }()
	client := NewClientConn(cc)

	res, err := client.Search(context.Background(), "fatigue, cold intolerance", &types.SearchConfig{Limit: 5})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if res.Query != "fatigue, cold intolerance" || res.Total != 2 || len(res.Nodes) != 1 {
		t.Fatalf("unexpected result: %+v", res)
	}
	if res.Nodes[0].Name != "Hypothyroidism" {
		t.Fatalf("node not round-tripped: %+v", res.Nodes[0])
	}
	if fake.gotQuery != "fatigue, cold intolerance" {
		t.Fatalf("server did not receive query: %q", fake.gotQuery)
	}

	// Health works without a body.
	if h, err := client.Health(context.Background()); err != nil || h != "ok" {
		t.Fatalf("health: %q err=%v", h, err)
	}
}
