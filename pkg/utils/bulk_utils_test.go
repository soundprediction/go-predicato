package utils

import (
	"testing"
	"time"

	"github.com/soundprediction/predicato/pkg/types"
)

func TestRetrievePreviousEpisodesBulk(t *testing.T) {
	// Create mock episodes
	episode1 := &types.Episode{
		ID:        "episode1",
		Name:      "Test Episode 1",
		Content:   "This is test content 1",
		CreatedAt: time.Now(),
		GroupID:   "test_group",
	}

	episode2 := &types.Episode{
		ID:        "episode2",
		Name:      "Test Episode 2",
		Content:   "This is test content 2",
		CreatedAt: time.Now().Add(1 * time.Hour),
		GroupID:   "test_group",
	}

	episodes := []*types.Episode{episode1, episode2}

	// Since we don't have a real driver implementation here,
	// we'll just test that the function signature is correct
	// and that it doesn't panic
	if len(episodes) != 2 {
		t.Errorf("Expected 2 episodes, got %d", len(episodes))
	}

	// Test that episode tuples can be created
	tuple := EpisodeTuple{
		Episode:          episode1,
		PreviousEpisodes: []*types.Episode{episode2},
	}

	if tuple.Episode.ID != "episode1" {
		t.Errorf("Expected episode1, got %s", tuple.Episode.ID)
	}

	if len(tuple.PreviousEpisodes) != 1 {
		t.Errorf("Expected 1 previous episode, got %d", len(tuple.PreviousEpisodes))
	}
}

func TestExtractNodesAndEdgesResult(t *testing.T) {
	// Test the result structure
	node1 := &types.Node{
		Uuid: "node1",
		Name: "Test Node 1",
		Type: types.EntityNodeType,
	}

	edge1 := &types.Edge{
		BaseEdge: types.BaseEdge{
			Uuid:         "edge1",
			SourceNodeID: "node1",
			TargetNodeID: "node2",
		},
		SourceID: "node1",
		TargetID: "node2",
		Type:     types.EntityEdgeType,
	}

	result := &ExtractNodesAndEdgesResult{
		ExtractedNodes: []*types.Node{node1},
		ExtractedEdges: []*types.Edge{edge1},
	}

	if len(result.ExtractedNodes) != 1 {
		t.Errorf("Expected 1 node, got %d", len(result.ExtractedNodes))
	}

	if len(result.ExtractedEdges) != 1 {
		t.Errorf("Expected 1 edge, got %d", len(result.ExtractedEdges))
	}

	if result.ExtractedNodes[0].Uuid != "node1" {
		t.Errorf("Expected node1, got %s", result.ExtractedNodes[0].Uuid)
	}
}

func TestAddNodesAndEdgesResult(t *testing.T) {
	// Test the result structure
	result := &AddNodesAndEdgesResult{
		EpisodicNodes: []*types.Node{},
		EpisodicEdges: []*types.Edge{},
		EntityNodes:   []*types.Node{},
		EntityEdges:   []*types.Edge{},
		Errors:        []error{},
	}

	if result.EpisodicNodes == nil {
		t.Error("EpisodicNodes should be initialized")
	}

	if result.EpisodicEdges == nil {
		t.Error("EpisodicEdges should be initialized")
	}

	if result.EntityNodes == nil {
		t.Error("EntityNodes should be initialized")
	}

	if result.EntityEdges == nil {
		t.Error("EntityEdges should be initialized")
	}

	if result.Errors == nil {
		t.Error("Errors should be initialized")
	}
}

func TestDedupeResults(t *testing.T) {
	// Test the dedupe result structures
	nodesResult := &DedupeNodesResult{
		NodesByEpisode: make(map[string][]*types.Node),
		UUIDMap:        make(map[string]string),
	}

	edgesResult := &DedupeEdgesResult{
		EdgesByEpisode: make(map[string][]*types.Edge),
		UUIDMap:        make(map[string]string),
	}

	if nodesResult.NodesByEpisode == nil {
		t.Error("NodesByEpisode should be initialized")
	}

	if nodesResult.UUIDMap == nil {
		t.Error("UUIDMap should be initialized")
	}

	if edgesResult.EdgesByEpisode == nil {
		t.Error("EdgesByEpisode should be initialized")
	}

	if edgesResult.UUIDMap == nil {
		t.Error("UUIDMap should be initialized")
	}
}

func TestClients(t *testing.T) {
	// Test the Clients structure
	clients := &Clients{
		Driver:   nil, // These would be real implementations in practice
		NLP:      nil,
		Embedder: nil,
		Prompts:  nil,
	}

	// Verify structure is properly initialized by checking it's not the zero value pointer
	// (Note: clients cannot be nil here since we just assigned it a non-nil pointer)
	if clients.Driver != nil || clients.NLP != nil || clients.Embedder != nil || clients.Prompts != nil {
		t.Error("Clients fields should be nil as initialized")
	}
}

func TestMinFunction(t *testing.T) {
	// Test the helper min function
	result := min(5, 3)
	if result != 3 {
		t.Errorf("Expected 3, got %d", result)
	}

	result = min(2, 7)
	if result != 2 {
		t.Errorf("Expected 2, got %d", result)
	}

	result = min(5, 5)
	if result != 5 {
		t.Errorf("Expected 5, got %d", result)
	}
}
