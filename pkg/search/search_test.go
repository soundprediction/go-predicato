package search

import (
	"context"
	"testing"
	"time"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/types"
)

// MockGraphDriver implements driver.GraphDriver for testing
type MockGraphDriver struct {
	nodes         map[string]*types.Node
	edges         map[string]*types.Edge
	searchResults struct {
		nodes []*types.Node
		edges []*types.Edge
	}
	vectorSearchResults struct {
		nodes []*types.Node
		edges []*types.Edge
	}
	neighborResults []*types.Node
	err             error
}

func NewMockGraphDriver() *MockGraphDriver {
	return &MockGraphDriver{
		nodes: make(map[string]*types.Node),
		edges: make(map[string]*types.Edge),
	}
}

func (m *MockGraphDriver) SetSearchResults(nodes []*types.Node, edges []*types.Edge) {
	m.searchResults.nodes = nodes
	m.searchResults.edges = edges
}

func (m *MockGraphDriver) SetVectorSearchResults(nodes []*types.Node, edges []*types.Edge) {
	m.vectorSearchResults.nodes = nodes
	m.vectorSearchResults.edges = edges
}

func (m *MockGraphDriver) SetNeighborResults(nodes []*types.Node) {
	m.neighborResults = nodes
}

func (m *MockGraphDriver) SetError(err error) {
	m.err = err
}

func (m *MockGraphDriver) AddNode(node *types.Node) {
	m.nodes[node.Uuid] = node
}

func (m *MockGraphDriver) AddEdge(edge *types.Edge) {
	m.edges[edge.Uuid] = edge
}

// GraphDriver interface implementations

func (m *MockGraphDriver) ExecuteQuery(ctx context.Context, cypherQuery string, kwargs map[string]interface{}) (interface{}, interface{}, interface{}, error) {
	return nil, nil, nil, m.err
}

func (m *MockGraphDriver) Session(database *string) driver.GraphDriverSession {
	return nil
}

func (m *MockGraphDriver) Close() error {
	return nil
}

func (m *MockGraphDriver) DeleteAllIndexes(database string) {}

func (m *MockGraphDriver) Provider() driver.GraphProvider {
	return driver.GraphProviderLadybug
}

func (m *MockGraphDriver) GetAossClient() interface{} {
	return nil
}

func (m *MockGraphDriver) GetNode(ctx context.Context, nodeID, groupID string) (*types.Node, error) {
	if m.err != nil {
		return nil, m.err
	}
	node, ok := m.nodes[nodeID]
	if !ok {
		return nil, nil
	}
	return node, nil
}

func (m *MockGraphDriver) UpsertNode(ctx context.Context, node *types.Node) error {
	if m.err != nil {
		return m.err
	}
	m.nodes[node.Uuid] = node
	return nil
}

func (m *MockGraphDriver) DeleteNode(ctx context.Context, nodeID, groupID string) error {
	if m.err != nil {
		return m.err
	}
	delete(m.nodes, nodeID)
	return nil
}

func (m *MockGraphDriver) GetNodes(ctx context.Context, nodeIDs []string, groupID string) ([]*types.Node, error) {
	if m.err != nil {
		return nil, m.err
	}
	var nodes []*types.Node
	for _, id := range nodeIDs {
		if node, ok := m.nodes[id]; ok {
			nodes = append(nodes, node)
		}
	}
	return nodes, nil
}

func (m *MockGraphDriver) GetEdge(ctx context.Context, edgeID, groupID string) (*types.Edge, error) {
	if m.err != nil {
		return nil, m.err
	}
	edge, ok := m.edges[edgeID]
	if !ok {
		return nil, nil
	}
	return edge, nil
}

func (m *MockGraphDriver) UpsertEdge(ctx context.Context, edge *types.Edge) error {
	if m.err != nil {
		return m.err
	}
	m.edges[edge.Uuid] = edge
	return nil
}

func (m *MockGraphDriver) UpsertEpisodicEdge(ctx context.Context, episodeUUID, entityUUID, groupID string) error {
	return m.err
}

func (m *MockGraphDriver) UpsertCommunityEdge(ctx context.Context, communityUUID, nodeUUID, uuid, groupID string) error {
	return m.err
}

func (m *MockGraphDriver) DeleteEdge(ctx context.Context, edgeID, groupID string) error {
	if m.err != nil {
		return m.err
	}
	delete(m.edges, edgeID)
	return nil
}

func (m *MockGraphDriver) GetEdges(ctx context.Context, edgeIDs []string, groupID string) ([]*types.Edge, error) {
	if m.err != nil {
		return nil, m.err
	}
	var edges []*types.Edge
	for _, id := range edgeIDs {
		if edge, ok := m.edges[id]; ok {
			edges = append(edges, edge)
		}
	}
	return edges, nil
}

func (m *MockGraphDriver) GetNeighbors(ctx context.Context, nodeID, groupID string, maxDistance int) ([]*types.Node, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.neighborResults, nil
}

func (m *MockGraphDriver) GetRelatedNodes(ctx context.Context, nodeID, groupID string, edgeTypes []types.EdgeType) ([]*types.Node, error) {
	return m.neighborResults, m.err
}

func (m *MockGraphDriver) GetNodeNeighbors(ctx context.Context, nodeUUID, groupID string) ([]types.Neighbor, error) {
	return nil, m.err
}

func (m *MockGraphDriver) GetBetweenNodes(ctx context.Context, sourceNodeID, targetNodeID string) ([]*types.Edge, error) {
	return nil, m.err
}

func (m *MockGraphDriver) SearchNodesByEmbedding(ctx context.Context, embedding []float32, groupID string, limit int) ([]*types.Node, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.vectorSearchResults.nodes, nil
}

func (m *MockGraphDriver) SearchEdgesByEmbedding(ctx context.Context, embedding []float32, groupID string, limit int) ([]*types.Edge, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.vectorSearchResults.edges, nil
}

func (m *MockGraphDriver) SearchNodes(ctx context.Context, query, groupID string, options *driver.SearchOptions) ([]*types.Node, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.searchResults.nodes, nil
}

func (m *MockGraphDriver) SearchEdges(ctx context.Context, query, groupID string, options *driver.SearchOptions) ([]*types.Edge, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.searchResults.edges, nil
}

func (m *MockGraphDriver) SearchNodesByVector(ctx context.Context, vector []float32, groupID string, options *driver.VectorSearchOptions) ([]*types.Node, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.vectorSearchResults.nodes, nil
}

func (m *MockGraphDriver) SearchEdgesByVector(ctx context.Context, vector []float32, groupID string, options *driver.VectorSearchOptions) ([]*types.Edge, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.vectorSearchResults.edges, nil
}

func (m *MockGraphDriver) UpsertNodes(ctx context.Context, nodes []*types.Node) error {
	if m.err != nil {
		return m.err
	}
	for _, node := range nodes {
		m.nodes[node.Uuid] = node
	}
	return nil
}

func (m *MockGraphDriver) UpsertEdges(ctx context.Context, edges []*types.Edge) error {
	if m.err != nil {
		return m.err
	}
	for _, edge := range edges {
		m.edges[edge.Uuid] = edge
	}
	return nil
}

func (m *MockGraphDriver) GetNodesInTimeRange(ctx context.Context, start, end time.Time, groupID string) ([]*types.Node, error) {
	return nil, m.err
}

func (m *MockGraphDriver) GetEdgesInTimeRange(ctx context.Context, start, end time.Time, groupID string) ([]*types.Edge, error) {
	return nil, m.err
}

func (m *MockGraphDriver) RetrieveEpisodes(ctx context.Context, referenceTime time.Time, groupIDs []string, limit int, episodeType *types.EpisodeType) ([]*types.Node, error) {
	return nil, m.err
}

func (m *MockGraphDriver) GetCommunities(ctx context.Context, groupID string, level int) ([]*types.Node, error) {
	return nil, m.err
}

func (m *MockGraphDriver) BuildCommunities(ctx context.Context, groupID string) error {
	return m.err
}

func (m *MockGraphDriver) GetExistingCommunity(ctx context.Context, entityUUID string) (*types.Node, error) {
	return nil, m.err
}

func (m *MockGraphDriver) FindModalCommunity(ctx context.Context, entityUUID string) (*types.Node, error) {
	return nil, m.err
}

func (m *MockGraphDriver) RemoveCommunities(ctx context.Context) error {
	return m.err
}

func (m *MockGraphDriver) CreateIndices(ctx context.Context) error {
	return m.err
}

func (m *MockGraphDriver) GetStats(ctx context.Context, groupID string) (*driver.GraphStats, error) {
	return nil, m.err
}

func (m *MockGraphDriver) ParseNodesFromRecords(records any) ([]*types.Node, error) {
	return nil, m.err
}

func (m *MockGraphDriver) GetEntityNodesByGroup(ctx context.Context, groupID string) ([]*types.Node, error) {
	return nil, m.err
}

func (m *MockGraphDriver) GetAllGroupIDs(ctx context.Context) ([]string, error) {
	return nil, m.err
}

// MockEmbedder implements embedder.Client for testing
type MockEmbedder struct {
	embeddings [][]float32
	dimensions int
	err        error
}

func NewMockEmbedder() *MockEmbedder {
	return &MockEmbedder{
		embeddings: [][]float32{{0.1, 0.2, 0.3, 0.4, 0.5}},
		dimensions: 5,
	}
}

func (m *MockEmbedder) SetEmbeddings(embeddings [][]float32) {
	m.embeddings = embeddings
}

func (m *MockEmbedder) SetError(err error) {
	m.err = err
}

func (m *MockEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.embeddings, nil
}

func (m *MockEmbedder) EmbedSingle(ctx context.Context, text string) ([]float32, error) {
	if m.err != nil {
		return nil, m.err
	}
	if len(m.embeddings) > 0 {
		return m.embeddings[0], nil
	}
	return nil, nil
}

func (m *MockEmbedder) Dimensions() int {
	return m.dimensions
}

func (m *MockEmbedder) Close() error {
	return nil
}

// Test functions

func TestNewSearcher(t *testing.T) {
	mockDriver := NewMockGraphDriver()
	mockEmbedder := NewMockEmbedder()

	searcher := NewSearcher(mockDriver, mockEmbedder, nil)
	if searcher == nil {
		t.Fatal("expected non-nil searcher")
	}

	if searcher.driver == nil {
		t.Error("expected driver to be set")
	}

	if searcher.embedder == nil {
		t.Error("expected embedder to be set")
	}
}

func TestSetCrossEncoder(t *testing.T) {
	searcher := NewSearcher(NewMockGraphDriver(), NewMockEmbedder(), nil)
	if searcher.crossEncoder != nil {
		t.Error("expected crossEncoder to be nil initially")
	}
	// Note: We don't set a mock cross encoder here as it would require more complex mocking
}

func TestSearchEmptyQuery(t *testing.T) {
	mockDriver := NewMockGraphDriver()
	mockEmbedder := NewMockEmbedder()
	searcher := NewSearcher(mockDriver, mockEmbedder, nil)

	ctx := context.Background()
	config := &SearchConfig{
		NodeConfig: &NodeSearchConfig{
			SearchMethods: []SearchMethod{BM25},
		},
		Limit: 10,
	}

	result, err := searcher.Search(ctx, "", config, &SearchFilters{}, "group1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result == nil {
		t.Fatal("expected non-nil result")
	}

	if len(result.Nodes) != 0 {
		t.Errorf("expected 0 nodes for empty query, got %d", len(result.Nodes))
	}

	// Test with whitespace-only query
	result, err = searcher.Search(ctx, "   ", config, &SearchFilters{}, "group1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Nodes) != 0 {
		t.Errorf("expected 0 nodes for whitespace query, got %d", len(result.Nodes))
	}
}

func TestSearchNodesBM25(t *testing.T) {
	mockDriver := NewMockGraphDriver()
	mockEmbedder := NewMockEmbedder()

	// Set up mock search results
	testNodes := []*types.Node{
		{Uuid: "node1", Name: "Test Node 1", GroupID: "group1", Type: types.EntityNodeType},
		{Uuid: "node2", Name: "Test Node 2", GroupID: "group1", Type: types.EntityNodeType},
	}
	mockDriver.SetSearchResults(testNodes, nil)

	searcher := NewSearcher(mockDriver, mockEmbedder, nil)

	ctx := context.Background()
	config := &SearchConfig{
		NodeConfig: &NodeSearchConfig{
			SearchMethods: []SearchMethod{BM25},
			Reranker:      RRFRerankType,
		},
		Limit: 10,
	}

	result, err := searcher.Search(ctx, "test query", config, &SearchFilters{}, "group1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result == nil {
		t.Fatal("expected non-nil result")
	}

	if len(result.Nodes) != 2 {
		t.Errorf("expected 2 nodes, got %d", len(result.Nodes))
	}

	if result.Query != "test query" {
		t.Errorf("expected query 'test query', got '%s'", result.Query)
	}
}

func TestSearchEdgesBM25(t *testing.T) {
	mockDriver := NewMockGraphDriver()
	mockEmbedder := NewMockEmbedder()

	// Set up mock search results
	testEdges := []*types.Edge{
		{BaseEdge: types.BaseEdge{Uuid: "edge1", GroupID: "group1"}, Name: "Test Edge 1", SourceID: "node1", TargetID: "node2"},
		{BaseEdge: types.BaseEdge{Uuid: "edge2", GroupID: "group1"}, Name: "Test Edge 2", SourceID: "node2", TargetID: "node3"},
	}
	mockDriver.SetSearchResults(nil, testEdges)

	searcher := NewSearcher(mockDriver, mockEmbedder, nil)

	ctx := context.Background()
	config := &SearchConfig{
		EdgeConfig: &EdgeSearchConfig{
			SearchMethods: []SearchMethod{BM25},
			Reranker:      RRFRerankType,
		},
		Limit: 10,
	}

	result, err := searcher.Search(ctx, "test query", config, &SearchFilters{}, "group1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Edges) != 2 {
		t.Errorf("expected 2 edges, got %d", len(result.Edges))
	}
}

func TestSearchNodesCosineSimilarity(t *testing.T) {
	mockDriver := NewMockGraphDriver()
	mockEmbedder := NewMockEmbedder()
	mockEmbedder.SetEmbeddings([][]float32{{0.1, 0.2, 0.3, 0.4, 0.5}})

	// Set up mock vector search results
	testNodes := []*types.Node{
		{Uuid: "node1", Name: "Semantic Node 1", GroupID: "group1", Type: types.EntityNodeType},
	}
	mockDriver.SetVectorSearchResults(testNodes, nil)

	searcher := NewSearcher(mockDriver, mockEmbedder, nil)

	ctx := context.Background()
	config := &SearchConfig{
		NodeConfig: &NodeSearchConfig{
			SearchMethods: []SearchMethod{CosineSimilarity},
			Reranker:      RRFRerankType,
			MinScore:      0.5,
		},
		Limit: 10,
	}

	result, err := searcher.Search(ctx, "semantic query", config, &SearchFilters{}, "group1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Nodes) != 1 {
		t.Errorf("expected 1 node, got %d", len(result.Nodes))
	}
}

func TestSearchHybrid(t *testing.T) {
	mockDriver := NewMockGraphDriver()
	mockEmbedder := NewMockEmbedder()
	mockEmbedder.SetEmbeddings([][]float32{{0.1, 0.2, 0.3, 0.4, 0.5}})

	// Set up both BM25 and vector search results
	bm25Nodes := []*types.Node{
		{Uuid: "node1", Name: "BM25 Node", GroupID: "group1", Type: types.EntityNodeType},
	}
	vectorNodes := []*types.Node{
		{Uuid: "node2", Name: "Vector Node", GroupID: "group1", Type: types.EntityNodeType},
	}

	// First call returns BM25 results, then vector results
	mockDriver.SetSearchResults(bm25Nodes, nil)
	mockDriver.SetVectorSearchResults(vectorNodes, nil)

	searcher := NewSearcher(mockDriver, mockEmbedder, nil)

	ctx := context.Background()
	config := &SearchConfig{
		NodeConfig: &NodeSearchConfig{
			SearchMethods: []SearchMethod{BM25, CosineSimilarity},
			Reranker:      RRFRerankType,
		},
		Limit: 10,
	}

	result, err := searcher.Search(ctx, "hybrid query", config, &SearchFilters{}, "group1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should have 2 unique nodes from both methods
	if len(result.Nodes) != 2 {
		t.Errorf("expected 2 nodes from hybrid search, got %d", len(result.Nodes))
	}
}

func TestNeedsEmbedding(t *testing.T) {
	searcher := NewSearcher(NewMockGraphDriver(), NewMockEmbedder(), nil)

	tests := []struct {
		name     string
		config   *SearchConfig
		expected bool
	}{
		{
			name: "BM25 only - no embedding needed",
			config: &SearchConfig{
				NodeConfig: &NodeSearchConfig{
					SearchMethods: []SearchMethod{BM25},
				},
			},
			expected: false,
		},
		{
			name: "Cosine similarity - embedding needed",
			config: &SearchConfig{
				NodeConfig: &NodeSearchConfig{
					SearchMethods: []SearchMethod{CosineSimilarity},
				},
			},
			expected: true,
		},
		{
			name: "MMR reranker - embedding needed",
			config: &SearchConfig{
				NodeConfig: &NodeSearchConfig{
					SearchMethods: []SearchMethod{BM25},
					Reranker:      MMRRerankType,
				},
			},
			expected: true,
		},
		{
			name: "Edge cosine similarity - embedding needed",
			config: &SearchConfig{
				EdgeConfig: &EdgeSearchConfig{
					SearchMethods: []SearchMethod{CosineSimilarity},
				},
			},
			expected: true,
		},
		{
			name: "Community cosine similarity - embedding needed",
			config: &SearchConfig{
				CommunityConfig: &CommunitySearchConfig{
					SearchMethods: []SearchMethod{CosineSimilarity},
				},
			},
			expected: true,
		},
		{
			name:     "Empty config - no embedding needed",
			config:   &SearchConfig{},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := searcher.needsEmbedding(tt.config)
			if result != tt.expected {
				t.Errorf("expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestRRFRerankNodes(t *testing.T) {
	searcher := NewSearcher(NewMockGraphDriver(), NewMockEmbedder(), nil)

	// Create search results from multiple methods
	results1 := []*types.Node{
		{Uuid: "node1", Name: "First Result"},
		{Uuid: "node2", Name: "Second Result"},
	}
	results2 := []*types.Node{
		{Uuid: "node2", Name: "Second Result"}, // Same node ranked first here
		{Uuid: "node3", Name: "Third Result"},
	}

	searchResults := [][]*types.Node{results1, results2}

	nodes, scores, err := searcher.rrfRerankNodes(searchResults, 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// node2 should be ranked first because it appears in both result sets
	if len(nodes) != 3 {
		t.Errorf("expected 3 unique nodes, got %d", len(nodes))
	}

	// Check that scores are in descending order
	for i := 1; i < len(scores); i++ {
		if scores[i] > scores[i-1] {
			t.Error("scores should be in descending order")
		}
	}
}

func TestRRFRerankEdges(t *testing.T) {
	searcher := NewSearcher(NewMockGraphDriver(), NewMockEmbedder(), nil)

	results1 := []*types.Edge{
		{BaseEdge: types.BaseEdge{Uuid: "edge1"}, Name: "First Edge"},
		{BaseEdge: types.BaseEdge{Uuid: "edge2"}, Name: "Second Edge"},
	}
	results2 := []*types.Edge{
		{BaseEdge: types.BaseEdge{Uuid: "edge2"}, Name: "Second Edge"},
		{BaseEdge: types.BaseEdge{Uuid: "edge3"}, Name: "Third Edge"},
	}

	searchResults := [][]*types.Edge{results1, results2}

	edges, scores, err := searcher.rrfRerankEdges(searchResults, 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(edges) != 3 {
		t.Errorf("expected 3 unique edges, got %d", len(edges))
	}

	// Check that scores are in descending order
	for i := 1; i < len(scores); i++ {
		if scores[i] > scores[i-1] {
			t.Error("scores should be in descending order")
		}
	}
}

func TestSearchWithLimit(t *testing.T) {
	mockDriver := NewMockGraphDriver()
	mockEmbedder := NewMockEmbedder()

	// Create many nodes
	var testNodes []*types.Node
	for i := 0; i < 20; i++ {
		testNodes = append(testNodes, &types.Node{
			Uuid:    "node" + string(rune('A'+i)),
			Name:    "Test Node",
			GroupID: "group1",
		})
	}
	mockDriver.SetSearchResults(testNodes, nil)

	searcher := NewSearcher(mockDriver, mockEmbedder, nil)

	ctx := context.Background()
	config := &SearchConfig{
		NodeConfig: &NodeSearchConfig{
			SearchMethods: []SearchMethod{BM25},
		},
		Limit: 5,
	}

	result, err := searcher.Search(ctx, "test", config, &SearchFilters{}, "group1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(result.Nodes) > 5 {
		t.Errorf("expected at most 5 nodes due to limit, got %d", len(result.Nodes))
	}
}

func TestSearchFilters(t *testing.T) {
	mockDriver := NewMockGraphDriver()
	mockEmbedder := NewMockEmbedder()

	testNodes := []*types.Node{
		{Uuid: "node1", Name: "Entity Node", GroupID: "group1", Type: types.EntityNodeType},
	}
	mockDriver.SetSearchResults(testNodes, nil)

	searcher := NewSearcher(mockDriver, mockEmbedder, nil)

	ctx := context.Background()
	config := &SearchConfig{
		NodeConfig: &NodeSearchConfig{
			SearchMethods: []SearchMethod{BM25},
		},
		Limit: 10,
	}

	filters := &SearchFilters{
		NodeTypes: []types.NodeType{types.EntityNodeType},
		GroupIDs:  []string{"group1"},
	}

	result, err := searcher.Search(ctx, "test", config, filters, "group1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify search completed (actual filtering happens in driver)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
}

func TestRerankNodesEmptyResults(t *testing.T) {
	searcher := NewSearcher(NewMockGraphDriver(), NewMockEmbedder(), nil)

	ctx := context.Background()
	config := &NodeSearchConfig{
		Reranker: RRFRerankType,
	}

	nodes, scores, err := searcher.rerankNodes(ctx, "query", nil, [][]*types.Node{}, config, 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(nodes) != 0 {
		t.Errorf("expected 0 nodes for empty results, got %d", len(nodes))
	}

	if len(scores) != 0 {
		t.Errorf("expected 0 scores for empty results, got %d", len(scores))
	}
}

func TestRerankEdgesEmptyResults(t *testing.T) {
	searcher := NewSearcher(NewMockGraphDriver(), NewMockEmbedder(), nil)

	ctx := context.Background()
	config := &EdgeSearchConfig{
		Reranker: RRFRerankType,
	}

	edges, scores, err := searcher.rerankEdges(ctx, "query", nil, [][]*types.Edge{}, config, 10)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(edges) != 0 {
		t.Errorf("expected 0 edges for empty results, got %d", len(edges))
	}

	if len(scores) != 0 {
		t.Errorf("expected 0 scores for empty results, got %d", len(scores))
	}
}

func TestMinFunction(t *testing.T) {
	tests := []struct {
		a, b     int
		expected int
	}{
		{5, 10, 5},
		{10, 5, 5},
		{5, 5, 5},
		{0, 5, 0},
		{-5, 5, -5},
	}

	for _, tt := range tests {
		result := min(tt.a, tt.b)
		if result != tt.expected {
			t.Errorf("min(%d, %d) = %d, expected %d", tt.a, tt.b, result, tt.expected)
		}
	}
}

func TestSearchMethodConstants(t *testing.T) {
	// Verify search method constants are as expected
	if CosineSimilarity != "cosine_similarity" {
		t.Errorf("expected CosineSimilarity = 'cosine_similarity', got '%s'", CosineSimilarity)
	}

	if BM25 != "bm25" {
		t.Errorf("expected BM25 = 'bm25', got '%s'", BM25)
	}

	if BreadthFirstSearch != "bfs" {
		t.Errorf("expected BreadthFirstSearch = 'bfs', got '%s'", BreadthFirstSearch)
	}
}

func TestRerankerTypeConstants(t *testing.T) {
	// Verify reranker type constants
	if RRFRerankType != "rrf" {
		t.Errorf("expected RRFRerankType = 'rrf', got '%s'", RRFRerankType)
	}

	if MMRRerankType != "mmr" {
		t.Errorf("expected MMRRerankType = 'mmr', got '%s'", MMRRerankType)
	}

	if CrossEncoderRerankType != "cross_encoder" {
		t.Errorf("expected CrossEncoderRerankType = 'cross_encoder', got '%s'", CrossEncoderRerankType)
	}
}

func TestHybridSearchResultStruct(t *testing.T) {
	result := &HybridSearchResult{
		Nodes: []*types.Node{
			{Uuid: "node1", Name: "Test Node"},
		},
		Edges: []*types.Edge{
			{BaseEdge: types.BaseEdge{Uuid: "edge1"}, Name: "Test Edge"},
		},
		NodeScores: []float64{0.9},
		EdgeScores: []float64{0.8},
		Query:      "test query",
		Total:      2,
	}

	if len(result.Nodes) != 1 {
		t.Errorf("expected 1 node, got %d", len(result.Nodes))
	}

	if len(result.Edges) != 1 {
		t.Errorf("expected 1 edge, got %d", len(result.Edges))
	}

	if result.Query != "test query" {
		t.Errorf("expected query 'test query', got '%s'", result.Query)
	}

	if result.Total != 2 {
		t.Errorf("expected total 2, got %d", result.Total)
	}
}

func TestSearchConfigStructs(t *testing.T) {
	// Test NodeSearchConfig
	nodeConfig := &NodeSearchConfig{
		SearchMethods: []SearchMethod{BM25, CosineSimilarity},
		Reranker:      RRFRerankType,
		MinScore:      0.5,
		MMRLambda:     0.7,
		MaxDepth:      3,
	}

	if len(nodeConfig.SearchMethods) != 2 {
		t.Errorf("expected 2 search methods, got %d", len(nodeConfig.SearchMethods))
	}

	// Test EdgeSearchConfig
	edgeConfig := &EdgeSearchConfig{
		SearchMethods: []SearchMethod{BM25},
		Reranker:      MMRRerankType,
		MinScore:      0.6,
	}

	if edgeConfig.Reranker != MMRRerankType {
		t.Errorf("expected MMR reranker, got %s", edgeConfig.Reranker)
	}

	// Test SearchFilters
	filters := &SearchFilters{
		GroupIDs:    []string{"group1", "group2"},
		NodeTypes:   []types.NodeType{types.EntityNodeType},
		EdgeTypes:   []types.EdgeType{types.EntityEdgeType},
		EntityTypes: []string{"person", "organization"},
	}

	if len(filters.GroupIDs) != 2 {
		t.Errorf("expected 2 group IDs, got %d", len(filters.GroupIDs))
	}
}
