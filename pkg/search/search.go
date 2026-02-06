package search

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/soundprediction/predicato/pkg/crossencoder"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/embedder"
	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/types"
)

type SearchMethod string

const (
	CosineSimilarity   SearchMethod = "cosine_similarity"
	BM25               SearchMethod = "bm25"
	BreadthFirstSearch SearchMethod = "bfs"
)

type RerankerType string

const (
	RRFRerankType             RerankerType = "rrf"
	MMRRerankType             RerankerType = "mmr"
	CrossEncoderRerankType    RerankerType = "cross_encoder"
	NodeDistanceRerankType    RerankerType = "node_distance"
	EpisodeMentionsRerankType RerankerType = "episode_mentions"
)

type SearchConfig struct {
	NodeConfig      *NodeSearchConfig      `json:"node_config,omitempty"`
	EdgeConfig      *EdgeSearchConfig      `json:"edge_config,omitempty"`
	EpisodeConfig   *EpisodeSearchConfig   `json:"episode_config,omitempty"`
	CommunityConfig *CommunitySearchConfig `json:"community_config,omitempty"`
	Limit           int                    `json:"limit"`
	MinScore        float64                `json:"min_score"`
}

type NodeSearchConfig struct {
	Reranker      RerankerType   `json:"reranker"`
	SearchMethods []SearchMethod `json:"search_methods"`
	MinScore      float64        `json:"min_score"`
	MMRLambda     float64        `json:"mmr_lambda"`
	MaxDepth      int            `json:"max_depth"`
}

type EdgeSearchConfig struct {
	Reranker      RerankerType   `json:"reranker"`
	SearchMethods []SearchMethod `json:"search_methods"`
	MinScore      float64        `json:"min_score"`
	MMRLambda     float64        `json:"mmr_lambda"`
	MaxDepth      int            `json:"max_depth"`
}

type EpisodeSearchConfig struct {
	Reranker      RerankerType   `json:"reranker"`
	SearchMethods []SearchMethod `json:"search_methods"`
	MinScore      float64        `json:"min_score"`
}

type CommunitySearchConfig struct {
	Reranker      RerankerType   `json:"reranker"`
	SearchMethods []SearchMethod `json:"search_methods"`
	MinScore      float64        `json:"min_score"`
	MMRLambda     float64        `json:"mmr_lambda"`
}

type SearchFilters struct {
	TimeRange   *types.TimeRange `json:"time_range,omitempty"`
	GroupIDs    []string         `json:"group_ids,omitempty"`
	NodeTypes   []types.NodeType `json:"node_types,omitempty"`
	EdgeTypes   []types.EdgeType `json:"edge_types,omitempty"`
	EntityTypes []string         `json:"entity_types,omitempty"`
}

type HybridSearchResult struct {
	Query      string        `json:"query"`
	Nodes      []*types.Node `json:"nodes"`
	Edges      []*types.Edge `json:"edges"`
	NodeScores []float64     `json:"node_scores"`
	EdgeScores []float64     `json:"edge_scores"`
	Total      int           `json:"total"`
}

type Searcher struct {
	driver       driver.GraphDriver
	embedder     embedder.Client
	nlProcessor  nlp.Client
	crossEncoder crossencoder.Client
}

// NewSearcher creates a new Searcher instance.
// Note: While Searcher primarily uses search methods, it also uses graph traversal
// for BFS-based search expansion, which requires the full GraphDriver interface.
// For search-only use cases, consider using the driver's search methods directly
// via the driver.GraphSearcher interface.
func NewSearcher(driver driver.GraphDriver, embedder embedder.Client, nlProcessor nlp.Client) *Searcher {
	return &Searcher{
		driver:       driver,
		embedder:     embedder,
		nlProcessor:  nlProcessor,
		crossEncoder: nil, // Will be set separately if needed
	}
}

// SetCrossEncoder sets the cross-encoder client for advanced reranking
func (s *Searcher) SetCrossEncoder(crossEncoder crossencoder.Client) {
	s.crossEncoder = crossEncoder
}

func (s *Searcher) Search(ctx context.Context, query string, config *SearchConfig, filters *SearchFilters, groupID string) (*HybridSearchResult, error) {
	if strings.TrimSpace(query) == "" {
		return &HybridSearchResult{}, nil
	}

	// Generate query embedding if needed for semantic search
	var queryVector []float32
	needsEmbedding := s.needsEmbedding(config)

	if needsEmbedding {
		vectors, err := s.embedder.Embed(ctx, []string{strings.ReplaceAll(query, "\n", " ")})
		if err != nil {
			return nil, fmt.Errorf("failed to create query embedding: %w", err)
		}
		if len(vectors) > 0 {
			queryVector = vectors[0]
		}
	}

	// Perform searches concurrently
	nodeResults := make([]*types.Node, 0)
	edgeResults := make([]*types.Edge, 0)
	nodeScores := make([]float64, 0)
	edgeScores := make([]float64, 0)

	// Node search
	if config.NodeConfig != nil {
		nodes, scores, err := s.searchNodes(ctx, query, queryVector, config.NodeConfig, filters, groupID, config.Limit)
		if err != nil {
			return nil, fmt.Errorf("node search failed: %w", err)
		}
		nodeResults = nodes
		nodeScores = scores
	}

	// Edge search
	if config.EdgeConfig != nil {
		edges, scores, err := s.searchEdges(ctx, query, queryVector, config.EdgeConfig, filters, groupID, config.Limit)
		if err != nil {
			return nil, fmt.Errorf("edge search failed: %w", err)
		}
		edgeResults = edges
		edgeScores = scores
	}

	return &HybridSearchResult{
		Nodes:      nodeResults,
		Edges:      edgeResults,
		NodeScores: nodeScores,
		EdgeScores: edgeScores,
		Query:      query,
		Total:      len(nodeResults) + len(edgeResults),
	}, nil
}

func (s *Searcher) needsEmbedding(config *SearchConfig) bool {
	if config.NodeConfig != nil {
		for _, method := range config.NodeConfig.SearchMethods {
			if method == CosineSimilarity {
				return true
			}
		}
		if config.NodeConfig.Reranker == MMRRerankType {
			return true
		}
	}

	if config.EdgeConfig != nil {
		for _, method := range config.EdgeConfig.SearchMethods {
			if method == CosineSimilarity {
				return true
			}
		}
		if config.EdgeConfig.Reranker == MMRRerankType {
			return true
		}
	}

	if config.CommunityConfig != nil {
		for _, method := range config.CommunityConfig.SearchMethods {
			if method == CosineSimilarity {
				return true
			}
		}
		if config.CommunityConfig.Reranker == MMRRerankType {
			return true
		}
	}

	return false
}

func (s *Searcher) searchNodes(ctx context.Context, query string, queryVector []float32, config *NodeSearchConfig, filters *SearchFilters, groupID string, limit int) ([]*types.Node, []float64, error) {
	// Pre-allocate with capacity for search methods (may add BFS results too)
	searchResults := make([][]*types.Node, 0, len(config.SearchMethods)+1)
	var bfsOriginNodes []string

	// Execute different search methods
	for _, method := range config.SearchMethods {
		switch method {
		case BM25:
			nodes, err := s.nodeFulltextSearch(ctx, query, filters, groupID, limit*2)
			if err != nil {
				return nil, nil, fmt.Errorf("BM25 node search failed: %w", err)
			}
			searchResults = append(searchResults, nodes)
			// Collect UUIDs for BFS
			for _, node := range nodes {
				bfsOriginNodes = append(bfsOriginNodes, node.Uuid)
			}

		case CosineSimilarity:
			if len(queryVector) == 0 {
				continue
			}
			nodes, err := s.nodeSimilaritySearch(ctx, queryVector, filters, groupID, limit*2, config.MinScore)
			if err != nil {
				return nil, nil, fmt.Errorf("similarity node search failed: %w", err)
			}
			searchResults = append(searchResults, nodes)
			// Collect UUIDs for BFS
			for _, node := range nodes {
				bfsOriginNodes = append(bfsOriginNodes, node.Uuid)
			}

		case BreadthFirstSearch:
			// BFS will be executed after other methods if origin nodes are available
			continue
		}
	}

	// If BFS is requested and we have origin nodes from other searches, execute BFS
	hasBFS := false
	for _, method := range config.SearchMethods {
		if method == BreadthFirstSearch {
			hasBFS = true
			break
		}
	}

	if hasBFS && len(bfsOriginNodes) > 0 {
		// Create search utilities for BFS
		searchUtils := NewSearchUtilities(s.driver)
		maxDepth := config.MaxDepth
		if maxDepth == 0 {
			maxDepth = MaxSearchDepth
		}

		bfsOptions := &BFSSearchOptions{
			MaxDepth:      maxDepth,
			Limit:         limit * 2,
			SearchFilters: filters,
			GroupIDs:      []string{groupID},
		}

		bfsNodes, err := searchUtils.NodeBFSSearch(ctx, bfsOriginNodes, bfsOptions)
		if err != nil {
			return nil, nil, fmt.Errorf("BFS node search failed: %w", err)
		}
		if len(bfsNodes) > 0 {
			searchResults = append(searchResults, bfsNodes)
		}
	}

	// Combine and rerank results
	return s.rerankNodes(ctx, query, queryVector, searchResults, config, limit)
}

func (s *Searcher) searchEdges(ctx context.Context, query string, queryVector []float32, config *EdgeSearchConfig, filters *SearchFilters, groupID string, limit int) ([]*types.Edge, []float64, error) {
	// Pre-allocate with capacity for search methods (may add BFS results too)
	searchResults := make([][]*types.Edge, 0, len(config.SearchMethods)+1)
	var bfsOriginNodes []string

	// Execute different search methods
	for _, method := range config.SearchMethods {
		switch method {
		case BM25:
			edges, err := s.edgeFulltextSearch(ctx, query, filters, groupID, limit*2)
			if err != nil {
				return nil, nil, fmt.Errorf("BM25 edge search failed: %w", err)
			}
			searchResults = append(searchResults, edges)
			// Collect source node UUIDs for BFS
			for _, edge := range edges {
				bfsOriginNodes = append(bfsOriginNodes, edge.SourceID)
			}

		case CosineSimilarity:
			if len(queryVector) == 0 {
				continue
			}
			edges, err := s.edgeSimilaritySearch(ctx, queryVector, filters, groupID, limit*2, config.MinScore)
			if err != nil {
				return nil, nil, fmt.Errorf("similarity edge search failed: %w", err)
			}
			searchResults = append(searchResults, edges)
			// Collect source node UUIDs for BFS
			for _, edge := range edges {
				bfsOriginNodes = append(bfsOriginNodes, edge.SourceID)
			}

		case BreadthFirstSearch:
			// BFS will be executed after other methods if origin nodes are available
			continue
		}
	}

	// If BFS is requested and we have origin nodes from other searches, execute BFS
	hasBFS := false
	for _, method := range config.SearchMethods {
		if method == BreadthFirstSearch {
			hasBFS = true
			break
		}
	}

	if hasBFS && len(bfsOriginNodes) > 0 {
		// Create search utilities for BFS
		searchUtils := NewSearchUtilities(s.driver)
		maxDepth := config.MaxDepth
		if maxDepth == 0 {
			maxDepth = MaxSearchDepth
		}

		bfsOptions := &BFSSearchOptions{
			MaxDepth:      maxDepth,
			Limit:         limit * 2,
			SearchFilters: filters,
			GroupIDs:      []string{groupID},
		}

		bfsEdges, err := searchUtils.EdgeBFSSearch(ctx, bfsOriginNodes, bfsOptions)
		if err != nil {
			return nil, nil, fmt.Errorf("BFS edge search failed: %w", err)
		}
		if len(bfsEdges) > 0 {
			searchResults = append(searchResults, bfsEdges)
		}
	}

	// Combine and rerank results
	return s.rerankEdges(ctx, query, queryVector, searchResults, config, limit)
}

func (s *Searcher) nodeFulltextSearch(ctx context.Context, query string, filters *SearchFilters, groupID string, limit int) ([]*types.Node, error) {
	// This would use the driver's fulltext search capabilities
	// For now, return a basic implementation
	return s.driver.SearchNodes(ctx, query, groupID, &driver.SearchOptions{
		Limit:       limit,
		UseFullText: true,
		NodeTypes:   filters.NodeTypes,
	})
}

func (s *Searcher) nodeSimilaritySearch(ctx context.Context, queryVector []float32, filters *SearchFilters, groupID string, limit int, minScore float64) ([]*types.Node, error) {
	// This would use vector similarity search
	return s.driver.SearchNodesByVector(ctx, queryVector, groupID, &driver.VectorSearchOptions{
		Limit:     limit,
		MinScore:  minScore,
		NodeTypes: filters.NodeTypes,
	})
}

func (s *Searcher) edgeFulltextSearch(ctx context.Context, query string, filters *SearchFilters, groupID string, limit int) ([]*types.Edge, error) {
	return s.driver.SearchEdges(ctx, query, groupID, &driver.SearchOptions{
		Limit:       limit,
		UseFullText: true,
		EdgeTypes:   filters.EdgeTypes,
	})
}

func (s *Searcher) edgeSimilaritySearch(ctx context.Context, queryVector []float32, filters *SearchFilters, groupID string, limit int, minScore float64) ([]*types.Edge, error) {
	return s.driver.SearchEdgesByVector(ctx, queryVector, groupID, &driver.VectorSearchOptions{
		Limit:     limit,
		MinScore:  minScore,
		EdgeTypes: filters.EdgeTypes,
	})
}

func (s *Searcher) rerankNodes(ctx context.Context, query string, queryVector []float32, searchResults [][]*types.Node, config *NodeSearchConfig, limit int) ([]*types.Node, []float64, error) {
	if len(searchResults) == 0 {
		return []*types.Node{}, []float64{}, nil
	}

	// Create node map for deduplication
	nodeMap := make(map[string]*types.Node)
	for _, results := range searchResults {
		for _, node := range results {
			nodeMap[node.Uuid] = node
		}
	}

	nodes := make([]*types.Node, 0, len(nodeMap))
	for _, node := range nodeMap {
		nodes = append(nodes, node)
	}

	switch config.Reranker {
	case RRFRerankType:
		return s.rrfRerankNodes(searchResults, limit)
	case MMRRerankType:
		return s.mmrRerankNodes(ctx, queryVector, nodes, config.MMRLambda, config.MinScore, limit)
	case CrossEncoderRerankType:
		return s.crossEncoderRerankNodes(ctx, query, nodes, config.MinScore, limit)
	default:
		// Default to simple score-based ranking
		scores := make([]float64, len(nodes))
		for i := range scores {
			scores[i] = 1.0 // Default score
		}
		return nodes[:min(limit, len(nodes))], scores[:min(limit, len(scores))], nil
	}
}

func (s *Searcher) rerankEdges(ctx context.Context, query string, queryVector []float32, searchResults [][]*types.Edge, config *EdgeSearchConfig, limit int) ([]*types.Edge, []float64, error) {
	if len(searchResults) == 0 {
		return []*types.Edge{}, []float64{}, nil
	}

	// Create edge map for deduplication
	edgeMap := make(map[string]*types.Edge)
	for _, results := range searchResults {
		for _, edge := range results {
			edgeMap[edge.Uuid] = edge
		}
	}

	edges := make([]*types.Edge, 0, len(edgeMap))
	for _, edge := range edgeMap {
		edges = append(edges, edge)
	}

	switch config.Reranker {
	case RRFRerankType:
		return s.rrfRerankEdges(searchResults, limit)
	case MMRRerankType:
		return s.mmrRerankEdges(ctx, queryVector, edges, config.MMRLambda, config.MinScore, limit)
	case CrossEncoderRerankType:
		return s.crossEncoderRerankEdges(ctx, query, edges, config.MinScore, limit)
	default:
		// Default to simple score-based ranking
		scores := make([]float64, len(edges))
		for i := range scores {
			scores[i] = 1.0 // Default score
		}
		return edges[:min(limit, len(edges))], scores[:min(limit, len(scores))], nil
	}
}

// RRF (Reciprocal Rank Fusion) reranking
func (s *Searcher) rrfRerankNodes(searchResults [][]*types.Node, limit int) ([]*types.Node, []float64, error) {
	scoreMap := make(map[string]float64)
	nodeMap := make(map[string]*types.Node)

	for _, results := range searchResults {
		for rank, node := range results {
			if _, exists := scoreMap[node.Uuid]; !exists {
				scoreMap[node.Uuid] = 0
			}
			// RRF formula: 1 / (rank + k), where k is typically 60
			scoreMap[node.Uuid] += 1.0 / float64(rank+60)
			nodeMap[node.Uuid] = node
		}
	}

	// Sort by score
	type nodeScore struct {
		node  *types.Node
		score float64
	}

	nodeScores := make([]nodeScore, 0, len(scoreMap))
	for id, score := range scoreMap {
		nodeScores = append(nodeScores, nodeScore{
			node:  nodeMap[id],
			score: score,
		})
	}

	sort.Slice(nodeScores, func(i, j int) bool {
		return nodeScores[i].score > nodeScores[j].score
	})

	// Extract results
	nodes := make([]*types.Node, 0, min(limit, len(nodeScores)))
	scores := make([]float64, 0, min(limit, len(nodeScores)))

	for i := 0; i < min(limit, len(nodeScores)); i++ {
		nodes = append(nodes, nodeScores[i].node)
		scores = append(scores, nodeScores[i].score)
	}

	return nodes, scores, nil
}

func (s *Searcher) rrfRerankEdges(searchResults [][]*types.Edge, limit int) ([]*types.Edge, []float64, error) {
	scoreMap := make(map[string]float64)
	edgeMap := make(map[string]*types.Edge)

	for _, results := range searchResults {
		for rank, edge := range results {
			if _, exists := scoreMap[edge.Uuid]; !exists {
				scoreMap[edge.Uuid] = 0
			}
			// RRF formula: 1 / (rank + k), where k is typically 60
			scoreMap[edge.Uuid] += 1.0 / float64(rank+60)
			edgeMap[edge.Uuid] = edge
		}
	}

	// Sort by score
	type edgeScore struct {
		edge  *types.Edge
		score float64
	}

	edgeScores := make([]edgeScore, 0, len(scoreMap))
	for id, score := range scoreMap {
		edgeScores = append(edgeScores, edgeScore{
			edge:  edgeMap[id],
			score: score,
		})
	}

	sort.Slice(edgeScores, func(i, j int) bool {
		return edgeScores[i].score > edgeScores[j].score
	})

	// Extract results
	edges := make([]*types.Edge, 0, min(limit, len(edgeScores)))
	scores := make([]float64, 0, min(limit, len(edgeScores)))

	for i := 0; i < min(limit, len(edgeScores)); i++ {
		edges = append(edges, edgeScores[i].edge)
		scores = append(scores, edgeScores[i].score)
	}

	return edges, scores, nil
}

// MMR (Maximal Marginal Relevance) reranking
func (s *Searcher) mmrRerankNodes(ctx context.Context, queryVector []float32, nodes []*types.Node, lambda float64, minScore float64, limit int) ([]*types.Node, []float64, error) {
	if len(queryVector) == 0 {
		return nodes[:min(limit, len(nodes))], make([]float64, min(limit, len(nodes))), nil
	}

	// Get embeddings for all nodes
	embeddings, err := GetEmbeddingsForNodes(ctx, s.driver, nodes)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get embeddings for nodes: %w", err)
	}

	// If no embeddings available, fall back to default ranking
	if len(embeddings) == 0 {
		scores := make([]float64, min(limit, len(nodes)))
		for i := range scores {
			scores[i] = 1.0
		}
		return nodes[:min(limit, len(nodes))], scores, nil
	}

	// Apply MMR reranking
	mmrUUIDs, mmrScores := MaximalMarginalRelevance(queryVector, embeddings, lambda, minScore)

	// Create node map for lookup
	nodeMap := make(map[string]*types.Node)
	for _, node := range nodes {
		nodeMap[node.Uuid] = node
	}

	// Build result arrays based on MMR ranking
	var resultNodes []*types.Node
	var resultScores []float64

	for i, uuid := range mmrUUIDs {
		if node, exists := nodeMap[uuid]; exists {
			resultNodes = append(resultNodes, node)
			resultScores = append(resultScores, mmrScores[i])
			if len(resultNodes) >= limit {
				break
			}
		}
	}

	return resultNodes, resultScores, nil
}

func (s *Searcher) mmrRerankEdges(ctx context.Context, queryVector []float32, edges []*types.Edge, lambda float64, minScore float64, limit int) ([]*types.Edge, []float64, error) {
	if len(queryVector) == 0 {
		return edges[:min(limit, len(edges))], make([]float64, min(limit, len(edges))), nil
	}

	// Get embeddings for all edges
	embeddings, err := GetEmbeddingsForEdges(ctx, s.driver, edges)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get embeddings for edges: %w", err)
	}

	// If no embeddings available, fall back to default ranking
	if len(embeddings) == 0 {
		scores := make([]float64, min(limit, len(edges)))
		for i := range scores {
			scores[i] = 1.0
		}
		return edges[:min(limit, len(edges))], scores, nil
	}

	// Apply MMR reranking
	mmrUUIDs, mmrScores := MaximalMarginalRelevance(queryVector, embeddings, lambda, minScore)

	// Create edge map for lookup
	edgeMap := make(map[string]*types.Edge)
	for _, edge := range edges {
		edgeMap[edge.Uuid] = edge
	}

	// Build result arrays based on MMR ranking
	var resultEdges []*types.Edge
	var resultScores []float64

	for i, uuid := range mmrUUIDs {
		if edge, exists := edgeMap[uuid]; exists {
			resultEdges = append(resultEdges, edge)
			resultScores = append(resultScores, mmrScores[i])
			if len(resultEdges) >= limit {
				break
			}
		}
	}

	return resultEdges, resultScores, nil
}

// Cross-encoder reranking
func (s *Searcher) crossEncoderRerankNodes(ctx context.Context, query string, nodes []*types.Node, minScore float64, limit int) ([]*types.Node, []float64, error) {
	if s.crossEncoder == nil {
		// Fallback to LLM-based scoring if no cross-encoder available
		return s.fallbackLLMRerankNodes(ctx, query, nodes, minScore, limit)
	}

	if len(nodes) == 0 {
		return []*types.Node{}, []float64{}, nil
	}

	// Prepare passages for reranking
	passages := make([]string, len(nodes))
	for i, node := range nodes {
		nodeContent := node.Summary
		if nodeContent == "" {
			nodeContent = node.Content
		}
		if nodeContent == "" {
			nodeContent = node.Name
		}
		passages[i] = nodeContent
	}

	// Use cross-encoder to rank passages
	rankedPassages, err := s.crossEncoder.Rank(ctx, query, passages)
	if err != nil {
		// Fallback to LLM-based scoring on error
		return s.fallbackLLMRerankNodes(ctx, query, nodes, minScore, limit)
	}

	// Map ranked passages back to nodes
	var resultNodes []*types.Node
	var resultScores []float64

	// Create a map for efficient lookup
	passageToNode := make(map[string]*types.Node)
	for i, passage := range passages {
		passageToNode[passage] = nodes[i]
	}

	// Convert ranked passages to nodes, applying filters
	for _, rankedPassage := range rankedPassages {
		if rankedPassage.Score >= minScore && len(resultNodes) < limit {
			if node, exists := passageToNode[rankedPassage.Passage]; exists {
				resultNodes = append(resultNodes, node)
				resultScores = append(resultScores, rankedPassage.Score)
			}
		}
	}

	return resultNodes, resultScores, nil
}

// fallbackLLMRerankNodes provides LLM-based reranking when cross-encoder is not available
func (s *Searcher) fallbackLLMRerankNodes(ctx context.Context, query string, nodes []*types.Node, minScore float64, limit int) ([]*types.Node, []float64, error) {
	if s.nlProcessor == nil {
		// Ultimate fallback to default scores
		scores := make([]float64, min(limit, len(nodes)))
		for i := range scores {
			scores[i] = 1.0
		}
		return nodes[:min(limit, len(nodes))], scores, nil
	}

	// Create pairs of query and node content for reranking
	type nodeScore struct {
		node  *types.Node
		score float64
		index int
	}

	nodeScores := make([]nodeScore, len(nodes))

	// Process nodes in batches to avoid overwhelming the LLM
	batchSize := 10
	for i := 0; i < len(nodes); i += batchSize {
		end := min(i+batchSize, len(nodes))
		batch := nodes[i:end]

		// Create relevance scoring prompt
		prompt := fmt.Sprintf(`Given the search query and a list of nodes, score each node's relevance to the query from 0.0 to 1.0.

Query: "%s"

Nodes to score:`, query)

		for j, node := range batch {
			nodeContent := node.Summary
			if nodeContent == "" {
				nodeContent = node.Content
			}
			if nodeContent == "" {
				nodeContent = node.Name
			}
			prompt += fmt.Sprintf("\n%d. %s", j+1, nodeContent)
		}

		prompt += `

Please respond with only comma-separated scores for each node in order (e.g., "0.9,0.3,0.7,0.1").
Consider semantic relevance, topical alignment, and contextual importance.`

		messages := []types.Message{
			nlp.NewSystemMessage("You are a relevance scoring system. Score how relevant each node is to the given query."),
			nlp.NewUserMessage(prompt),
		}

		response, err := s.nlProcessor.Chat(ctx, messages)
		if err != nil {
			// On error, assign default scores
			for j := range batch {
				nodeScores[i+j] = nodeScore{
					node:  batch[j],
					score: 0.5, // Default mid-range score
					index: i + j,
				}
			}
			continue
		}

		// Parse scores from response
		scoreStrs := strings.Split(strings.TrimSpace(response.Content), ",")
		for j, scoreStr := range scoreStrs {
			if i+j >= len(nodes) {
				break
			}

			score := 0.5 // Default score
			if parsedScore, err := strconv.ParseFloat(strings.TrimSpace(scoreStr), 64); err == nil {
				score = parsedScore
			}

			nodeScores[i+j] = nodeScore{
				node:  batch[j],
				score: score,
				index: i + j,
			}
		}

		// Handle case where we have more nodes than scores returned
		for j := len(scoreStrs); j < len(batch); j++ {
			nodeScores[i+j] = nodeScore{
				node:  batch[j],
				score: 0.5,
				index: i + j,
			}
		}
	}

	// Sort by score (descending)
	sort.Slice(nodeScores, func(i, j int) bool {
		return nodeScores[i].score > nodeScores[j].score
	})

	// Filter by minimum score and apply limit
	var resultNodes []*types.Node
	var resultScores []float64

	for _, ns := range nodeScores {
		if ns.score >= minScore && len(resultNodes) < limit {
			resultNodes = append(resultNodes, ns.node)
			resultScores = append(resultScores, ns.score)
		}
	}

	return resultNodes, resultScores, nil
}

func (s *Searcher) crossEncoderRerankEdges(ctx context.Context, query string, edges []*types.Edge, minScore float64, limit int) ([]*types.Edge, []float64, error) {
	if s.crossEncoder == nil {
		// Fallback to LLM-based scoring if no cross-encoder available
		return s.fallbackLLMRerankEdges(ctx, query, edges, minScore, limit)
	}

	if len(edges) == 0 {
		return []*types.Edge{}, []float64{}, nil
	}

	// Prepare passages for reranking
	passages := make([]string, len(edges))
	for i, edge := range edges {
		edgeContent := edge.Summary
		if edgeContent == "" {
			edgeContent = edge.Name
		}
		if edgeContent == "" {
			edgeContent = fmt.Sprintf("%s %s %s", edge.SourceID, string(edge.Type), edge.TargetID)
		}
		passages[i] = fmt.Sprintf("%s -> %s: %s", edge.SourceID, edge.TargetID, edgeContent)
	}

	// Use cross-encoder to rank passages
	rankedPassages, err := s.crossEncoder.Rank(ctx, query, passages)
	if err != nil {
		// Fallback to LLM-based scoring on error
		return s.fallbackLLMRerankEdges(ctx, query, edges, minScore, limit)
	}

	// Map ranked passages back to edges
	var resultEdges []*types.Edge
	var resultScores []float64

	// Create a map for efficient lookup
	passageToEdge := make(map[string]*types.Edge)
	for i, passage := range passages {
		passageToEdge[passage] = edges[i]
	}

	// Convert ranked passages to edges, applying filters
	for _, rankedPassage := range rankedPassages {
		if rankedPassage.Score >= minScore && len(resultEdges) < limit {
			if edge, exists := passageToEdge[rankedPassage.Passage]; exists {
				resultEdges = append(resultEdges, edge)
				resultScores = append(resultScores, rankedPassage.Score)
			}
		}
	}

	return resultEdges, resultScores, nil
}

// fallbackLLMRerankEdges provides LLM-based reranking when cross-encoder is not available
func (s *Searcher) fallbackLLMRerankEdges(ctx context.Context, query string, edges []*types.Edge, minScore float64, limit int) ([]*types.Edge, []float64, error) {
	if s.nlProcessor == nil {
		// Ultimate fallback to default scores
		scores := make([]float64, min(limit, len(edges)))
		for i := range scores {
			scores[i] = 1.0
		}
		return edges[:min(limit, len(edges))], scores, nil
	}

	// Create pairs of query and edge content for reranking
	type edgeScore struct {
		edge  *types.Edge
		score float64
		index int
	}

	edgeScores := make([]edgeScore, len(edges))

	// Process edges in batches to avoid overwhelming the LLM
	batchSize := 10
	for i := 0; i < len(edges); i += batchSize {
		end := min(i+batchSize, len(edges))
		batch := edges[i:end]

		// Create relevance scoring prompt
		prompt := fmt.Sprintf(`Given the search query and a list of relationship edges, score each edge's relevance to the query from 0.0 to 1.0.

Query: "%s"

Edges to score (format: source -> target: description):`, query)

		for j, edge := range batch {
			edgeContent := edge.Summary
			if edgeContent == "" {
				edgeContent = edge.Name
			}
			if edgeContent == "" {
				edgeContent = fmt.Sprintf("%s %s %s", edge.SourceID, string(edge.Type), edge.TargetID)
			}
			prompt += fmt.Sprintf("\n%d. %s -> %s: %s", j+1, edge.SourceID, edge.TargetID, edgeContent)
		}

		prompt += `

Please respond with only comma-separated scores for each edge in order (e.g., "0.9,0.3,0.7,0.1").
Consider semantic relevance, relationship importance, and contextual significance.`

		messages := []types.Message{
			nlp.NewSystemMessage("You are a relevance scoring system. Score how relevant each relationship edge is to the given query."),
			nlp.NewUserMessage(prompt),
		}

		response, err := s.nlProcessor.Chat(ctx, messages)
		if err != nil {
			// On error, assign default scores
			for j := range batch {
				edgeScores[i+j] = edgeScore{
					edge:  batch[j],
					score: 0.5, // Default mid-range score
					index: i + j,
				}
			}
			continue
		}

		// Parse scores from response
		scoreStrs := strings.Split(strings.TrimSpace(response.Content), ",")
		for j, scoreStr := range scoreStrs {
			if i+j >= len(edges) {
				break
			}

			score := 0.5 // Default score
			if parsedScore, err := strconv.ParseFloat(strings.TrimSpace(scoreStr), 64); err == nil {
				score = parsedScore
			}

			edgeScores[i+j] = edgeScore{
				edge:  batch[j],
				score: score,
				index: i + j,
			}
		}

		// Handle case where we have more edges than scores returned
		for j := len(scoreStrs); j < len(batch); j++ {
			edgeScores[i+j] = edgeScore{
				edge:  batch[j],
				score: 0.5,
				index: i + j,
			}
		}
	}

	// Sort by score (descending)
	sort.Slice(edgeScores, func(i, j int) bool {
		return edgeScores[i].score > edgeScores[j].score
	})

	// Filter by minimum score and apply limit
	var resultEdges []*types.Edge
	var resultScores []float64

	for _, es := range edgeScores {
		if es.score >= minScore && len(resultEdges) < limit {
			resultEdges = append(resultEdges, es.edge)
			resultScores = append(resultScores, es.score)
		}
	}

	return resultEdges, resultScores, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
