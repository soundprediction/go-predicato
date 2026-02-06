package search

import (
	"context"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/types"
)

// EpisodeSearchOptions holds options for episode search
type EpisodeSearchOptions struct {
	TimeRange *types.TimeRange
	GroupIDs  []string
	Limit     int
}

// CommunitySearchOptions holds options for community search
type CommunitySearchOptions struct {
	GroupIDs  []string
	Limit     int
	MinScore  float64
	MMRLambda float64
}

// EpisodeFulltextSearch performs fulltext search on episodic nodes
func (su *SearchUtilities) EpisodeFulltextSearch(ctx context.Context, query string, options *EpisodeSearchOptions) ([]*types.Node, error) {
	if options == nil {
		options = &EpisodeSearchOptions{
			Limit: RelevantSchemaLimit,
		}
	}

	if options.Limit <= 0 {
		options.Limit = RelevantSchemaLimit
	}

	fulltextQuery := FulltextQuery(query, options.GroupIDs)
	if fulltextQuery == "" {
		return []*types.Node{}, nil
	}

	// Create search options for episodic nodes
	searchOptions := &driver.SearchOptions{
		Limit:       options.Limit,
		UseFullText: true,
		NodeTypes:   []types.NodeType{types.EpisodicNodeType},
		TimeRange:   options.TimeRange,
	}

	// Use the first group ID if available
	var targetGroupID string
	if len(options.GroupIDs) > 0 {
		targetGroupID = options.GroupIDs[0]
	}

	return su.driver.SearchNodes(ctx, fulltextQuery, targetGroupID, searchOptions)
}

// CommunityFulltextSearch performs fulltext search on community nodes
func (su *SearchUtilities) CommunityFulltextSearch(ctx context.Context, query string, options *CommunitySearchOptions) ([]*types.Node, error) {
	if options == nil {
		options = &CommunitySearchOptions{
			Limit:    RelevantSchemaLimit,
			MinScore: DefaultMinScore,
		}
	}

	if options.Limit <= 0 {
		options.Limit = RelevantSchemaLimit
	}

	fulltextQuery := FulltextQuery(query, options.GroupIDs)
	if fulltextQuery == "" {
		return []*types.Node{}, nil
	}

	// Create search filters for community nodes
	searchOptions := &driver.SearchOptions{
		Limit:       options.Limit,
		UseFullText: true,
		NodeTypes:   []types.NodeType{types.CommunityNodeType},
	}

	// Use the first group ID if available
	var targetGroupID string
	if len(options.GroupIDs) > 0 {
		targetGroupID = options.GroupIDs[0]
	}

	return su.driver.SearchNodes(ctx, fulltextQuery, targetGroupID, searchOptions)
}

// CommunitySimilaritySearch performs vector similarity search on community nodes
func (su *SearchUtilities) CommunitySimilaritySearch(ctx context.Context, searchVector []float32, options *CommunitySearchOptions) ([]*types.Node, error) {
	if options == nil {
		options = &CommunitySearchOptions{
			Limit:    RelevantSchemaLimit,
			MinScore: DefaultMinScore,
		}
	}

	if options.Limit <= 0 {
		options.Limit = RelevantSchemaLimit
	}

	if options.MinScore == 0 {
		options.MinScore = DefaultMinScore
	}

	searchOptions := &driver.VectorSearchOptions{
		Limit:     options.Limit,
		MinScore:  options.MinScore,
		NodeTypes: []types.NodeType{types.CommunityNodeType},
	}

	// Use the first group ID if available
	var targetGroupID string
	if len(options.GroupIDs) > 0 {
		targetGroupID = options.GroupIDs[0]
	}

	return su.driver.SearchNodesByVector(ctx, searchVector, targetGroupID, searchOptions)
}

// GetRelevantNodes finds nodes relevant to a given set of query nodes
func (su *SearchUtilities) GetRelevantNodes(ctx context.Context, queryNodes []*types.Node, searchFilter *SearchFilters, minScore float64, limit int) ([][]*types.Node, error) {
	if len(queryNodes) == 0 {
		return [][]*types.Node{}, nil
	}

	if limit <= 0 {
		limit = RelevantSchemaLimit
	}
	if minScore == 0 {
		minScore = DefaultMinScore
	}

	// Get the group ID from the first query node
	groupID := queryNodes[0].GroupID

	var allRelevantNodes [][]*types.Node

	for _, queryNode := range queryNodes {
		// Search for nodes similar to this query node
		var relevantNodes []*types.Node

		// If the node has embeddings, use similarity search
		if queryNode.Metadata != nil {
			if embeddingData, exists := queryNode.Metadata["name_embedding"]; exists {
				if embedding := toFloat32Slice(embeddingData); embedding != nil {
					nodes, err := su.NodeSimilaritySearch(ctx, embedding, searchFilter, []string{groupID}, limit, minScore)
					if err == nil {
						relevantNodes = append(relevantNodes, nodes...)
					}
				}
			}
		}

		// Also use fulltext search on the node name
		if queryNode.Name != "" {
			nodes, err := su.NodeFulltextSearch(ctx, queryNode.Name, searchFilter, []string{groupID}, limit)
			if err == nil {
				relevantNodes = append(relevantNodes, nodes...)
			}
		}

		// Deduplicate and limit results
		nodeMap := make(map[string]*types.Node)
		for _, node := range relevantNodes {
			nodeMap[node.Uuid] = node
		}

		var uniqueNodes []*types.Node
		for _, node := range nodeMap {
			uniqueNodes = append(uniqueNodes, node)
			if len(uniqueNodes) >= limit {
				break
			}
		}

		allRelevantNodes = append(allRelevantNodes, uniqueNodes)
	}

	return allRelevantNodes, nil
}

// GetRelevantEdges finds edges relevant to a given set of query edges
func (su *SearchUtilities) GetRelevantEdges(ctx context.Context, queryEdges []*types.Edge, searchFilter *SearchFilters, minScore float64, limit int) ([][]*types.Edge, error) {
	if len(queryEdges) == 0 {
		return [][]*types.Edge{}, nil
	}

	if limit <= 0 {
		limit = RelevantSchemaLimit
	}
	if minScore == 0 {
		minScore = DefaultMinScore
	}

	var allRelevantEdges [][]*types.Edge

	for _, queryEdge := range queryEdges {
		var relevantEdges []*types.Edge

		// If the edge has embeddings, use similarity search
		if queryEdge.Metadata != nil {
			if embeddingData, exists := queryEdge.Metadata["fact_embedding"]; exists {
				if embedding := toFloat32Slice(embeddingData); embedding != nil {
					edges, err := su.EdgeSimilaritySearch(ctx, embedding, "", "", searchFilter, []string{queryEdge.GroupID}, limit, minScore)
					if err == nil {
						relevantEdges = append(relevantEdges, edges...)
					}
				}
			}
		}

		// Also use fulltext search on the edge content
		searchText := queryEdge.Name
		if queryEdge.Summary != "" {
			searchText = queryEdge.Summary
		}

		if searchText != "" {
			edges, err := su.EdgeFulltextSearch(ctx, searchText, searchFilter, []string{queryEdge.GroupID}, limit)
			if err == nil {
				relevantEdges = append(relevantEdges, edges...)
			}
		}

		// Deduplicate and limit results
		edgeMap := make(map[string]*types.Edge)
		for _, edge := range relevantEdges {
			edgeMap[edge.Uuid] = edge
		}

		var uniqueEdges []*types.Edge
		for _, edge := range edgeMap {
			uniqueEdges = append(uniqueEdges, edge)
			if len(uniqueEdges) >= limit {
				break
			}
		}

		allRelevantEdges = append(allRelevantEdges, uniqueEdges)
	}

	return allRelevantEdges, nil
}

// Entity-specific search functions

// EntitySearchOptions holds options for entity-specific searches
type EntitySearchOptions struct {
	EntityTypes []string
	GroupIDs    []string
	Limit       int
	MinScore    float64
}

// SearchEntitiesByType performs search filtered by specific entity types
func (su *SearchUtilities) SearchEntitiesByType(ctx context.Context, query string, entityTypes []string, options *EntitySearchOptions) ([]*types.Node, error) {
	if options == nil {
		options = &EntitySearchOptions{
			Limit:    RelevantSchemaLimit,
			MinScore: DefaultMinScore,
		}
	}

	if len(entityTypes) > 0 {
		options.EntityTypes = entityTypes
	}

	// Create search filter with entity types
	searchFilter := &SearchFilters{
		EntityTypes: options.EntityTypes,
		NodeTypes:   []types.NodeType{types.EntityNodeType}, // Focus on entity nodes
	}

	return su.NodeFulltextSearch(ctx, query, searchFilter, options.GroupIDs, options.Limit)
}

// SearchEntitiesByEmbedding performs vector search filtered by entity types
func (su *SearchUtilities) SearchEntitiesByEmbedding(ctx context.Context, searchVector []float32, entityTypes []string, options *EntitySearchOptions) ([]*types.Node, error) {
	if options == nil {
		options = &EntitySearchOptions{
			Limit:    RelevantSchemaLimit,
			MinScore: DefaultMinScore,
		}
	}

	if len(entityTypes) > 0 {
		options.EntityTypes = entityTypes
	}

	// Create search filter with entity types
	searchFilter := &SearchFilters{
		EntityTypes: options.EntityTypes,
		NodeTypes:   []types.NodeType{types.EntityNodeType}, // Focus on entity nodes
	}

	return su.NodeSimilaritySearch(ctx, searchVector, searchFilter, options.GroupIDs, options.Limit, options.MinScore)
}

// Multi-modal search functions

// MultiModalSearchResult represents results from multi-modal search
type MultiModalSearchResult struct {
	Nodes        []*types.Node
	Edges        []*types.Edge
	Episodes     []*types.Node
	Communities  []*types.Node
	NodeScores   []float64
	EdgeScores   []float64
	TotalResults int
}

// MultiModalSearch performs search across all node and edge types
func (su *SearchUtilities) MultiModalSearch(ctx context.Context, query string, searchVector []float32, groupIDs []string, limit int) (*MultiModalSearchResult, error) {
	if limit <= 0 {
		limit = RelevantSchemaLimit
	}

	result := &MultiModalSearchResult{
		Nodes:       []*types.Node{},
		Edges:       []*types.Edge{},
		Episodes:    []*types.Node{},
		Communities: []*types.Node{},
		NodeScores:  []float64{},
		EdgeScores:  []float64{},
	}

	// Search entity nodes
	if query != "" {
		nodes, err := su.NodeFulltextSearch(ctx, query, &SearchFilters{NodeTypes: []types.NodeType{types.EntityNodeType}}, groupIDs, limit)
		if err == nil {
			result.Nodes = append(result.Nodes, nodes...)
		}
	}

	if len(searchVector) > 0 {
		nodes, err := su.NodeSimilaritySearch(ctx, searchVector, &SearchFilters{NodeTypes: []types.NodeType{types.EntityNodeType}}, groupIDs, limit, DefaultMinScore)
		if err == nil {
			result.Nodes = append(result.Nodes, nodes...)
		}
	}

	// Search edges
	if query != "" {
		edges, err := su.EdgeFulltextSearch(ctx, query, &SearchFilters{}, groupIDs, limit)
		if err == nil {
			result.Edges = append(result.Edges, edges...)
		}
	}

	if len(searchVector) > 0 {
		edges, err := su.EdgeSimilaritySearch(ctx, searchVector, "", "", &SearchFilters{}, groupIDs, limit, DefaultMinScore)
		if err == nil {
			result.Edges = append(result.Edges, edges...)
		}
	}

	// Search episodes
	if query != "" {
		episodes, err := su.EpisodeFulltextSearch(ctx, query, &EpisodeSearchOptions{Limit: limit, GroupIDs: groupIDs})
		if err == nil {
			result.Episodes = append(result.Episodes, episodes...)
		}
	}

	// Search communities
	if query != "" {
		communities, err := su.CommunityFulltextSearch(ctx, query, &CommunitySearchOptions{Limit: limit, GroupIDs: groupIDs})
		if err == nil {
			result.Communities = append(result.Communities, communities...)
		}
	}

	if len(searchVector) > 0 {
		communities, err := su.CommunitySimilaritySearch(ctx, searchVector, &CommunitySearchOptions{Limit: limit, GroupIDs: groupIDs})
		if err == nil {
			result.Communities = append(result.Communities, communities...)
		}
	}

	// Deduplicate results
	result.Nodes = su.deduplicateNodes(result.Nodes)
	result.Edges = su.deduplicateEdges(result.Edges)
	result.Episodes = su.deduplicateNodes(result.Episodes)
	result.Communities = su.deduplicateNodes(result.Communities)

	// Calculate total
	result.TotalResults = len(result.Nodes) + len(result.Edges) + len(result.Episodes) + len(result.Communities)

	// Create default scores
	result.NodeScores = make([]float64, len(result.Nodes))
	result.EdgeScores = make([]float64, len(result.Edges))
	for i := range result.NodeScores {
		result.NodeScores[i] = 1.0
	}
	for i := range result.EdgeScores {
		result.EdgeScores[i] = 1.0
	}

	return result, nil
}

// Helper functions

// deduplicateNodes removes duplicate nodes based on ID
func (su *SearchUtilities) deduplicateNodes(nodes []*types.Node) []*types.Node {
	nodeMap := make(map[string]*types.Node)
	for _, node := range nodes {
		nodeMap[node.Uuid] = node
	}

	var uniqueNodes []*types.Node
	for _, node := range nodeMap {
		uniqueNodes = append(uniqueNodes, node)
	}

	return uniqueNodes
}

// deduplicateEdges removes duplicate edges based on ID
func (su *SearchUtilities) deduplicateEdges(edges []*types.Edge) []*types.Edge {
	edgeMap := make(map[string]*types.Edge)
	for _, edge := range edges {
		edgeMap[edge.Uuid] = edge
	}

	var uniqueEdges []*types.Edge
	for _, edge := range edgeMap {
		uniqueEdges = append(uniqueEdges, edge)
	}

	return uniqueEdges
}
