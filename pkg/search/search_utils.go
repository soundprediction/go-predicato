package search

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/types"
)

// Constants for search operations
const (
	RelevantSchemaLimit = 10
	DefaultMinScore     = 0.6
	DefaultMMRLambda    = 0.5
	MaxSearchDepth      = 3
	MaxQueryLength      = 128
	DefaultRankConstant = 60
)

// SearchUtilities provides utility functions for graph search operations
type SearchUtilities struct {
	driver driver.GraphDriver
}

// NewSearchUtilities creates a new SearchUtilities instance
func NewSearchUtilities(driver driver.GraphDriver) *SearchUtilities {
	return &SearchUtilities{
		driver: driver,
	}
}

// CalculateCosineSimilarity calculates cosine similarity between two vectors
func CalculateCosineSimilarity(vector1, vector2 []float32) float64 {
	if len(vector1) != len(vector2) {
		return 0.0
	}

	var dotProduct float64
	var norm1, norm2 float64

	for i := range vector1 {
		dotProduct += float64(vector1[i]) * float64(vector2[i])
		norm1 += float64(vector1[i]) * float64(vector1[i])
		norm2 += float64(vector2[i]) * float64(vector2[i])
	}

	norm1 = math.Sqrt(norm1)
	norm2 = math.Sqrt(norm2)

	if norm1 == 0 || norm2 == 0 {
		return 0.0 // Handle zero vectors
	}

	return dotProduct / (norm1 * norm2)
}

// FulltextQuery constructs a fulltext search query with group ID filtering
func FulltextQuery(query string, groupIDs []string) string {
	// Handle simple cases first
	if strings.TrimSpace(query) == "" {
		return ""
	}

	// For simplicity, we'll use the query as-is for now
	// In a full implementation, this would handle Lucene syntax sanitization
	// and provider-specific query formatting
	if len(strings.Fields(query)) > MaxQueryLength {
		return ""
	}

	// For Neo4j-like fulltext search syntax
	sanitizedQuery := sanitizeQuery(query)

	if len(groupIDs) > 0 {
		groupFilter := ""
		for i, groupID := range groupIDs {
			if i > 0 {
				groupFilter += " OR "
			}
			groupFilter += fmt.Sprintf(`group_id:"%s"`, groupID)
		}
		return fmt.Sprintf("(%s) AND (%s)", groupFilter, sanitizedQuery)
	}

	return sanitizedQuery
}

// sanitizeQuery performs basic Lucene query sanitization
func sanitizeQuery(query string) string {
	// Basic sanitization - in a full implementation this would be more comprehensive
	replacer := strings.NewReplacer(
		"+", "\\+",
		"-", "\\-",
		"&&", "\\&&",
		"||", "\\||",
		"!", "\\!",
		"(", "\\(",
		")", "\\)",
		"{", "\\{",
		"}", "\\}",
		"[", "\\[",
		"]", "\\]",
		"^", "\\^",
		"~", "\\~",
		"*", "\\*",
		"?", "\\?",
		":", "\\:",
		"\"", "\\\"",
	)
	return replacer.Replace(query)
}

// NodeFulltextSearch performs BM25/fulltext search on nodes
func (su *SearchUtilities) NodeFulltextSearch(ctx context.Context, query string, searchFilter *SearchFilters, groupIDs []string, limit int) ([]*types.Node, error) {
	if limit <= 0 {
		limit = RelevantSchemaLimit
	}

	fulltextQuery := FulltextQuery(query, groupIDs)
	if fulltextQuery == "" {
		return []*types.Node{}, nil
	}

	// Build search options
	options := &driver.SearchOptions{
		Limit:       limit,
		UseFullText: true,
	}

	if searchFilter != nil {
		options.NodeTypes = searchFilter.NodeTypes
		options.TimeRange = searchFilter.TimeRange
	}

	// Use the first group ID if available
	var targetGroupID string
	if len(groupIDs) > 0 {
		targetGroupID = groupIDs[0]
	}

	return su.driver.SearchNodes(ctx, fulltextQuery, targetGroupID, options)
}

// NodeSimilaritySearch performs vector similarity search on nodes
func (su *SearchUtilities) NodeSimilaritySearch(ctx context.Context, searchVector []float32, searchFilter *SearchFilters, groupIDs []string, limit int, minScore float64) ([]*types.Node, error) {
	if limit <= 0 {
		limit = RelevantSchemaLimit
	}
	if minScore == 0 {
		minScore = DefaultMinScore
	}

	options := &driver.VectorSearchOptions{
		Limit:    limit,
		MinScore: minScore,
	}

	if searchFilter != nil {
		options.NodeTypes = searchFilter.NodeTypes
		options.TimeRange = searchFilter.TimeRange
	}

	// Use the first group ID if available
	var targetGroupID string
	if len(groupIDs) > 0 {
		targetGroupID = groupIDs[0]
	}

	return su.driver.SearchNodesByVector(ctx, searchVector, targetGroupID, options)
}

// EdgeFulltextSearch performs BM25/fulltext search on edges
func (su *SearchUtilities) EdgeFulltextSearch(ctx context.Context, query string, searchFilter *SearchFilters, groupIDs []string, limit int) ([]*types.Edge, error) {
	if limit <= 0 {
		limit = RelevantSchemaLimit
	}

	fulltextQuery := FulltextQuery(query, groupIDs)
	if fulltextQuery == "" {
		return []*types.Edge{}, nil
	}

	options := &driver.SearchOptions{
		Limit:       limit,
		UseFullText: true,
	}

	if searchFilter != nil {
		options.EdgeTypes = searchFilter.EdgeTypes
		options.TimeRange = searchFilter.TimeRange
	}

	// Use the first group ID if available
	var targetGroupID string
	if len(groupIDs) > 0 {
		targetGroupID = groupIDs[0]
	}

	return su.driver.SearchEdges(ctx, fulltextQuery, targetGroupID, options)
}

// EdgeSimilaritySearch performs vector similarity search on edges
func (su *SearchUtilities) EdgeSimilaritySearch(ctx context.Context, searchVector []float32, sourceNodeUUID, targetNodeUUID string, searchFilter *SearchFilters, groupIDs []string, limit int, minScore float64) ([]*types.Edge, error) {
	if limit <= 0 {
		limit = RelevantSchemaLimit
	}
	if minScore == 0 {
		minScore = DefaultMinScore
	}

	options := &driver.VectorSearchOptions{
		Limit:    limit,
		MinScore: minScore,
	}

	if searchFilter != nil {
		options.EdgeTypes = searchFilter.EdgeTypes
		options.TimeRange = searchFilter.TimeRange
	}

	// Use the first group ID if available
	var targetGroupID string
	if len(groupIDs) > 0 {
		targetGroupID = groupIDs[0]
	}

	return su.driver.SearchEdgesByVector(ctx, searchVector, targetGroupID, options)
}

// HybridNodeSearch performs hybrid search combining fulltext and vector similarity
func (su *SearchUtilities) HybridNodeSearch(ctx context.Context, queries []string, embeddings [][]float32, searchFilter *SearchFilters, groupIDs []string, limit int) ([]*types.Node, error) {
	if limit <= 0 {
		limit = RelevantSchemaLimit
	}

	start := time.Now()
	var allResults [][]*types.Node

	// Perform fulltext searches
	for _, query := range queries {
		nodes, err := su.NodeFulltextSearch(ctx, query, searchFilter, groupIDs, limit*2)
		if err != nil {
			return nil, fmt.Errorf("fulltext search failed: %w", err)
		}
		allResults = append(allResults, nodes)
	}

	// Perform vector similarity searches
	for _, embedding := range embeddings {
		nodes, err := su.NodeSimilaritySearch(ctx, embedding, searchFilter, groupIDs, limit*2, DefaultMinScore)
		if err != nil {
			return nil, fmt.Errorf("similarity search failed: %w", err)
		}
		allResults = append(allResults, nodes)
	}

	// Create node map for deduplication
	nodeUUIDMap := make(map[string]*types.Node)
	var resultUUIDs [][]string

	for _, result := range allResults {
		var uuids []string
		for _, node := range result {
			nodeUUIDMap[node.Uuid] = node
			uuids = append(uuids, node.Uuid)
		}
		resultUUIDs = append(resultUUIDs, uuids)
	}

	// Apply RRF reranking
	rankedUUIDs, _ := RRF(resultUUIDs, DefaultRankConstant, 0)

	// Build final result
	var relevantNodes []*types.Node
	for _, uuid := range rankedUUIDs {
		if node, exists := nodeUUIDMap[uuid]; exists {
			relevantNodes = append(relevantNodes, node)
			if len(relevantNodes) >= limit {
				break
			}
		}
	}

	duration := time.Since(start)
	// Log debug info if needed
	_ = duration

	return relevantNodes, nil
}

// GetEpisodesByMentions retrieves episodes that mention the given nodes/edges
func (su *SearchUtilities) GetEpisodesByMentions(ctx context.Context, nodes []*types.Node, edges []*types.Edge, limit int) ([]*types.Node, error) {
	// Note: limit parameter is for future implementation when this queries the database
	_ = limit

	var episodeUUIDs []string

	// Collect episode UUIDs from edges
	for _, edge := range edges {
		if edge.Metadata != nil {
			if episodes, exists := edge.Metadata["episodes"]; exists {
				if episodeList, ok := episodes.([]string); ok {
					episodeUUIDs = append(episodeUUIDs, episodeList...)
				}
			}
		}
	}

	// For now, return empty slice as this would require direct database queries
	// In a full implementation, this would query episodic nodes by UUIDs
	// and limit the results based on the 'limit' parameter and 'episodeUUIDs'.
	_ = episodeUUIDs // Acknowledge the variable for future implementation
	return []*types.Node{}, nil
}

// GetMentionedNodes retrieves nodes mentioned by the given episodes
func (su *SearchUtilities) GetMentionedNodes(ctx context.Context, episodes []*types.Node) ([]*types.Node, error) {
	// This would require a database query to find nodes mentioned by episodes
	// For now, return empty slice
	return []*types.Node{}, nil
}

// GetCommunitiesByNodes retrieves communities that contain the given nodes as members
func (su *SearchUtilities) GetCommunitiesByNodes(ctx context.Context, nodes []*types.Node) ([]*types.Node, error) {
	// This would require a database query to find communities containing these nodes
	// For now, return empty slice
	return []*types.Node{}, nil
}

// Utility functions for string/numeric conversions
func toStringSlice(v interface{}) []string {
	if v == nil {
		return nil
	}
	switch val := v.(type) {
	case []string:
		return val
	case []interface{}:
		result := make([]string, len(val))
		for i, item := range val {
			result[i] = fmt.Sprintf("%v", item)
		}
		return result
	}
	return nil
}

func toFloat32Slice(v interface{}) []float32 {
	if v == nil {
		return nil
	}
	switch val := v.(type) {
	case []float32:
		return val
	case []float64:
		result := make([]float32, len(val))
		for i, f := range val {
			result[i] = float32(f)
		}
		return result
	case []interface{}:
		result := make([]float32, 0, len(val))
		for _, item := range val {
			switch v := item.(type) {
			case float64:
				result = append(result, float32(v))
			case float32:
				result = append(result, v)
			case string:
				if f, err := strconv.ParseFloat(v, 64); err == nil {
					result = append(result, float32(f))
				}
			}
		}
		return result
	}
	return nil
}

// MMRRerank performs Maximal Marginal Relevance reranking to reduce redundancy
func MMRRerank(entities []*types.Node, queryEmbedding []float32, lambdaParam float64, topK int) []*types.Node {
	if len(entities) == 0 || len(queryEmbedding) == 0 {
		return entities
	}

	if lambdaParam <= 0 {
		lambdaParam = DefaultMMRLambda
	}
	if topK <= 0 {
		topK = RelevantSchemaLimit
	}

	selected := make([]*types.Node, 0, topK)
	remaining := make([]*types.Node, len(entities))
	copy(remaining, entities)

	for len(selected) < topK && len(remaining) > 0 {
		var bestIndex int
		var bestScore float64 = -1

		for i, entity := range remaining {
			// Calculate relevance score (similarity to query)
			var relevanceScore float64
			if len(entity.Embedding) > 0 {
				relevanceScore = CalculateCosineSimilarity(queryEmbedding, entity.Embedding)
			}

			// Calculate diversity score (maximum similarity to already selected items)
			var maxSimilarity float64
			for _, selectedEntity := range selected {
				if len(selectedEntity.Embedding) > 0 && len(entity.Embedding) > 0 {
					similarity := CalculateCosineSimilarity(entity.Embedding, selectedEntity.Embedding)
					if similarity > maxSimilarity {
						maxSimilarity = similarity
					}
				}
			}

			// MMR score: λ * relevance - (1-λ) * max_similarity
			mmrScore := lambdaParam*relevanceScore - (1-lambdaParam)*maxSimilarity

			if mmrScore > bestScore {
				bestScore = mmrScore
				bestIndex = i
			}
		}

		// Add best item to selected and remove from remaining
		selected = append(selected, remaining[bestIndex])
		remaining = append(remaining[:bestIndex], remaining[bestIndex+1:]...)
	}

	return selected
}
