package predicato

import (
	"context"
	"fmt"
	"time"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/factstore"
	"github.com/soundprediction/predicato/pkg/search"
	"github.com/soundprediction/predicato/pkg/types"
)

// Search performs hybrid search across the knowledge graph.
func (c *Client) Search(ctx context.Context, query string, config *types.SearchConfig) (*types.SearchResults, error) {
	if config == nil {
		config = c.config.SearchConfig
	}

	// Convert types.SearchConfig to search.SearchConfig
	searchConfig := &search.SearchConfig{
		Limit:    config.Limit,
		MinScore: config.MinScore,
	}

	// Convert node config if present
	if config.NodeConfig != nil {
		searchConfig.NodeConfig = &search.NodeSearchConfig{
			SearchMethods: convertSearchMethods(config.NodeConfig.SearchMethods),
			Reranker:      convertReranker(config.NodeConfig.Reranker),
			MinScore:      config.NodeConfig.MinScore,
			MMRLambda:     0.5, // Default MMR lambda
			MaxDepth:      config.CenterNodeDistance,
		}
	} else {
		// Default: use all search methods for comprehensive results
		searchConfig.NodeConfig = &search.NodeSearchConfig{
			SearchMethods: []search.SearchMethod{search.CosineSimilarity, search.BM25, search.BreadthFirstSearch},
			Reranker:      search.RRFRerankType,
			MinScore:      0.0,
			MMRLambda:     0.5,
			MaxDepth:      config.CenterNodeDistance,
		}
	}

	// Convert edge config if present
	if config.EdgeConfig != nil {
		searchConfig.EdgeConfig = &search.EdgeSearchConfig{
			SearchMethods: convertSearchMethods(config.EdgeConfig.SearchMethods),
			Reranker:      convertReranker(config.EdgeConfig.Reranker),
			MinScore:      config.EdgeConfig.MinScore,
			MMRLambda:     0.5, // Default MMR lambda
			MaxDepth:      config.CenterNodeDistance,
		}
	} else {
		searchConfig.EdgeConfig = &search.EdgeSearchConfig{
			SearchMethods: []search.SearchMethod{search.CosineSimilarity, search.BM25, search.BreadthFirstSearch},
			Reranker:      search.RRFRerankType,
			MinScore:      0.0,
			MMRLambda:     0.5,
			MaxDepth:      config.CenterNodeDistance,
		}
	}

	// Create search filters
	filters := &search.SearchFilters{}

	// Perform the search
	result, err := c.searcher.Search(ctx, query, searchConfig, filters, c.config.GroupID)
	if err != nil {
		return nil, err
	}

	// Convert back to types.SearchResults
	searchResults := &types.SearchResults{
		Nodes: result.Nodes,
		Edges: result.Edges,
		Query: result.Query,
		Total: result.Total,
	}

	return searchResults, nil
}

// GetNode retrieves a node by ID.
func (c *Client) GetNode(ctx context.Context, nodeID string) (*types.Node, error) {
	return c.driver.GetNode(ctx, nodeID, c.config.GroupID)
}

// GetEdge retrieves an edge by ID.
func (c *Client) GetEdge(ctx context.Context, edgeID string) (*types.Edge, error) {
	return c.driver.GetEdge(ctx, edgeID, c.config.GroupID)
}

// GetStats retrieves statistics about the knowledge graph.
func (c *Client) GetStats(ctx context.Context) (*driver.GraphStats, error) {
	return c.driver.GetStats(ctx, c.config.GroupID)
}

// RetrieveEpisodes retrieves episodes from the knowledge graph with temporal filtering.
// This is an exact translation of the Python retrieve_episodes() function from
// predicato_core/utils/maintenance/graph_data_operations.py:122-181
//
// Parameters:
//   - referenceTime: Only episodes with valid_at <= referenceTime will be retrieved
//   - groupIDs: List of group IDs to filter by (can be nil for all groups)
//   - limit: Maximum number of episodes to retrieve
//   - episodeType: Optional episode type filter (nil for all types)
//
// Returns episodes in chronological order (oldest first).
//
// Note: This method delegates to driver-specific implementations to handle
// database-specific temporal type comparisons (e.g., Memgraph's zoned_date_time
// vs Ladybug's TIMESTAMP).
func (c *Client) RetrieveEpisodes(
	ctx context.Context,
	referenceTime time.Time,
	groupIDs []string,
	limit int,
	episodeType *types.EpisodeType,
) ([]*types.Node, error) {
	// Call the driver-specific implementation
	return c.driver.RetrieveEpisodes(ctx, referenceTime, groupIDs, limit, episodeType)
}

// GetEpisodes retrieves recent episodes from the knowledge graph.
// This is a simplified wrapper around RetrieveEpisodes for backward compatibility.
func (c *Client) GetEpisodes(ctx context.Context, groupID string, limit int) ([]*types.Node, error) {
	if groupID == "" {
		groupID = c.config.GroupID
	}

	// Use current time as reference time
	referenceTime := time.Now()

	// Call the full RetrieveEpisodes with temporal filtering
	return c.RetrieveEpisodes(ctx, referenceTime, []string{groupID}, limit, nil)
}

// GetNodesAndEdgesByEpisode retrieves all nodes and edges mentioned in a specific episode.
func (c *Client) GetNodesAndEdgesByEpisode(ctx context.Context, episodeUUID string) ([]*types.Node, []*types.Edge, error) {
	// Get the episode first
	episode, err := c.GetNode(ctx, episodeUUID)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get episode: %w", err)
	}
	if episode.Type != types.EpisodicNodeType {
		return nil, nil, fmt.Errorf("node %s is not an episode", episodeUUID)
	}

	// Find nodes mentioned by the episode
	mentionedNodes, err := types.GetMentionedNodes(ctx, c.driver, []*types.Node{episode})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get mentioned nodes: %w", err)
	}

	// Find edges mentioned by the episode
	wrapper := &driverWrapper{c.driver}
	edges, err := types.GetEntityEdgesByUUIDs(ctx, wrapper, episode.EntityEdges)
	if err != nil {
		return mentionedNodes, nil, fmt.Errorf("failed to get entity edges: %w", err)
	}

	return mentionedNodes, edges, nil
}

// NewDefaultSearchConfig creates a default search configuration.
func NewDefaultSearchConfig() *types.SearchConfig {
	return &types.SearchConfig{
		Limit:              20,
		CenterNodeDistance: 2,
		MinScore:           0.0,
		IncludeEdges:       true,
		Rerank:             false,
	}
}

// convertSearchMethods converts string search methods to search.SearchMethod enum.
func convertSearchMethods(methods []string) []search.SearchMethod {
	converted := make([]search.SearchMethod, len(methods))
	for i, method := range methods {
		switch method {
		case "cosine_similarity":
			converted[i] = search.CosineSimilarity
		case "bm25":
			converted[i] = search.BM25
		case "bfs", "breadth_first_search":
			converted[i] = search.BreadthFirstSearch
		default:
			converted[i] = search.BM25 // Default fallback
		}
	}
	return converted
}

// convertReranker converts string reranker to search.RerankerType enum.
func convertReranker(reranker string) search.RerankerType {
	switch reranker {
	case "rrf":
		return search.RRFRerankType
	case "mmr":
		return search.MMRRerankType
	case "cross_encoder":
		return search.CrossEncoderRerankType
	case "node_distance":
		return search.NodeDistanceRerankType
	case "episode_mentions":
		return search.EpisodeMentionsRerankType
	default:
		return search.RRFRerankType // Default fallback
	}
}

// SearchFacts performs RAG search directly on the factstore without graph queries.
// This is useful for simpler RAG use cases that don't need relationship traversal.
// The query is embedded using the configured embedder, then hybrid search is performed.
func (c *Client) SearchFacts(ctx context.Context, query string, config *types.SearchConfig) (*factstore.FactSearchResults, error) {
	// Check if factstore is configured
	if c.factStore == nil {
		return nil, fmt.Errorf("factstore not configured: set DbConfig in Config")
	}

	// Check if embedder is available
	if c.embedder == nil {
		return nil, fmt.Errorf("embedder not configured: required for SearchFacts")
	}

	// Generate embedding from query using EmbedSingle
	embedding, err := c.embedder.EmbedSingle(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}

	// Convert types.SearchConfig to factstore.FactSearchConfig
	factConfig := &factstore.FactSearchConfig{
		GroupID:  c.config.GroupID,
		Limit:    10,
		MinScore: 0.0,
	}

	if config != nil {
		if config.Limit > 0 {
			factConfig.Limit = config.Limit
		}
		if config.MinScore > 0 {
			factConfig.MinScore = config.MinScore
		}

		// Map search methods from NodeConfig if available
		if config.NodeConfig != nil && len(config.NodeConfig.SearchMethods) > 0 {
			factConfig.SearchMethods = convertToFactstoreSearchMethods(config.NodeConfig.SearchMethods)
		}
	}

	// Perform hybrid search on factstore
	results, err := c.factStore.HybridSearch(ctx, query, embedding, factConfig)
	if err != nil {
		return nil, fmt.Errorf("factstore search failed: %w", err)
	}

	return results, nil
}

// convertToFactstoreSearchMethods converts types.SearchConfig search method strings
// to factstore.SearchMethod values.
func convertToFactstoreSearchMethods(methods []string) []factstore.SearchMethod {
	var factMethods []factstore.SearchMethod
	for _, m := range methods {
		switch m {
		case "cosine_similarity", "vector":
			factMethods = append(factMethods, factstore.VectorSearch)
		case "bm25", "keyword":
			factMethods = append(factMethods, factstore.KeywordSearch)
		}
	}
	return factMethods
}
