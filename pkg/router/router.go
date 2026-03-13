//go:build cgo

package router

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"github.com/soundprediction/predicato/pkg/types"
)

// Router routes queries to appropriate predicato clients and merges results.
type Router struct {
	classifier TopicClassifier
	manager    *Manager
	merger     *ResultMerger
	logger     *slog.Logger
	config     RouterConfig
}

// NewRouter creates a new Router with the given components.
func NewRouter(
	manager *Manager,
	classifier TopicClassifier,
	routerConfig RouterConfig,
	logger *slog.Logger,
) *Router {
	if logger == nil {
		logger = slog.Default()
	}

	return &Router{
		manager:    manager,
		classifier: classifier,
		merger:     NewResultMerger(routerConfig.GetMergeStrategyOrDefault()),
		config:     routerConfig,
		logger:     logger,
	}
}

// Route determines which clients should handle the given query.
func (r *Router) Route(ctx context.Context, query string) ([]string, error) {
	matches, err := r.classifier.Classify(ctx, query)
	if err != nil {
		r.logger.Error("Failed to classify query, falling back to default client",
			"error", err,
			"query_preview", truncateString(query, 100),
			"default_client", r.manager.defaultClientName,
		)
		if r.manager.defaultClientName != "" {
			return []string{r.manager.defaultClientName}, nil
		}
		return nil, fmt.Errorf("failed to classify query and no default client: %w", err)
	}

	minConfidence := r.config.GetMinConfidenceOrDefault()
	filtered := make([]TopicMatch, 0, len(matches))
	for _, match := range matches {
		if match.Confidence >= minConfidence {
			filtered = append(filtered, match)
		}
	}

	if len(filtered) == 0 {
		r.logger.Debug("No topic matches above threshold, using fallback/default",
			"query", query,
			"min_confidence", minConfidence,
		)

		if r.manager.fallbackClientName != "" {
			return []string{r.manager.fallbackClientName}, nil
		}
		if r.manager.defaultClientName != "" {
			return []string{r.manager.defaultClientName}, nil
		}
		return nil, fmt.Errorf("no matching clients and no fallback/default configured")
	}

	maxGraphs := r.config.GetMaxGraphsOrDefault()
	if len(filtered) > maxGraphs {
		filtered = filtered[:maxGraphs]
	}

	clientNames := make([]string, len(filtered))
	for i, match := range filtered {
		clientNames[i] = match.ClientName
	}

	r.logger.Debug("Routed query to clients",
		"query", query,
		"clients", clientNames,
		"matches", filtered,
	)

	return clientNames, nil
}

// Search routes the query to appropriate clients and returns merged results.
func (r *Router) Search(ctx context.Context, query string, searchConfig *types.SearchConfig) (*MergedSearchResults, error) {
	clientNames, err := r.Route(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to route query: %w", err)
	}

	return r.SearchWithClients(ctx, query, clientNames, searchConfig)
}

// SearchWithClients searches specific clients (bypassing routing) and merges results.
func (r *Router) SearchWithClients(
	ctx context.Context,
	query string,
	clientNames []string,
	searchConfig *types.SearchConfig,
) (*MergedSearchResults, error) {
	if len(clientNames) == 0 {
		return &MergedSearchResults{
			Nodes:         []*types.Node{},
			Edges:         []*types.Edge{},
			Sources:       make(map[string][]string),
			ClientResults: make(map[string]*types.SearchResults),
		}, nil
	}

	if searchConfig == nil {
		searchConfig = &types.SearchConfig{
			Limit:        10,
			IncludeEdges: true,
			Rerank:       true,
			NodeConfig: &types.NodeSearchConfig{
				SearchMethods: []string{"cosine_similarity", "bm25", "bfs"},
				Reranker:      "mmr",
				MinScore:      0.0,
			},
			EdgeConfig: &types.EdgeSearchConfig{
				SearchMethods: []string{"cosine_similarity", "bm25"},
				Reranker:      "mmr",
				MinScore:      0.0,
			},
		}
	}

	results := make(map[string]*types.SearchResults)
	errors := make(map[string]error)
	var mu sync.Mutex
	var wg sync.WaitGroup

	for _, clientName := range clientNames {
		wg.Add(1)
		go func(name string) {
			defer wg.Done()

			client, err := r.manager.GetClient(name)
			if err != nil {
				mu.Lock()
				errors[name] = err
				mu.Unlock()
				r.logger.Error("Failed to get client", "name", name, "error", err)
				return
			}

			searchResult, err := client.Search(ctx, query, searchConfig)
			mu.Lock()
			if err != nil {
				errors[name] = err
				r.logger.Error("Search failed", "client", name, "error", err)
			} else {
				results[name] = searchResult
				r.logger.Debug("Search completed",
					"client", name,
					"nodes", len(searchResult.Nodes),
					"edges", len(searchResult.Edges),
				)
			}
			mu.Unlock()
		}(clientName)
	}

	wg.Wait()

	limit := 10
	if searchConfig.Limit > 0 {
		limit = searchConfig.Limit
	}

	var preferredPredicates []string
	if searchConfig != nil && len(searchConfig.PreferredPredicates) > 0 {
		preferredPredicates = searchConfig.PreferredPredicates
	} else {
		// Auto-detect predicates from query intent
		preferredPredicates = InferPredicatesFromQuery(query)
	}

	merged := r.merger.Merge(results, limit, preferredPredicates)
	merged.Errors = errors

	r.logger.Info("Search completed",
		"query", query,
		"clients_queried", len(clientNames),
		"clients_succeeded", len(results),
		"clients_failed", len(errors),
		"total_nodes", len(merged.Nodes),
	)

	return merged, nil
}

// SearchDefault searches only the default client.
func (r *Router) SearchDefault(ctx context.Context, query string, searchConfig *types.SearchConfig) (*types.SearchResults, error) {
	client, err := r.manager.GetDefaultClient()
	if err != nil {
		return nil, fmt.Errorf("failed to get default client: %w", err)
	}

	return client.Search(ctx, query, searchConfig)
}

// GetManager returns the underlying Manager.
func (r *Router) GetManager() *Manager {
	return r.manager
}

// GetClientStatuses returns the status of all configured clients.
func (r *Router) GetClientStatuses() []ClientStatus {
	return r.manager.GetClientStatuses()
}

// Close closes the router and all its managed clients.
func (r *Router) Close(ctx context.Context) error {
	return r.manager.Close(ctx)
}

// RouteInfo contains information about how a query was routed.
type RouteInfo struct {
	ClientNames  []string     `json:"client_names"`
	TopicMatches []TopicMatch `json:"topic_matches"`
	UsedFallback bool         `json:"used_fallback"`
	UsedDefault  bool         `json:"used_default"`
}

// GetRouteInfo returns detailed routing information for a query without executing the search.
func (r *Router) GetRouteInfo(ctx context.Context, query string) (*RouteInfo, error) {
	info := &RouteInfo{}

	matches, err := r.classifier.Classify(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to classify query: %w", err)
	}

	info.TopicMatches = matches

	minConfidence := r.config.GetMinConfidenceOrDefault()
	filtered := make([]TopicMatch, 0, len(matches))
	for _, match := range matches {
		if match.Confidence >= minConfidence {
			filtered = append(filtered, match)
		}
	}

	if len(filtered) == 0 {
		if r.manager.fallbackClientName != "" {
			info.ClientNames = []string{r.manager.fallbackClientName}
			info.UsedFallback = true
		} else if r.manager.defaultClientName != "" {
			info.ClientNames = []string{r.manager.defaultClientName}
			info.UsedDefault = true
		}
	} else {
		maxGraphs := r.config.GetMaxGraphsOrDefault()
		if len(filtered) > maxGraphs {
			filtered = filtered[:maxGraphs]
		}

		info.ClientNames = make([]string, len(filtered))
		for i, match := range filtered {
			info.ClientNames[i] = match.ClientName
		}
	}

	return info, nil
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	if maxLen > 3 {
		return s[:maxLen-3] + "..."
	}
	return s[:maxLen]
}
