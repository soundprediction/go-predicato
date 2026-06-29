//go:build cgo

package predicato

import (
	"context"
	"fmt"
	"log/slog"
	"os"

	core "github.com/soundprediction/predicato"
	"github.com/soundprediction/predicato/pkg/config"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/factstore"
	predicatoLogger "github.com/soundprediction/predicato/pkg/logger"
	"github.com/soundprediction/predicato/pkg/modeler"
	"github.com/soundprediction/predicato/pkg/router"
	"github.com/soundprediction/predicato/pkg/types"
	"github.com/spf13/cobra"
)

func initializeRouterPredicato(cmd *cobra.Command, cfg *config.Config) (core.Predicato, error) {
	sourceDir, _ := cmd.Flags().GetString("source-dir")
	if sourceDir == "" {
		return nil, fmt.Errorf("source-dir is required for pooled router mode")
	}

	configs, routingTexts, err := router.DiscoverTopicGraphs(sourceDir)
	if err != nil {
		return nil, err
	}
	if len(configs) == 0 {
		return nil, fmt.Errorf("no topic graphs discovered in %s", sourceDir)
	}

	logger := slog.New(predicatoLogger.NewColorHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	embedderClient, err := createEmbedderClient(cfg)
	if err != nil {
		return nil, err
	}

	rerankerClient, err := createRerankerClient(cfg.Reranker, embedderClient)
	if err != nil {
		return nil, err
	}

	routerConfig := router.RouterConfig{
		Strategy:      routerStrategy,
		MergeStrategy: routerMergeStrategy,
		MinConfidence: routerMinConfidence,
		MaxGraphs:     routerMaxGraphs,
	}
	classifier, err := router.NewTopicClassifier(configs, routerConfig, embedderClient, rerankerClient, routingTexts)
	if err != nil {
		return nil, err
	}

	manager := router.NewManagerWithOptions(configs, nil, embedderClient, logger, routerMaxOpenGraphs)
	if rerankerClient != nil {
		manager.SetCrossEncoder(rerankerClient)
		fmt.Printf("Cross-encoder reranker enabled for pooled router: %s (%s)\n", cfg.Reranker.Model, cfg.Reranker.BaseURL)
	}

	fmt.Printf("Discovered %d topic graphs in %s\n", len(configs), sourceDir)
	fmt.Printf("Router strategy: %s, merge strategy: %s, max graphs: %d, min confidence: %.2f, max open graphs: %d\n",
		routerConfig.GetStrategyOrDefault(),
		routerConfig.GetMergeStrategyOrDefault(),
		routerConfig.GetMaxGraphsOrDefault(),
		routerConfig.GetMinConfidenceOrDefault(),
		routerMaxOpenGraphs,
	)

	return &routerPredicato{
		router:  router.NewRouter(manager, classifier, routerConfig, logger),
		manager: manager,
	}, nil
}

type routerPredicato struct {
	router  *router.Router
	manager *router.Manager
}

func (rp *routerPredicato) defaultClient() (*core.Client, error) {
	return rp.manager.GetDefaultClient()
}

func (rp *routerPredicato) Search(ctx context.Context, query string, cfg *types.SearchConfig) (*types.SearchResults, error) {
	merged, err := rp.router.Search(ctx, query, cfg)
	if err != nil {
		return nil, err
	}
	return &types.SearchResults{
		Query: query,
		Nodes: merged.Nodes,
		Edges: merged.Edges,
		Total: merged.Total,
	}, nil
}

func (rp *routerPredicato) Add(ctx context.Context, episodes []types.Episode, options *core.AddEpisodeOptions) (*types.AddBulkEpisodeResults, error) {
	client, err := rp.defaultClient()
	if err != nil {
		return nil, err
	}
	return client.Add(ctx, episodes, options)
}

func (rp *routerPredicato) AddEpisode(ctx context.Context, episode types.Episode, options *core.AddEpisodeOptions) (*types.AddEpisodeResults, error) {
	client, err := rp.defaultClient()
	if err != nil {
		return nil, err
	}
	return client.AddEpisode(ctx, episode, options)
}

func (rp *routerPredicato) GetNode(ctx context.Context, nodeID string) (*types.Node, error) {
	client, err := rp.defaultClient()
	if err != nil {
		return nil, err
	}
	return client.GetNode(ctx, nodeID)
}

func (rp *routerPredicato) GetEdge(ctx context.Context, edgeID string) (*types.Edge, error) {
	client, err := rp.defaultClient()
	if err != nil {
		return nil, err
	}
	return client.GetEdge(ctx, edgeID)
}

func (rp *routerPredicato) GetEpisodes(ctx context.Context, groupID string, limit int) ([]*types.Node, error) {
	client, err := rp.defaultClient()
	if err != nil {
		return nil, err
	}
	return client.GetEpisodes(ctx, groupID, limit)
}

func (rp *routerPredicato) ClearGraph(ctx context.Context, groupID string) error {
	client, err := rp.defaultClient()
	if err != nil {
		return err
	}
	return client.ClearGraph(ctx, groupID)
}

func (rp *routerPredicato) CreateIndices(ctx context.Context) error {
	client, err := rp.defaultClient()
	if err != nil {
		return err
	}
	return client.CreateIndices(ctx)
}

func (rp *routerPredicato) AddTriplet(ctx context.Context, sourceNode *types.Node, edge *types.Edge, targetNode *types.Node) (*types.AddTripletResults, error) {
	client, err := rp.defaultClient()
	if err != nil {
		return nil, err
	}
	return client.AddTriplet(ctx, sourceNode, edge, targetNode)
}

func (rp *routerPredicato) RemoveEpisode(ctx context.Context, episodeUUID string) error {
	client, err := rp.defaultClient()
	if err != nil {
		return err
	}
	return client.RemoveEpisode(ctx, episodeUUID)
}

func (rp *routerPredicato) GetNodesAndEdgesByEpisode(ctx context.Context, episodeUUID string) ([]*types.Node, []*types.Edge, error) {
	client, err := rp.defaultClient()
	if err != nil {
		return nil, nil, err
	}
	return client.GetNodesAndEdgesByEpisode(ctx, episodeUUID)
}

func (rp *routerPredicato) Close(ctx context.Context) error {
	return rp.router.Close(ctx)
}

func (rp *routerPredicato) UpdateCommunities(ctx context.Context, episodeUUID string, groupID string) ([]*types.Node, []*types.Edge, error) {
	client, err := rp.defaultClient()
	if err != nil {
		return nil, nil, err
	}
	return client.UpdateCommunities(ctx, episodeUUID, groupID)
}

func (rp *routerPredicato) GetFactStore() factstore.FactsDB {
	client, err := rp.defaultClient()
	if err != nil {
		return nil
	}
	return client.GetFactStore()
}

func (rp *routerPredicato) SearchFacts(ctx context.Context, query string, cfg *types.SearchConfig) (*factstore.FactSearchResults, error) {
	client, err := rp.defaultClient()
	if err != nil {
		return nil, err
	}
	return client.SearchFacts(ctx, query, cfg)
}

func (rp *routerPredicato) ExtractToFacts(ctx context.Context, episode types.Episode, options *core.AddEpisodeOptions) (*types.ExtractionResults, error) {
	client, err := rp.defaultClient()
	if err != nil {
		return nil, err
	}
	return client.ExtractToFacts(ctx, episode, options)
}

func (rp *routerPredicato) PromoteToGraph(ctx context.Context, sourceID string, options *core.AddEpisodeOptions) (*types.AddEpisodeResults, error) {
	client, err := rp.defaultClient()
	if err != nil {
		return nil, err
	}
	return client.PromoteToGraph(ctx, sourceID, options)
}

func (rp *routerPredicato) ValidateModeler(ctx context.Context, gm modeler.GraphModeler) (*modeler.ModelerValidationResult, error) {
	client, err := rp.defaultClient()
	if err != nil {
		return nil, err
	}
	return client.ValidateModeler(ctx, gm)
}

func (rp *routerPredicato) AnalyzeContentRelevance(ctx context.Context, content string, topics []string) (bool, error) {
	client, err := rp.defaultClient()
	if err != nil {
		return false, err
	}
	return client.AnalyzeContentRelevance(ctx, content, topics)
}

func (rp *routerPredicato) ExtractSourceInfo(ctx context.Context, content string, url string) (string, string, error) {
	client, err := rp.defaultClient()
	if err != nil {
		return "", "", err
	}
	return client.ExtractSourceInfo(ctx, content, url)
}

func (rp *routerPredicato) GetStats(ctx context.Context) (*driver.GraphStats, error) {
	client, err := rp.defaultClient()
	if err != nil {
		return nil, err
	}
	return client.GetStats(ctx)
}
