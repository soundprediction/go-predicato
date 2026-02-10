//go:build cgo

package crossencoder

import (
	"context"
	"fmt"
	"sort"

	"github.com/soundprediction/go-embedeverything/pkg/embedder"
)

// EmbedEverythingClient implements the Client interface for EmbedEverything reranking.
type EmbedEverythingClient struct {
	reranker *embedder.Reranker
	config   *EmbedEverythingConfig
}

// EmbedEverythingConfig extends Config with EmbedEverything-specific settings.
type EmbedEverythingConfig struct {
	*Config
}

// NewEmbedEverythingClient creates a new EmbedEverything reranker client.
func NewEmbedEverythingClient(config *EmbedEverythingConfig) (*EmbedEverythingClient, error) {
	reranker, err := embedder.NewReranker(config.Model)
	if err != nil {
		return nil, fmt.Errorf("failed to create reranker: %w", err)
	}

	return &EmbedEverythingClient{
		reranker: reranker,
		config:   config,
	}, nil
}

// Rank ranks the given passages based on their relevance to the query.
func (e *EmbedEverythingClient) Rank(ctx context.Context, query string, passages []string) ([]RankedPassage, error) {
	if len(passages) == 0 {
		return []RankedPassage{}, nil
	}

	// go-embedeverything does not support context yet
	results, err := e.reranker.Rerank(query, passages)
	if err != nil {
		return nil, fmt.Errorf("failed to rerank passages: %w", err)
	}

	// Convert to RankedPassage format
	rankedPassages := make([]RankedPassage, len(results))
	for i, result := range results {
		rankedPassages[i] = RankedPassage{
			Passage: result.Text,
			Score:   float64(result.Score),
		}
	}

	// Sort by score (descending) - ensure consistency
	sort.Slice(rankedPassages, func(i, j int) bool {
		return rankedPassages[i].Score > rankedPassages[j].Score
	})

	return rankedPassages, nil
}

// Close cleans up any resources.
func (e *EmbedEverythingClient) Close() error {
	e.reranker.Close()
	return nil
}
