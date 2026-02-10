//go:build cgo

package candle

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/soundprediction/predicato/pkg/crossencoder"
)

// CandleRerankerClient implements crossencoder.Client using embedding-based
// cosine similarity (bi-encoder approach).
type CandleRerankerClient struct {
	embedder *CandleEmbedderClient
}

// NewCandleRerankerClient creates a new candle reranker backed by an embedding client.
func NewCandleRerankerClient(embedder *CandleEmbedderClient) *CandleRerankerClient {
	return &CandleRerankerClient{embedder: embedder}
}

// Rank ranks the given passages based on their relevance to the query.
func (c *CandleRerankerClient) Rank(ctx context.Context, query string, passages []string) ([]crossencoder.RankedPassage, error) {
	if len(passages) == 0 {
		return []crossencoder.RankedPassage{}, nil
	}

	// Embed query + passages in a single batch
	allTexts := make([]string, 0, len(passages)+1)
	allTexts = append(allTexts, query)
	allTexts = append(allTexts, passages...)

	allEmbeddings, err := c.embedder.Embed(ctx, allTexts)
	if err != nil {
		return nil, fmt.Errorf("failed to create embeddings: %w", err)
	}

	if len(allEmbeddings) != len(allTexts) {
		return nil, fmt.Errorf("embedding count mismatch: expected %d, got %d", len(allTexts), len(allEmbeddings))
	}

	queryEmbedding := allEmbeddings[0]
	passageEmbeddings := allEmbeddings[1:]

	results := make([]crossencoder.RankedPassage, len(passages))
	for i, passage := range passages {
		results[i] = crossencoder.RankedPassage{
			Passage: passage,
			Score:   cosineSimilarity(queryEmbedding, passageEmbeddings[i]),
		}
	}

	// Normalize to 0-1 range
	if len(results) > 1 {
		minScore := results[0].Score
		maxScore := results[0].Score
		for _, r := range results[1:] {
			if r.Score < minScore {
				minScore = r.Score
			}
			if r.Score > maxScore {
				maxScore = r.Score
			}
		}
		if maxScore > minScore {
			scoreRange := maxScore - minScore
			for i := range results {
				results[i].Score = (results[i].Score - minScore) / scoreRange
			}
		} else {
			for i := range results {
				results[i].Score = 1.0
			}
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results, nil
}

// Close cleans up resources.
func (c *CandleRerankerClient) Close() error {
	if c.embedder != nil {
		return c.embedder.Close()
	}
	return nil
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}
