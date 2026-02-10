//go:build cgo

package candle

import (
	"context"
	"fmt"
	"sync"

	gocandle "github.com/soundprediction/go-candle/pkg/candle"
)

// CandleEmbedderConfig holds configuration for the candle embedder.
type CandleEmbedderConfig struct {
	Model      string `json:"model"`
	CacheDir   string `json:"cache_dir,omitempty"`
	Dimensions int    `json:"dimensions"`
	Normalize  bool   `json:"normalize"`
}

// CandleEmbedderClient implements embedder.Client using go-candle's EmbeddingPipeline.
// Models are auto-downloaded from HuggingFace Hub on first use.
type CandleEmbedderClient struct {
	pipeline *gocandle.EmbeddingPipeline
	config   *CandleEmbedderConfig
	mu       sync.Mutex
}

// NewCandleEmbedderClient creates a new candle-based embedder client.
func NewCandleEmbedderClient(cfg *CandleEmbedderConfig) (*CandleEmbedderClient, error) {
	if cfg.Model == "" {
		return nil, fmt.Errorf("model is required")
	}

	if err := gocandle.Init(); err != nil {
		return nil, fmt.Errorf("failed to init candle: %w", err)
	}

	pipeline, err := gocandle.NewEmbeddingPipeline(gocandle.EmbeddingConfig{
		ModelID:   cfg.Model,
		CacheDir:  cfg.CacheDir,
		Normalize: cfg.Normalize,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create embedding pipeline: %w", err)
	}

	return &CandleEmbedderClient{
		pipeline: pipeline,
		config:   cfg,
	}, nil
}

// Embed generates embeddings for the given texts.
func (c *CandleEmbedderClient) Embed(_ context.Context, texts []string) ([][]float32, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.pipeline == nil {
		return nil, fmt.Errorf("embedder pipeline is closed")
	}

	return c.pipeline.EmbedBatch(texts)
}

// EmbedSingle generates an embedding for a single text.
func (c *CandleEmbedderClient) EmbedSingle(_ context.Context, text string) ([]float32, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.pipeline == nil {
		return nil, fmt.Errorf("embedder pipeline is closed")
	}

	return c.pipeline.Embed(text)
}

// Dimensions returns the number of dimensions in the embeddings.
func (c *CandleEmbedderClient) Dimensions() int {
	return c.config.Dimensions
}

// Close cleans up resources.
func (c *CandleEmbedderClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.pipeline != nil {
		c.pipeline.Close()
		c.pipeline = nil
	}
	return nil
}
