//go:build cgo

package embedder

import (
	"context"
	"fmt"

	"github.com/soundprediction/go-embedeverything/pkg/embedder"
)

// EmbedEverythingClient implements the Client interface for EmbedEverything.
type EmbedEverythingClient struct {
	client *embedder.Embedder
	config *EmbedEverythingConfig
}

// EmbedEverythingConfig extends Config with EmbedEverything-specific settings.
type EmbedEverythingConfig struct {
	*Config
}

// NewEmbedEverythingClient creates a new EmbedEverything client.
func NewEmbedEverythingClient(config *EmbedEverythingConfig) (*EmbedEverythingClient, error) {
	client, err := embedder.NewEmbedder(config.Model)
	if err != nil {
		return nil, fmt.Errorf("failed to create embedder: %w", err)
	}

	return &EmbedEverythingClient{
		client: client,
		config: config,
	}, nil
}

// Embed generates embeddings for the given texts.
func (e *EmbedEverythingClient) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	// go-embedeverything does not support context yet
	embeddings, err := e.client.Embed(texts)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embeddings: %w", err)
	}
	return embeddings, nil
}

// EmbedSingle generates an embedding for a single text.
func (e *EmbedEverythingClient) EmbedSingle(ctx context.Context, text string) ([]float32, error) {
	embeddings, err := e.Embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}
	return embeddings[0], nil
}

// Dimensions returns the number of dimensions in the embeddings.
func (e *EmbedEverythingClient) Dimensions() int {
	return e.config.Dimensions
}

// Close cleans up any resources.
func (e *EmbedEverythingClient) Close() error {
	e.client.Close()
	return nil
}
