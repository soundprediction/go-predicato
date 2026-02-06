package embedder

import (
	"context"
)

// Client defines the interface for embedding operations.
type Client interface {
	// Embed generates embeddings for the given texts.
	Embed(ctx context.Context, texts []string) ([][]float32, error)

	// EmbedSingle generates an embedding for a single text.
	EmbedSingle(ctx context.Context, text string) ([]float32, error)

	// Dimensions returns the number of dimensions in the embeddings.
	Dimensions() int

	// Close cleans up any resources.
	Close() error
}

// Config holds configuration for embedding clients.
type Config struct {
	Headers    map[string]string `json:"headers,omitempty"` // Additional headers for requests
	Model      string            `json:"model"`
	BaseURL    string            `json:"base_url,omitempty"` // Custom base URL for OpenAI-compatible services
	BatchSize  int               `json:"batch_size"`
	Dimensions int               `json:"dimensions"`
	MaxRetries int               `json:"max_retries"` // Maximum number of retry attempts
}
