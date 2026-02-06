/*
Package crossencoder provides cross-encoder functionality for ranking passages
based on their relevance to a query.

Cross-encoders are used in information retrieval to rerank search results by
computing relevance scores between a query and candidate passages. This package
provides multiple implementations including OpenAI-based, Jina-compatible APIs
(vLLM, LocalAI, etc.), embedding-based, local similarity-based, EmbedEverything-based
local reranking, and mock implementations for testing.

Usage:

	// Using OpenAI reranker
	nlProcessor := nlp.NewOpenAIClient("api-key", nlp.Config{Model: "gpt-4o-mini"})
	reranker := crossencoder.NewOpenAIRerankerClient(nlProcessor, crossencoder.Config{
		MaxConcurrency: 5,
	})

	// Rank passages
	results, err := reranker.Rank(ctx, "search query", []string{
		"passage 1 text",
		"passage 2 text",
	})

	// Using Jina-compatible reranker (works with vLLM, LocalAI, etc.)
	vllmReranker := crossencoder.NewVLLMRerankerClient("http://localhost:8000/v1", "BAAI/bge-reranker-large")
	results, err := vllmReranker.Rank(ctx, query, passages)

	// Using local similarity reranker
	localReranker := crossencoder.NewLocalRerankerClient(crossencoder.Config{})
	results, err := localReranker.Rank(ctx, query, passages)

	// Using EmbedEverything reranker
	eeReranker, err := crossencoder.NewEmbedEverythingClient(&crossencoder.EmbedEverythingConfig{
		Config: &crossencoder.Config{Model: "BAAI/bge-reranker-base"},
	})
	results, err := eeReranker.Rank(ctx, query, passages)

The package supports different reranking strategies:
- OpenAI API-based reranking using boolean classification prompts
- Jina-compatible API reranking (supports vLLM, LocalAI, Jina AI, and others)
- Embedding-based similarity reranking using cosine similarity
- Local text similarity using cosine similarity of term frequency vectors
- EmbedEverything-based local reranking using go-embedeverything library
- Mock implementation for testing with deterministic results
*/package crossencoder

import (
	"fmt"

	"github.com/soundprediction/predicato/pkg/embedder"
	"github.com/soundprediction/predicato/pkg/nlp"
)

// Provider represents the type of cross-encoder provider
type Provider string

const (
	// ProviderOpenAI uses OpenAI API for reranking
	ProviderOpenAI Provider = "openai"

	// ProviderLocal uses local text similarity algorithms
	ProviderLocal Provider = "local"

	// ProviderMock uses mock implementation for testing
	ProviderMock Provider = "mock"

	// ProviderReranker uses Jina-compatible reranking APIs (Jina, vLLM, LocalAI, etc.)
	ProviderReranker Provider = "reranker"

	// ProviderEmbedding uses embedding-based similarity for reranking
	ProviderEmbedding Provider = "embedding"

	// ProviderEmbedEverything uses go-embedeverything for local reranking
	ProviderEmbedEverything Provider = "embedeverything"
)

// ClientConfig holds configuration for creating cross-encoder clients
type ClientConfig struct {
	LLMClient             nlp.Client             `json:"-"`                                // Not serialized, passed at runtime
	EmbedderClient        embedder.Client        `json:"-"`                                // Required for embedding provider
	RerankerConfig        *RerankerConfig        `json:"reranker_config,omitempty"`        // Jina-compatible reranker config
	EmbeddingConfig       *EmbeddingConfig       `json:"embedding_config,omitempty"`       // Embedding-specific config
	EmbedEverythingConfig *EmbedEverythingConfig `json:"embedeverything_config,omitempty"` // EmbedEverything-specific config
	Provider              Provider               `json:"provider"`
	Config                Config                 `json:"config"`
}

// NewClient creates a new cross-encoder client based on the provider type
func NewClient(clientConfig ClientConfig) (Client, error) {
	switch clientConfig.Provider {
	case ProviderOpenAI:
		if clientConfig.LLMClient == nil {
			return nil, fmt.Errorf("LLM client is required for OpenAI provider")
		}
		return NewOpenAIRerankerClient(clientConfig.LLMClient, clientConfig.Config), nil

	case ProviderLocal:
		return NewLocalRerankerClient(clientConfig.Config), nil

	case ProviderMock:
		return NewMockRerankerClient(clientConfig.Config), nil

	case ProviderReranker:
		rerankerConfig := RerankerConfig{Config: clientConfig.Config}
		if clientConfig.RerankerConfig != nil {
			rerankerConfig = *clientConfig.RerankerConfig
		}
		return NewRerankerClient(rerankerConfig), nil

	case ProviderEmbedding:
		if clientConfig.EmbedderClient == nil {
			return nil, fmt.Errorf("embedder client is required for embedding provider")
		}
		embeddingConfig := EmbeddingConfig{Config: clientConfig.Config}
		if clientConfig.EmbeddingConfig != nil {
			embeddingConfig = *clientConfig.EmbeddingConfig
		}
		return NewEmbeddingRerankerClient(clientConfig.EmbedderClient, embeddingConfig), nil

	case ProviderEmbedEverything:
		embedEverythingConfig := &EmbedEverythingConfig{Config: &clientConfig.Config}
		if clientConfig.EmbedEverythingConfig != nil {
			embedEverythingConfig = clientConfig.EmbedEverythingConfig
		}
		return NewEmbedEverythingClient(embedEverythingConfig)

	default:
		return nil, fmt.Errorf("unsupported cross-encoder provider: %s", clientConfig.Provider)
	}
}

// DefaultConfig returns a default configuration for the given provider
func DefaultConfig(provider Provider) Config {
	switch provider {
	case ProviderOpenAI:
		return Config{
			Model:          "gpt-4o-mini",
			BatchSize:      10,
			MaxConcurrency: 5,
		}
	case ProviderLocal:
		return Config{
			BatchSize: 100, // Local processing can handle larger batches
		}
	case ProviderMock:
		return Config{
			BatchSize: 100,
		}
	case ProviderReranker:
		return Config{
			Model:          "reranker", // Generic default, should be overridden
			BatchSize:      100,        // Jina-compatible APIs can handle large batches
			MaxConcurrency: 3,          // Conservative for external APIs
		}
	case ProviderEmbedding:
		return Config{
			BatchSize:      50, // Moderate batch size for embedding computation
			MaxConcurrency: 10, // Can be higher since embeddings are typically faster
		}
	case ProviderEmbedEverything:
		return Config{
			Model:          "BAAI/bge-reranker-base", // Default model for local reranking
			BatchSize:      100,                      // Local processing can handle large batches
			MaxConcurrency: 1,                        // Local processing is typically single-threaded
		}
	default:
		return Config{}
	}
}
