package crossencoder

// This file implements a generic Jina-compatible reranking API client.
// It works with any service that implements the Jina reranking API specification,
// allowing flexibility to use different providers:
//
// Supported Services:
// - Jina AI reranking service (https://api.jina.ai/v1/rerank)
// - vLLM with cross-encoder models (http://localhost:8000/v1/rerank)
// - LocalAI with reranking support
// - Hugging Face Text Generation Inference with reranking
// - Any other service implementing the Jina reranking API
//
// The API specification expects:
// - POST /rerank endpoint
// - JSON request with: model, query, documents, top_k (optional)
// - JSON response with: results array containing index, document, relevance_score
//
// This approach avoids vendor lock-in and provides maximum flexibility
// for choosing reranking services based on your requirements.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"time"
)

// RerankerClient implements cross-encoder functionality using Jina-compatible reranking APIs
// This works with any service implementing the Jina reranking API specification:
// - Jina AI reranking service
// - vLLM with cross-encoder models
// - LocalAI with reranking support
// - Any other Jina-compatible reranking service
type RerankerClient struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
	config     Config
}

// RerankRequest represents the request structure for Jina-compatible rerank APIs
type RerankRequest struct {
	TopK      *int     `json:"top_k,omitempty"`
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
}

// RerankResponse represents the response structure from Jina-compatible rerank APIs
type RerankResponse struct {
	Usage   *Usage         `json:"usage,omitempty"`
	Model   string         `json:"model"`
	Results []RankedResult `json:"results"`
}

// RankedResult represents a single ranking result
type RankedResult struct {
	Document       string  `json:"document"`
	Index          int     `json:"index"`
	RelevanceScore float64 `json:"relevance_score"`
}

// Usage represents token usage information
type Usage struct {
	TotalTokens  int `json:"total_tokens"`
	PromptTokens int `json:"prompt_tokens"`
}

// RerankerConfig holds configuration for Jina-compatible reranking services
type RerankerConfig struct {
	TopK    *int   `json:"top_k,omitempty"`
	BaseURL string `json:"base_url,omitempty"`
	APIKey  string `json:"api_key,omitempty"`
	Config
}

// NewRerankerClient creates a new client for any Jina-compatible reranking service
func NewRerankerClient(config RerankerConfig) *RerankerClient {
	// Default to a generic model name if none provided
	if config.Model == "" {
		config.Model = "reranker"
	}
	// No default BaseURL - must be provided for the specific service
	if config.BaseURL == "" {
		config.BaseURL = "http://localhost:8000/v1" // Common default for local services
	}

	return &RerankerClient{
		baseURL: config.BaseURL,
		apiKey:  config.APIKey,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		config: config.Config,
	}
}

// NewJinaRerankerClient creates a new client for Jina AI's reranking service (convenience function)
func NewJinaRerankerClient(apiKey string, model string) *RerankerClient {
	config := RerankerConfig{
		Config: Config{
			Model: model,
		},
		BaseURL: "https://api.jina.ai/v1",
		APIKey:  apiKey,
	}
	if model == "" {
		config.Model = "jina-reranker-v1-base-en"
	}
	return NewRerankerClient(config)
}

// NewVLLMRerankerClient creates a new client for vLLM's reranking service (convenience function)
func NewVLLMRerankerClient(baseURL string, model string) *RerankerClient {
	config := RerankerConfig{
		Config: Config{
			Model: model,
		},
		BaseURL: baseURL,
		APIKey:  "", // vLLM typically doesn't require API keys
	}
	if baseURL == "" {
		config.BaseURL = "http://localhost:8000/v1"
	}
	if model == "" {
		config.Model = "BAAI/bge-reranker-large"
	}
	return NewRerankerClient(config)
}

// NewLocalAIRerankerClient creates a new client for LocalAI's reranking service (convenience function)
func NewLocalAIRerankerClient(baseURL string, model string, apiKey string) *RerankerClient {
	config := RerankerConfig{
		Config: Config{
			Model: model,
		},
		BaseURL: baseURL,
		APIKey:  apiKey, // LocalAI may require API keys depending on configuration
	}
	if baseURL == "" {
		config.BaseURL = "http://localhost:8080/v1"
	}
	if model == "" {
		config.Model = "reranker"
	}
	return NewRerankerClient(config)
}

// Rank ranks the given passages based on their relevance to the query using a Jina-compatible API
func (c *RerankerClient) Rank(ctx context.Context, query string, passages []string) ([]RankedPassage, error) {
	if len(passages) == 0 {
		return []RankedPassage{}, nil
	}

	// Prepare the request
	request := RerankRequest{
		Model:     c.config.Model,
		Query:     query,
		Documents: passages,
	}

	// Marshal the request
	requestBytes, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/rerank", bytes.NewBuffer(requestBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	// Make the request
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	// Read the response
	responseBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	// Check for HTTP errors
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(responseBytes))
	}

	// Parse the response
	var rerankResponse RerankResponse
	if err := json.Unmarshal(responseBytes, &rerankResponse); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// Convert to RankedPassage format
	results := make([]RankedPassage, len(rerankResponse.Results))
	for i, result := range rerankResponse.Results {
		results[i] = RankedPassage{
			Passage: result.Document,
			Score:   result.RelevanceScore,
		}
	}

	// Sort by score (descending) - Most services return them sorted, but ensure consistency
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results, nil
}

// Close cleans up any resources used by the client
func (c *RerankerClient) Close() error {
	// HTTP client doesn't need explicit cleanup
	return nil
}
