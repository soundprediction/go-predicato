package crossencoder

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

// BGERerankerClient implements cross-encoder functionality using BGE reranker models
// This client can work with BGE reranker endpoints like BAAI/bge-reranker-base
type BGERerankerClient struct {
	httpClient *http.Client
	config     BGEConfig
}

// BGEConfig extends Config with BGE-specific settings
type BGEConfig struct {
	Headers map[string]string `json:"headers,omitempty"`
	BaseURL string            `json:"base_url"`
	APIKey  string            `json:"api_key,omitempty"`
	Config
}

// NewBGERerankerClient creates a new BGE-based reranker client
func NewBGERerankerClient(config BGEConfig) *BGERerankerClient {
	if config.Model == "" {
		config.Model = "BAAI/bge-reranker-base"
	}
	if config.BatchSize <= 0 {
		config.BatchSize = 32
	}
	if config.MaxConcurrency <= 0 {
		config.MaxConcurrency = 5
	}

	return &BGERerankerClient{
		config: config,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// bgeRerankRequest represents the request structure for BGE reranking API
type bgeRerankRequest struct {
	Model    string   `json:"model"`
	Query    string   `json:"query"`
	Passages []string `json:"passages"`
	TopK     int      `json:"top_k,omitempty"`
}

// bgeRerankResponse represents the response from BGE reranking API
type bgeRerankResponse struct {
	Error   *bgeError         `json:"error,omitempty"`
	Results []bgeRerankResult `json:"results"`
}

// bgeRerankResult represents a single reranking result
type bgeRerankResult struct {
	Document string  `json:"document"`
	Index    int     `json:"index"`
	Score    float64 `json:"relevance_score"`
}

// bgeError represents an error response
type bgeError struct {
	Message string `json:"message"`
	Code    string `json:"code"`
}

// Rank ranks the given passages based on their relevance to the query
func (c *BGERerankerClient) Rank(ctx context.Context, query string, passages []string) ([]RankedPassage, error) {
	if len(passages) == 0 {
		return []RankedPassage{}, nil
	}

	var allResults []RankedPassage

	// Process passages in batches
	for i := 0; i < len(passages); i += c.config.BatchSize {
		end := i + c.config.BatchSize
		if end > len(passages) {
			end = len(passages)
		}

		batch := passages[i:end]
		batchResults, err := c.rerankBatch(ctx, query, batch, i)
		if err != nil {
			return nil, fmt.Errorf("failed to rerank batch starting at %d: %w", i, err)
		}

		allResults = append(allResults, batchResults...)
	}

	// Sort all results by score descending
	sort.Slice(allResults, func(i, j int) bool {
		return allResults[i].Score > allResults[j].Score
	})

	return allResults, nil
}

// rerankBatch processes a batch of passages
func (c *BGERerankerClient) rerankBatch(ctx context.Context, query string, passages []string, startIndex int) ([]RankedPassage, error) {
	req := bgeRerankRequest{
		Model:    c.config.Model,
		Query:    query,
		Passages: passages,
		TopK:     len(passages), // Return all passages with scores
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.config.BaseURL+"/rerank", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.config.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.config.APIKey)
	}

	// Add any additional headers
	for key, value := range c.config.Headers {
		httpReq.Header.Set(key, value)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var bgeResp bgeRerankResponse
	if err := json.Unmarshal(body, &bgeResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if bgeResp.Error != nil {
		return nil, fmt.Errorf("API error: %s", bgeResp.Error.Message)
	}

	results := make([]RankedPassage, len(bgeResp.Results))
	for i, result := range bgeResp.Results {
		results[i] = RankedPassage{
			Passage: result.Document,
			Score:   result.Score,
		}
	}

	return results, nil
}

// Close cleans up any resources used by the client
func (c *BGERerankerClient) Close() error {
	// Nothing to clean up for HTTP client
	return nil
}
