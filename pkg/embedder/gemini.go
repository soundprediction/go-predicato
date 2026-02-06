package embedder

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// GeminiEmbedder implements the Client interface for Google Gemini embeddings.
type GeminiEmbedder struct {
	config     *GeminiConfig
	httpClient *http.Client
}

// GeminiConfig extends Config with Gemini-specific settings.
type GeminiConfig struct {
	*Config
	APIKey string `json:"api_key"`
}

// NewGeminiEmbedder creates a new Gemini embedder.
func NewGeminiEmbedder(config *GeminiConfig) *GeminiEmbedder {
	if config.BaseURL == "" {
		config.BaseURL = "https://generativelanguage.googleapis.com"
	}
	if config.BatchSize == 0 {
		config.BatchSize = 100
	}

	return &GeminiEmbedder{
		config: config,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// geminiEmbeddingRequest represents the request structure for Gemini embeddings API.
type geminiEmbeddingRequest struct {
	Requests []geminiEmbedRequest `json:"requests"`
}

// geminiEmbedRequest represents a single embedding request.
type geminiEmbedRequest struct {
	Model   string             `json:"model"`
	Content geminiEmbedContent `json:"content"`
}

// geminiEmbedContent represents the content to embed.
type geminiEmbedContent struct {
	Parts []geminiEmbedPart `json:"parts"`
}

// geminiEmbedPart represents a part of the content.
type geminiEmbedPart struct {
	Text string `json:"text"`
}

// geminiEmbeddingResponse represents the response from Gemini embeddings API.
type geminiEmbeddingResponse struct {
	Error      *geminiError      `json:"error,omitempty"`
	Embeddings []geminiEmbedding `json:"embeddings"`
}

// geminiEmbedding represents a single embedding in the response.
type geminiEmbedding struct {
	Values []float32 `json:"values"`
}

// geminiError represents an error response.
type geminiError struct {
	Message string `json:"message"`
	Status  string `json:"status"`
	Code    int    `json:"code"`
}

// Embed generates embeddings for the given texts.
func (g *GeminiEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("no texts provided")
	}

	var allEmbeddings [][]float32

	// Process texts in batches
	for i := 0; i < len(texts); i += g.config.BatchSize {
		end := i + g.config.BatchSize
		if end > len(texts) {
			end = len(texts)
		}

		batch := texts[i:end]
		embeddings, err := g.embedBatch(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("failed to embed batch: %w", err)
		}

		allEmbeddings = append(allEmbeddings, embeddings...)
	}

	return allEmbeddings, nil
}

// embedBatch processes a batch of texts.
func (g *GeminiEmbedder) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	// For batch requests, Gemini expects multiple requests
	requests := make([]geminiEmbedRequest, len(texts))
	for i, text := range texts {
		requests[i] = geminiEmbedRequest{
			Model: g.config.Model,
			Content: geminiEmbedContent{
				Parts: []geminiEmbedPart{{Text: text}},
			},
		}
	}

	req := geminiEmbeddingRequest{
		Requests: requests,
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:batchEmbedContents?key=%s",
		g.config.BaseURL, g.config.Model, g.config.APIKey)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// Add any additional headers
	for key, value := range g.config.Headers {
		httpReq.Header.Set(key, value)
	}

	resp, err := g.httpClient.Do(httpReq)
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

	var geminiResp geminiEmbeddingResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if geminiResp.Error != nil {
		return nil, fmt.Errorf("API error: %s", geminiResp.Error.Message)
	}

	embeddings := make([][]float32, len(geminiResp.Embeddings))
	for i, embedding := range geminiResp.Embeddings {
		embeddings[i] = embedding.Values
	}

	return embeddings, nil
}

// EmbedSingle generates an embedding for a single text.
func (g *GeminiEmbedder) EmbedSingle(ctx context.Context, text string) ([]float32, error) {
	// For single requests, use the single endpoint
	req := geminiEmbedRequest{
		Model: g.config.Model,
		Content: geminiEmbedContent{
			Parts: []geminiEmbedPart{{Text: text}},
		},
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:embedContent?key=%s",
		g.config.BaseURL, g.config.Model, g.config.APIKey)

	httpReq, err := http.NewRequestWithContext(context.Background(), "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// Add any additional headers
	for key, value := range g.config.Headers {
		httpReq.Header.Set(key, value)
	}

	resp, err := g.httpClient.Do(httpReq)
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

	var geminiResp struct {
		Error     *geminiError    `json:"error,omitempty"`
		Embedding geminiEmbedding `json:"embedding"`
	}

	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if geminiResp.Error != nil {
		return nil, fmt.Errorf("API error: %s", geminiResp.Error.Message)
	}

	return geminiResp.Embedding.Values, nil
}

// Dimensions returns the number of dimensions in the embeddings.
func (g *GeminiEmbedder) Dimensions() int {
	return g.config.Dimensions
}

// Close cleans up any resources.
func (g *GeminiEmbedder) Close() error {
	// Nothing to clean up for HTTP client
	return nil
}
