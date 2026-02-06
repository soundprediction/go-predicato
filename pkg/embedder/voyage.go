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

// VoyageEmbedder implements the Client interface for Voyage AI embeddings.
type VoyageEmbedder struct {
	config     *VoyageConfig
	httpClient *http.Client
}

// VoyageConfig extends Config with Voyage-specific settings.
type VoyageConfig struct {
	*Config
	APIKey string `json:"api_key"`
}

// NewVoyageEmbedder creates a new Voyage embedder.
func NewVoyageEmbedder(config *VoyageConfig) *VoyageEmbedder {
	if config.BaseURL == "" {
		config.BaseURL = "https://api.voyageai.com"
	}
	if config.BatchSize == 0 {
		config.BatchSize = 128
	}

	return &VoyageEmbedder{
		config: config,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// voyageEmbeddingRequest represents the request structure for Voyage AI embeddings API.
type voyageEmbeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// voyageEmbeddingResponse represents the response from Voyage AI embeddings API.
type voyageEmbeddingResponse struct {
	Error  *voyageEmbeddingError `json:"error,omitempty"`
	Object string                `json:"object"`
	Model  string                `json:"model"`
	Data   []voyageEmbeddingData `json:"data"`
	Usage  voyageEmbeddingUsage  `json:"usage"`
}

// voyageEmbeddingData represents a single embedding in the response.
type voyageEmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

// voyageEmbeddingUsage represents usage information in the response.
type voyageEmbeddingUsage struct {
	TotalTokens int `json:"total_tokens"`
}

// voyageEmbeddingError represents an error response.
type voyageEmbeddingError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// Embed generates embeddings for the given texts.
func (v *VoyageEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("no texts provided")
	}

	var allEmbeddings [][]float32

	// Process texts in batches
	for i := 0; i < len(texts); i += v.config.BatchSize {
		end := i + v.config.BatchSize
		if end > len(texts) {
			end = len(texts)
		}

		batch := texts[i:end]
		embeddings, err := v.embedBatch(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("failed to embed batch: %w", err)
		}

		allEmbeddings = append(allEmbeddings, embeddings...)
	}

	return allEmbeddings, nil
}

// embedBatch processes a batch of texts.
func (v *VoyageEmbedder) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	req := voyageEmbeddingRequest{
		Input: texts,
		Model: v.config.Model,
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", v.config.BaseURL+"/v1/embeddings", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+v.config.APIKey)

	// Add any additional headers
	for key, value := range v.config.Headers {
		httpReq.Header.Set(key, value)
	}

	resp, err := v.httpClient.Do(httpReq)
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

	var voyageResp voyageEmbeddingResponse
	if err := json.Unmarshal(body, &voyageResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if voyageResp.Error != nil {
		return nil, fmt.Errorf("API error: %s", voyageResp.Error.Message)
	}

	embeddings := make([][]float32, len(voyageResp.Data))
	for _, data := range voyageResp.Data {
		embeddings[data.Index] = data.Embedding
	}

	return embeddings, nil
}

// EmbedSingle generates an embedding for a single text.
func (v *VoyageEmbedder) EmbedSingle(ctx context.Context, text string) ([]float32, error) {
	embeddings, err := v.Embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}

	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return embeddings[0], nil
}

// Dimensions returns the number of dimensions in the embeddings.
func (v *VoyageEmbedder) Dimensions() int {
	return v.config.Dimensions
}

// Close cleans up any resources.
func (v *VoyageEmbedder) Close() error {
	// Nothing to clean up for HTTP client
	return nil
}
