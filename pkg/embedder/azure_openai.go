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

// AzureOpenAIEmbedder implements the Client interface for Azure OpenAI embeddings.
type AzureOpenAIEmbedder struct {
	config       *AzureOpenAIConfig
	httpClient   *http.Client
	apiVersion   string
	deploymentID string
}

// AzureOpenAIConfig extends Config with Azure-specific settings.
type AzureOpenAIConfig struct {
	*Config
	APIKey       string `json:"api_key"`
	APIVersion   string `json:"api_version,omitempty"`
	DeploymentID string `json:"deployment_id"`
}

// NewAzureOpenAIEmbedder creates a new Azure OpenAI embedder.
func NewAzureOpenAIEmbedder(config *AzureOpenAIConfig) *AzureOpenAIEmbedder {
	if config.APIVersion == "" {
		config.APIVersion = "2024-02-15-preview"
	}
	if config.BatchSize == 0 {
		config.BatchSize = 100
	}

	return &AzureOpenAIEmbedder{
		config:       config,
		apiVersion:   config.APIVersion,
		deploymentID: config.DeploymentID,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// azureEmbeddingRequest represents the request structure for Azure OpenAI embeddings API.
type azureEmbeddingRequest struct {
	Model string   `json:"model,omitempty"`
	Input []string `json:"input"`
}

// azureEmbeddingResponse represents the response from Azure OpenAI embeddings API.
type azureEmbeddingResponse struct {
	Error  *azureEmbeddingError `json:"error,omitempty"`
	Object string               `json:"object"`
	Model  string               `json:"model"`
	Data   []azureEmbeddingData `json:"data"`
	Usage  azureEmbeddingUsage  `json:"usage"`
}

// azureEmbeddingData represents a single embedding in the response.
type azureEmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

// azureEmbeddingUsage represents usage information in the response.
type azureEmbeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

// azureEmbeddingError represents an error response.
type azureEmbeddingError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// Embed generates embeddings for the given texts.
func (a *AzureOpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, fmt.Errorf("no texts provided")
	}

	if a.deploymentID == "" {
		return nil, fmt.Errorf("deployment ID is required for Azure OpenAI")
	}

	var allEmbeddings [][]float32

	// Process texts in batches
	for i := 0; i < len(texts); i += a.config.BatchSize {
		end := i + a.config.BatchSize
		if end > len(texts) {
			end = len(texts)
		}

		batch := texts[i:end]
		embeddings, err := a.embedBatch(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("failed to embed batch: %w", err)
		}

		allEmbeddings = append(allEmbeddings, embeddings...)
	}

	return allEmbeddings, nil
}

// embedBatch processes a batch of texts.
func (a *AzureOpenAIEmbedder) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	req := azureEmbeddingRequest{
		Input: texts,
		Model: a.config.Model,
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Azure OpenAI URL format: https://{resource-name}.openai.azure.com/openai/deployments/{deployment-id}/embeddings?api-version={api-version}
	url := fmt.Sprintf("%s/openai/deployments/%s/embeddings?api-version=%s",
		a.config.BaseURL, a.deploymentID, a.apiVersion)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("api-key", a.config.APIKey)

	// Add any additional headers
	for key, value := range a.config.Headers {
		httpReq.Header.Set(key, value)
	}

	resp, err := a.httpClient.Do(httpReq)
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

	var azureResp azureEmbeddingResponse
	if err := json.Unmarshal(body, &azureResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if azureResp.Error != nil {
		return nil, fmt.Errorf("API error: %s", azureResp.Error.Message)
	}

	embeddings := make([][]float32, len(azureResp.Data))
	for _, data := range azureResp.Data {
		embeddings[data.Index] = data.Embedding
	}

	return embeddings, nil
}

// EmbedSingle generates an embedding for a single text.
func (a *AzureOpenAIEmbedder) EmbedSingle(ctx context.Context, text string) ([]float32, error) {
	embeddings, err := a.Embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}

	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return embeddings[0], nil
}

// Dimensions returns the number of dimensions in the embeddings.
func (a *AzureOpenAIEmbedder) Dimensions() int {
	return a.config.Dimensions
}

// Close cleans up any resources.
func (a *AzureOpenAIEmbedder) Close() error {
	// Nothing to clean up for HTTP client
	return nil
}
