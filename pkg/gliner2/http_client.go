package gliner2

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type HTTPClient struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
	timeout    time.Duration
}

type LocalConfig struct {
	Endpoint string        `json:"endpoint"`
	Timeout  time.Duration `json:"timeout"`
}

type FastinoConfig struct {
	Endpoint string        `json:"endpoint"`
	APIKey   string        `json:"api_key"`
	Timeout  time.Duration `json:"timeout"`
}

// Future: NativeConfig for go-gline-rs GLInER2
// type NativeConfig struct {
// 	ModelPath string `json:"model_path"`
// }

type Config struct {
	Provider Provider       `json:"provider"`
	Local    *LocalConfig   `json:"local,omitempty"`
	Fastino  *FastinoConfig `json:"fastino,omitempty"`
	// Future:
	// Native *NativeConfig `json:"native,omitempty"`
}

func NewHTTPClient(config Config) (*HTTPClient, error) {
	var baseURL, apiKey string
	var timeout time.Duration

	switch config.Provider {
	case ProviderLocal:
		if config.Local == nil {
			return nil, fmt.Errorf("local config required for local provider")
		}
		baseURL = config.Local.Endpoint
		timeout = config.Local.Timeout
	case ProviderFastino:
		if config.Fastino == nil {
			return nil, fmt.Errorf("fastino config required for fastino provider")
		}
		baseURL = config.Fastino.Endpoint
		apiKey = config.Fastino.APIKey
		timeout = config.Fastino.Timeout
	default:
		return nil, fmt.Errorf("unsupported provider: %v", config.Provider)
	}

	if baseURL == "" {
		return nil, fmt.Errorf("endpoint URL is required")
	}

	client := &HTTPClient{
		baseURL: baseURL,
		apiKey:  apiKey,
		timeout: timeout,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}

	return client, nil
}

func (c *HTTPClient) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/health", nil)
	if err != nil {
		return fmt.Errorf("failed to create health request: %w", err)
	}

	// Add API key header for Fastino
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed with status: %d", resp.StatusCode)
	}

	return nil
}

func (c *HTTPClient) ExtractEntities(ctx context.Context, text string, schema interface{}, threshold float64) (*EntityResult, error) {
	request := ExtractRequest{
		Task:      "extract_entities",
		Text:      text,
		Schema:    schema,
		Threshold: threshold,
	}

	var result EntityResult
	err := c.makeRequest(ctx, request, &result)
	if err != nil {
		return nil, fmt.Errorf("entity extraction failed: %w", err)
	}

	return &result, nil
}

func (c *HTTPClient) ExtractRelations(ctx context.Context, text string, schema interface{}, threshold float64) (*RelationResult, error) {
	request := ExtractRequest{
		Task:      "extract_relations", // GLInER2 uses extract_relations
		Text:      text,
		Schema:    schema,
		Threshold: threshold,
	}

	var result RelationResult
	err := c.makeRequest(ctx, request, &result)
	if err != nil {
		return nil, fmt.Errorf("relation extraction failed: %w", err)
	}

	return &result, nil
}

func (c *HTTPClient) ExtractFacts(ctx context.Context, text string, schema interface{}, threshold float64) ([]Fact, error) {
	// GLInER2 uses relation extraction for facts
	relations, err := c.ExtractRelations(ctx, text, schema, threshold)
	if err != nil {
		return nil, err
	}

	// Convert GLInER2 relations to Predicato facts
	var facts []Fact
	for relationType, tuples := range relations.RelationExtraction {
		for _, tuple := range tuples {
			fact := Fact{
				Source:     tuple.Head,
				Target:     tuple.Tail,
				Type:       relationType,
				Confidence: 1.0, // GLInER2 doesn't provide confidence in tuple format
			}
			facts = append(facts, fact)
		}
	}

	return facts, nil
}

func (c *HTTPClient) ClassifyText(ctx context.Context, text string, schema interface{}, threshold float64) (*ClassificationResult, error) {
	request := ExtractRequest{
		Task:      "classify_text",
		Text:      text,
		Schema:    schema,
		Threshold: threshold,
	}

	var result ClassificationResult
	err := c.makeRequest(ctx, request, &result)
	if err != nil {
		return nil, fmt.Errorf("text classification failed: %w", err)
	}

	return &result, nil
}

func (c *HTTPClient) ExtractStructured(ctx context.Context, text string, schema interface{}, threshold float64) (*StructuredResult, error) {
	request := ExtractRequest{
		Task:      "extract_json",
		Text:      text,
		Schema:    schema,
		Threshold: threshold,
	}

	var result StructuredResult
	err := c.makeRequest(ctx, request, &result)
	if err != nil {
		return nil, fmt.Errorf("structured extraction failed: %w", err)
	}

	return &result, nil
}

func (c *HTTPClient) Close() error {
	// No explicit cleanup needed for HTTP client
	return nil
}

func (c *HTTPClient) makeRequest(ctx context.Context, request ExtractRequest, result interface{}) error {
	reqBody, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/gliner-2", strings.NewReader(string(reqBody)))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	// Add API key header for Fastino
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var apiError struct {
			Detail string `json:"detail"`
		}
		_ = json.Unmarshal(body, &apiError)
		return fmt.Errorf("API error (status %d): %s", resp.StatusCode, apiError.Detail)
	}

	response := ExtractResponse{Result: result}
	return json.Unmarshal(body, &response)
}
