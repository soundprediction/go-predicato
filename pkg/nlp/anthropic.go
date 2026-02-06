package nlp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/soundprediction/predicato/pkg/types"
)

// AnthropicClient implements the Client interface for Anthropic Claude models.
type AnthropicClient struct {
	config     *LLMConfig
	httpClient *http.Client
}

// NewAnthropicClient creates a new Anthropic client.
func NewAnthropicClient(config *LLMConfig) *AnthropicClient {
	if config.BaseURL == "" {
		config.BaseURL = "https://api.anthropic.com"
	}

	return &AnthropicClient{
		config: config,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// anthropicRequest represents the request structure for Anthropic API.
type anthropicRequest struct {
	Model       string             `json:"model"`
	System      string             `json:"system,omitempty"`
	Messages    []anthropicMessage `json:"messages"`
	MaxTokens   int                `json:"max_tokens"`
	Temperature float64            `json:"temperature,omitempty"`
}

// anthropicMessage represents a message in Anthropic format.
type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// anthropicResponse represents the response from Anthropic API.
type anthropicResponse struct {
	Error   *anthropicError    `json:"error,omitempty"`
	Content []anthropicContent `json:"content"`
}

// anthropicContent represents content in the response.
type anthropicContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// anthropicError represents an error response.
type anthropicError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// Chat implements the Client interface for Anthropic.
func (a *AnthropicClient) Chat(ctx context.Context, messages []types.Message) (*types.Response, error) {
	if len(messages) == 0 {
		return nil, fmt.Errorf("no messages provided")
	}

	// Convert messages to Anthropic format
	anthropicMessages := make([]anthropicMessage, 0, len(messages))
	var systemMessage string

	for _, msg := range messages {
		if msg.Role == RoleSystem {
			// Anthropic handles system messages separately
			systemMessage = msg.Content
		} else {
			anthropicMessages = append(anthropicMessages, anthropicMessage{
				Role:    string(msg.Role),
				Content: msg.Content,
			})
		}
	}

	req := anthropicRequest{
		Model:       a.config.Model,
		MaxTokens:   a.config.MaxTokens,
		Messages:    anthropicMessages,
		Temperature: float64(a.config.Temperature),
		System:      systemMessage,
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", a.config.BaseURL+"/v1/messages", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", a.config.APIKey)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

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

	var anthropicResp anthropicResponse
	if err := json.Unmarshal(body, &anthropicResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if anthropicResp.Error != nil {
		return nil, fmt.Errorf("API error: %s", anthropicResp.Error.Message)
	}

	if len(anthropicResp.Content) == 0 {
		return nil, fmt.Errorf("no content in response")
	}

	// TODO: Capture actual token usage if available in response
	return &types.Response{
		Content: anthropicResp.Content[0].Text,
	}, nil
}

// GetCapabilities returns the list of capabilities supported by this client.
func (a *AnthropicClient) GetCapabilities() []TaskCapability {
	return []TaskCapability{TaskTextGeneration}
}

// ChatWithStructuredOutput implements structured output for Anthropic.
// Note: Anthropic doesn't natively support structured output like OpenAI,
// so this implementation uses prompt engineering to request JSON format.
func (a *AnthropicClient) ChatWithStructuredOutput(ctx context.Context, messages []types.Message, schema interface{}) (*types.Response, error) {
	// Add a message requesting JSON format
	schemaBytes, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}

	modifiedMessages := append(messages, types.Message{
		Role:    "user",
		Content: fmt.Sprintf("Please respond with valid JSON that matches this schema: %s", string(schemaBytes)),
	})

	resp, err := a.Chat(ctx, modifiedMessages)
	if err != nil {
		return nil, err
	}

	// AnthropicClient.Chat now returns *types.Response
	return resp, nil
}

// GetModel returns the model identifier.
func (a *AnthropicClient) GetModel() string {
	return a.config.Model
}

// Close releases any resources held by the client.
// For AnthropicClient, this closes the underlying HTTP client's idle connections.
func (a *AnthropicClient) Close() error {
	if a.httpClient != nil {
		a.httpClient.CloseIdleConnections()
	}
	return nil
}
