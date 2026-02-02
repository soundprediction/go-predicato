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

// GeminiClient implements the Client interface for Google Gemini models.
type GeminiClient struct {
	config     *LLMConfig
	httpClient *http.Client
}

// NewGeminiClient creates a new Gemini client.
func NewGeminiClient(config *LLMConfig) *GeminiClient {
	if config.BaseURL == "" {
		config.BaseURL = "https://generativelanguage.googleapis.com"
	}

	return &GeminiClient{
		config: config,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// geminiRequest represents the request structure for Gemini API.
type geminiRequest struct {
	Contents         []geminiContent         `json:"contents"`
	GenerationConfig *geminiGenerationConfig `json:"generationConfig,omitempty"`
}

// geminiContent represents content in Gemini format.
type geminiContent struct {
	Role  string       `json:"role"`
	Parts []geminiPart `json:"parts"`
}

// geminiPart represents a part of content.
type geminiPart struct {
	Text string `json:"text"`
}

// geminiGenerationConfig represents generation configuration.
type geminiGenerationConfig struct {
	Temperature float64 `json:"temperature,omitempty"`
	MaxTokens   int     `json:"maxOutputTokens,omitempty"`
	TopP        float64 `json:"topP,omitempty"`
	TopK        int     `json:"topK,omitempty"`
}

// geminiResponse represents the response from Gemini API.
type geminiResponse struct {
	Candidates []geminiCandidate `json:"candidates"`
	Error      *geminiError      `json:"error,omitempty"`
}

// geminiCandidate represents a candidate response.
type geminiCandidate struct {
	Content geminiContent `json:"content"`
}

// geminiError represents an error response.
type geminiError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

// Chat implements the Client interface for Gemini.
func (g *GeminiClient) Chat(ctx context.Context, messages []types.Message) (*types.Response, error) {
	if len(messages) == 0 {
		return nil, fmt.Errorf("no messages provided")
	}

	// Convert messages to Gemini format
	contents := make([]geminiContent, 0, len(messages))

	for _, msg := range messages {
		role := string(msg.Role)
		// Convert OpenAI roles to Gemini roles
		if role == "assistant" {
			role = "model"
		} else if msg.Role == RoleSystem {
			// Gemini doesn't have a system role, prepend to first user message
			if len(contents) == 0 {
				contents = append(contents, geminiContent{
					Role:  "user",
					Parts: []geminiPart{{Text: msg.Content}},
				})
				continue
			} else {
				// Append to last user message if exists
				for i := len(contents) - 1; i >= 0; i-- {
					if contents[i].Role == "user" {
						contents[i].Parts[0].Text = msg.Content + "\n\n" + contents[i].Parts[0].Text
						break
					}
				}
				continue
			}
		}

		contents = append(contents, geminiContent{
			Role:  role,
			Parts: []geminiPart{{Text: msg.Content}},
		})
	}

	req := geminiRequest{
		Contents: contents,
		GenerationConfig: &geminiGenerationConfig{
			Temperature: float64(g.config.Temperature),
			MaxTokens:   g.config.MaxTokens,
		},
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:generateContent?key=%s",
		g.config.BaseURL, g.config.Model, g.config.APIKey)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

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

	var geminiResp geminiResponse
	if err := json.Unmarshal(body, &geminiResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if geminiResp.Error != nil {
		return nil, fmt.Errorf("API error: %s", geminiResp.Error.Message)
	}

	if len(geminiResp.Candidates) == 0 || len(geminiResp.Candidates[0].Content.Parts) == 0 {
		return nil, fmt.Errorf("no content in response")
	}

	// TODO: Capture actual token usage if available in response
	return &types.Response{
		Content: geminiResp.Candidates[0].Content.Parts[0].Text,
	}, nil
}

// GetCapabilities returns the list of capabilities supported by this client.
func (g *GeminiClient) GetCapabilities() []TaskCapability {
	return []TaskCapability{TaskTextGeneration}
}

// ChatWithStructuredOutput implements structured output for Gemini.
// Similar to Anthropic, Gemini uses prompt engineering for structured output.
func (g *GeminiClient) ChatWithStructuredOutput(ctx context.Context, messages []types.Message, schema interface{}) (*types.Response, error) {
	// Add a message requesting JSON format
	schemaBytes, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}

	modifiedMessages := append(messages, types.Message{
		Role:    "user",
		Content: fmt.Sprintf("Please respond with valid JSON that matches this schema: %s", string(schemaBytes)),
	})

	resp, err := g.Chat(ctx, modifiedMessages)
	if err != nil {
		return nil, err
	}

	// GeminiClient.Chat now returns *types.Response
	return resp, nil
}

// GetModel returns the model identifier.
func (g *GeminiClient) GetModel() string {
	return g.config.Model
}

// Close releases any resources held by the client.
// For GeminiClient, this closes the underlying HTTP client's idle connections.
func (g *GeminiClient) Close() error {
	if g.httpClient != nil {
		g.httpClient.CloseIdleConnections()
	}
	return nil
}
