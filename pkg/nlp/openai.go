package nlp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"strings"

	"github.com/sashabaranov/go-openai"
	"github.com/soundprediction/predicato/pkg/types"
)

// OpenAIClient implements the Client interface for OpenAI's language models.
// This is the legacy client, use OpenAIGenericClient for new implementations.
type OpenAIClient struct {
	client *openai.Client
	config Config
}

// NewOpenAIClient creates a new OpenAI client.
// Supports OpenAI-compatible services through custom BaseURL configuration.
func NewOpenAIClient(apiKey string, config Config) (*OpenAIClient, error) {
	var client *openai.Client

	if config.BaseURL != "" {
		// Validate and configure custom base URL for OpenAI-compatible services
		if err := validateBaseURL(config.BaseURL); err != nil {
			return nil, fmt.Errorf("invalid base URL: %w", err)
		}

		// Use dummy API key if none provided (some services don't require authentication)
		if apiKey == "" {
			apiKey = "dummy-key"
		}

		// Create OpenAI client configuration with custom base URL
		clientConfig := openai.DefaultConfig(apiKey)
		clientConfig.BaseURL = config.BaseURL

		// Handle common base URL patterns
		// Many services expect "/v1" to be appended to the base URL
		if !hasAPIPath(config.BaseURL) {
			clientConfig.BaseURL = config.BaseURL + "/v1"
		}

		client = openai.NewClientWithConfig(clientConfig)
	} else {
		// Use default OpenAI client
		client = openai.NewClient(apiKey)
	}

	if config.Model == "" {
		if config.BaseURL != "" {
			config.Model = "gpt-3.5-turbo" // Default fallback for OpenAI-compatible services
		} else {
			config.Model = openai.GPT4o // Default for OpenAI
		}
	}

	return &OpenAIClient{
		client: client,
		config: config,
	}, nil
}

// Chat sends a chat completion request to OpenAI or OpenAI-compatible service.
func (c *OpenAIClient) Chat(ctx context.Context, messages []types.Message) (*types.Response, error) {
	req := c.buildChatRequest(messages, false, nil)

	resp, err := c.client.CreateChatCompletion(ctx, req)
	if err != nil {
		if c.config.BaseURL != "" {
			return nil, fmt.Errorf("openai-compatible chat completion failed: %w", err)
		}
		return nil, fmt.Errorf("openai chat completion failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		if c.config.BaseURL != "" {
			return nil, fmt.Errorf("no choices returned from openai-compatible service")
		}
		return nil, fmt.Errorf("no choices returned from openai")
	}

	choice := resp.Choices[0]
	response := &types.Response{
		Content:      choice.Message.Content,
		FinishReason: string(choice.FinishReason),
		Model:        resp.Model,
	}

	// Include token usage if available (some OpenAI-compatible services might not provide this)
	if resp.Usage.TotalTokens > 0 {
		response.TokensUsed = &types.TokenUsage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
	}

	return response, nil
}

// ChatWithStructuredOutput sends a chat completion request with structured output.
func (c *OpenAIClient) ChatWithStructuredOutput(ctx context.Context, messages []types.Message, schema any) (*types.Response, error) {
	req := c.buildChatRequest(messages, true, schema)

	resp, err := c.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("openai structured output failed: %w", err)
	}

	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices returned from openai")
	}

	choice := resp.Choices[0]
	response := &types.Response{
		Content:      choice.Message.Content,
		FinishReason: string(choice.FinishReason),
		Model:        resp.Model,
	}

	// Include token usage if available
	if resp.Usage.TotalTokens > 0 {
		response.TokensUsed = &types.TokenUsage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		}
	}

	return response, nil
}

// GetModel returns the model identifier.
func (c *OpenAIClient) GetModel() string {
	return c.config.Model
}

// Close cleans up resources (no-op for OpenAI client).
func (c *OpenAIClient) Close() error {
	return nil
}

// ExtractEntities implements the Client interface using prompt engineering
func (c *OpenAIClient) ExtractEntities(ctx context.Context, text string, entityTypes []string) ([]ExtractedEntity, error) {
	// Construct a prompt to extract entities
	sysPrompt := "You are an entity extraction system. Extract entities from the text. Return JSON output."
	if len(entityTypes) > 0 {
		sysPrompt += fmt.Sprintf(" specific entity types: %s", strings.Join(entityTypes, ", "))
	}

	prompt := fmt.Sprintf("Extract entities from the following text:\n\n%s", text)

	type EntityResult struct {
		Entities []ExtractedEntity `json:"entities"`
	}

	messages := []types.Message{
		NewSystemMessage(sysPrompt),
		NewUserMessage(prompt),
	}

	resp, err := c.ChatWithStructuredOutput(ctx, messages, EntityResult{})
	if err != nil {
		return nil, err
	}

	var result EntityResult
	if err := json.Unmarshal([]byte(resp.Content), &result); err != nil {
		return nil, fmt.Errorf("failed to parse extraction result: %w", err)
	}

	return result.Entities, nil
}

// ExtractRelations implements the Client interface using prompt engineering
func (c *OpenAIClient) ExtractRelations(ctx context.Context, text string, relationTypes []string) ([]ExtractedRelation, error) {
	sysPrompt := "You are a relationship extraction system. Extract relationships between entities from the text. Return JSON output."
	if len(relationTypes) > 0 {
		sysPrompt += fmt.Sprintf(" specific relation types: %s", strings.Join(relationTypes, ", "))
	}

	prompt := fmt.Sprintf("Extract relationships from the following text:\n\n%s", text)

	type RelationResult struct {
		Relations []ExtractedRelation `json:"relations"`
	}

	messages := []types.Message{
		NewSystemMessage(sysPrompt),
		NewUserMessage(prompt),
	}

	resp, err := c.ChatWithStructuredOutput(ctx, messages, RelationResult{})
	if err != nil {
		return nil, err
	}

	var result RelationResult
	if err := json.Unmarshal([]byte(resp.Content), &result); err != nil {
		return nil, fmt.Errorf("failed to parse relation result: %w", err)
	}

	return result.Relations, nil
}

// Summarize implements the Client interface
func (c *OpenAIClient) Summarize(ctx context.Context, text string) (string, error) {
	messages := []types.Message{
		NewSystemMessage("You are a helpful assistant that summarizes text."),
		NewUserMessage(fmt.Sprintf("Please summarize the following text:\n\n%s", text)),
	}

	resp, err := c.Chat(ctx, messages)
	if err != nil {
		return "", err
	}

	return resp.Content, nil
}

// GenerateText implements the Client interface
func (c *OpenAIClient) GenerateText(ctx context.Context, prompt string) (string, error) {
	messages := []types.Message{
		NewUserMessage(prompt),
	}

	resp, err := c.Chat(ctx, messages)
	if err != nil {
		return "", err
	}

	return resp.Content, nil
}

// GetCapabilities returns the list of capabilities supported by this client.
func (c *OpenAIClient) GetCapabilities() []TaskCapability {
	return []TaskCapability{
		TaskTextGeneration,
		TaskSummarization,
		TaskNamedEntityRecognition,
		TaskRelationExtraction,
	}
}

func (c *OpenAIClient) buildChatRequest(messages []types.Message, structuredOutput bool, schema any) openai.ChatCompletionRequest {
	openaiMessages := make([]openai.ChatCompletionMessage, len(messages))
	for i, msg := range messages {
		openaiMessages[i] = openai.ChatCompletionMessage{
			Role:    string(msg.Role),
			Content: msg.Content,
		}
	}

	req := openai.ChatCompletionRequest{
		Model:    c.config.Model,
		Messages: openaiMessages,
	}

	if c.config.Temperature != nil {
		req.Temperature = *c.config.Temperature
	}
	if c.config.MaxTokens != nil {
		req.MaxTokens = *c.config.MaxTokens
	}
	if c.config.TopP != nil {
		req.TopP = *c.config.TopP
	}
	if len(c.config.Stop) > 0 {
		req.Stop = c.config.Stop
	}

	if structuredOutput {
		req.ResponseFormat = &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeJSONObject,
		}

		// Add instruction for JSON output if not already present for OpenAI-compatible services
		if c.config.BaseURL != "" && len(openaiMessages) > 0 {
			lastMessage := &req.Messages[len(req.Messages)-1]
			if lastMessage.Role == string(RoleUser) {
				lastMessage.Content += "\n\nPlease respond with valid JSON only."
			}
		}
	}

	return req
}

// validateBaseURL validates the base URL format.
func validateBaseURL(baseURL string) error {
	if baseURL == "" {
		return fmt.Errorf("baseURL cannot be empty")
	}

	// Validate URL format
	parsedURL, err := url.Parse(baseURL)
	if err != nil {
		return fmt.Errorf("invalid baseURL format: %w", err)
	}

	// Ensure URL has a valid scheme
	if parsedURL.Scheme == "" {
		return fmt.Errorf("baseURL must include scheme (http:// or https://)")
	}

	// Ensure scheme is http or https
	if parsedURL.Scheme != "http" && parsedURL.Scheme != "https" {
		return fmt.Errorf("baseURL must use http:// or https:// scheme")
	}

	return nil
}

// hasAPIPath checks if the base URL already includes an API path component.
func hasAPIPath(baseURL string) bool {
	commonPaths := []string{"/v1", "/api", "/v1/", "/api/"}
	for _, path := range commonPaths {
		if len(baseURL) >= len(path) && baseURL[len(baseURL)-len(path):] == path {
			return true
		}
	}
	return false
}
