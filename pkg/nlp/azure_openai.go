package nlp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/soundprediction/predicato/pkg/types"
)

// AzureOpenAIClient implements the Client interface for Azure OpenAI models.
type AzureOpenAIClient struct {
	config       *LLMConfig
	httpClient   *http.Client
	apiVersion   string
	deploymentID string
}

// AzureOpenAIConfig extends LLMConfig with Azure-specific settings.
type AzureOpenAIConfig struct {
	*LLMConfig
	APIVersion   string `json:"api_version,omitempty"`
	DeploymentID string `json:"deployment_id,omitempty"`
}

// NewAzureOpenAIClient creates a new Azure OpenAI client.
func NewAzureOpenAIClient(config *AzureOpenAIConfig) *AzureOpenAIClient {
	if config.APIVersion == "" {
		config.APIVersion = "2024-02-15-preview"
	}

	return &AzureOpenAIClient{
		config:       config.LLMConfig,
		apiVersion:   config.APIVersion,
		deploymentID: config.DeploymentID,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// azureOpenAIRequest represents the request structure for Azure OpenAI API.
type azureOpenAIRequest struct {
	Messages    []azureOpenAIMessage `json:"messages"`
	MaxTokens   int                  `json:"max_tokens,omitempty"`
	Temperature float64              `json:"temperature,omitempty"`
	Stream      bool                 `json:"stream"`
}

// azureOpenAIMessage represents a message in Azure OpenAI format.
type azureOpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// azureOpenAIResponse represents the response from Azure OpenAI API.
type azureOpenAIResponse struct {
	Error   *azureOpenAIError   `json:"error,omitempty"`
	ID      string              `json:"id"`
	Object  string              `json:"object"`
	Model   string              `json:"model"`
	Choices []azureOpenAIChoice `json:"choices"`
	Created int64               `json:"created"`
}

// azureOpenAIChoice represents a choice in the response.
type azureOpenAIChoice struct {
	Message      azureOpenAIMessage `json:"message"`
	FinishReason string             `json:"finish_reason"`
	Index        int                `json:"index"`
}

// azureOpenAIError represents an error response.
type azureOpenAIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// Chat implements the Client interface for Azure OpenAI.
func (a *AzureOpenAIClient) Chat(ctx context.Context, messages []types.Message) (*types.Response, error) {
	if len(messages) == 0 {
		return nil, fmt.Errorf("no messages provided")
	}

	if a.deploymentID == "" {
		return nil, fmt.Errorf("deployment ID is required for Azure OpenAI")
	}

	// Convert messages to Azure OpenAI format
	azureMessages := make([]azureOpenAIMessage, 0, len(messages))
	for _, msg := range messages {
		azureMessages = append(azureMessages, azureOpenAIMessage{
			Role:    string(msg.Role),
			Content: msg.Content,
		})
	}

	req := azureOpenAIRequest{
		Messages:    azureMessages,
		MaxTokens:   a.config.MaxTokens,
		Temperature: float64(a.config.Temperature),
		Stream:      false,
	}

	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Azure OpenAI URL format: https://{resource-name}.openai.azure.com/openai/deployments/{deployment-id}/chat/completions?api-version={api-version}
	url := fmt.Sprintf("%s/openai/deployments/%s/chat/completions?api-version=%s",
		a.config.BaseURL, a.deploymentID, a.apiVersion)

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("api-key", a.config.APIKey)

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

	var azureResp azureOpenAIResponse
	if err := json.Unmarshal(body, &azureResp); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if azureResp.Error != nil {
		return nil, fmt.Errorf("API error: %s", azureResp.Error.Message)
	}

	if len(azureResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	// TODO: Capture actual token usage if available in response
	return &types.Response{
		Content: azureResp.Choices[0].Message.Content,
	}, nil
}

// GetCapabilities returns the list of capabilities supported by this client.
func (a *AzureOpenAIClient) GetCapabilities() []TaskCapability {
	return []TaskCapability{TaskTextGeneration, TaskExtendedExtraction}
}

// GetModel returns the model identifier.
func (a *AzureOpenAIClient) GetModel() string {
	if a.deploymentID != "" {
		return a.deploymentID
	}
	return a.config.Model
}

// ChatWithStructuredOutput implements structured output for Azure OpenAI.
// Azure OpenAI supports structured output similar to OpenAI.
func (a *AzureOpenAIClient) ChatWithStructuredOutput(ctx context.Context, messages []types.Message, schema interface{}) (*types.Response, error) {
	// For now, use prompt engineering approach
	schemaBytes, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}

	modifiedMessages := append(messages, types.Message{
		Role:    RoleUser,
		Content: fmt.Sprintf("Please respond with valid JSON that matches this schema: %s", string(schemaBytes)),
	})

	resp, err := a.Chat(ctx, modifiedMessages)
	if err != nil {
		return nil, err
	}

	// AzureOpenAIClient.Chat now returns *types.Response
	return resp, nil
}

// Close cleans up resources (no-op for Azure OpenAI client).
func (a *AzureOpenAIClient) Close() error {
	return nil
}

// ExtractEntities implements the Client interface using prompt engineering.
func (a *AzureOpenAIClient) ExtractEntities(ctx context.Context, text string, entityTypes []string) ([]ExtractedEntity, error) {
	sysPrompt := "You are an entity extraction system. Extract entities from the text. Return JSON output."
	if len(entityTypes) > 0 {
		sysPrompt += fmt.Sprintf(" Specific entity types: %s", strings.Join(entityTypes, ", "))
	}

	messages := []types.Message{
		NewSystemMessage(sysPrompt),
		NewUserMessage(fmt.Sprintf("Extract entities from the following text:\n\n%s", text)),
	}

	resp, err := a.ChatWithStructuredOutput(ctx, messages, nil)
	if err != nil {
		return nil, err
	}

	type EntityResult struct {
		Entities []ExtractedEntity `json:"entities"`
	}
	var result EntityResult
	if err := json.Unmarshal([]byte(resp.Content), &result); err != nil {
		return nil, fmt.Errorf("failed to parse extraction result: %w", err)
	}
	return result.Entities, nil
}

// ExtractRelations implements the Client interface using prompt engineering.
func (a *AzureOpenAIClient) ExtractRelations(ctx context.Context, text string, relationTypes []string) ([]ExtractedRelation, error) {
	sysPrompt := "You are a relationship extraction system. Extract relationships between entities from the text. Return JSON output."
	if len(relationTypes) > 0 {
		sysPrompt += fmt.Sprintf(" Specific relation types: %s", strings.Join(relationTypes, ", "))
	}

	messages := []types.Message{
		NewSystemMessage(sysPrompt),
		NewUserMessage(fmt.Sprintf("Extract relationships from the following text:\n\n%s", text)),
	}

	resp, err := a.ChatWithStructuredOutput(ctx, messages, nil)
	if err != nil {
		return nil, err
	}

	type RelationResult struct {
		Relations []ExtractedRelation `json:"relations"`
	}
	var result RelationResult
	if err := json.Unmarshal([]byte(resp.Content), &result); err != nil {
		return nil, fmt.Errorf("failed to parse relation result: %w", err)
	}
	return result.Relations, nil
}

// Summarize implements the Client interface.
func (a *AzureOpenAIClient) Summarize(ctx context.Context, text string) (string, error) {
	messages := []types.Message{
		NewSystemMessage("You are a helpful assistant that summarizes text."),
		NewUserMessage(fmt.Sprintf("Please summarize the following text:\n\n%s", text)),
	}
	resp, err := a.Chat(ctx, messages)
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

// GenerateText implements the Client interface.
func (a *AzureOpenAIClient) GenerateText(ctx context.Context, prompt string) (string, error) {
	messages := []types.Message{
		NewUserMessage(prompt),
	}
	resp, err := a.Chat(ctx, messages)
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

// ExtractExtended performs structured extraction (entities, relations, triples, rules) using prompt engineering.
func (a *AzureOpenAIClient) ExtractExtended(ctx context.Context, text string, entityTypes, relationTypes []string) (*ExtendedExtractionResult, error) {
	return ExtractExtendedHelper(ctx, a, text, entityTypes, relationTypes)
}
