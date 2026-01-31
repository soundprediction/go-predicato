package nlp

import (
	"context"

	"github.com/soundprediction/predicato/pkg/types"
)

// Client defines the interface for language model operations.
type Client interface {
	// Chat sends a chat completion request and returns the response.
	Chat(ctx context.Context, messages []types.Message) (*types.Response, error)

	// ChatWithStructuredOutput sends a chat completion request with structured output.
	ChatWithStructuredOutput(ctx context.Context, messages []types.Message, schema any) (*types.Response, error)

	// ExtractEntities extracts named entities from the given text.
	ExtractEntities(ctx context.Context, text string, entityTypes []string) ([]ExtractedEntity, error)

	// ExtractRelations extracts relationships between entities from the given text.
	// relationTypes is a list of relationship labels to consider.
	ExtractRelations(ctx context.Context, text string, relationTypes []string) ([]ExtractedRelation, error)

	// Summarize generates a summary of the provided text.
	Summarize(ctx context.Context, text string) (string, error)

	// GenerateText generates text based on the provided prompt.
	GenerateText(ctx context.Context, prompt string) (string, error)

	// GetCapabilities returns the list of capabilities supported by this client.
	GetCapabilities() []TaskCapability

	// Close cleans up any resources.
	Close() error
}

const (
	// RoleSystem represents a system message.
	RoleSystem types.Role = "system"
	// RoleUser represents a user message.
	RoleUser types.Role = "user"
	// RoleAssistant represents an assistant message.
	RoleAssistant types.Role = "assistant"
)

// Config holds legacy configuration for LLM clients.
//
// Deprecated: Use LLMConfig instead, which provides a fluent builder API
// and better support for various LLM providers. Migration example:
//
//	// Old:
//	config := &nlp.Config{Model: "gpt-4", Temperature: &temp}
//
//	// New:
//	config := nlp.NewLLMConfig().
//	    WithModel("gpt-4").
//	    WithTemperature(temp).
//	    WithAPIKey(apiKey)
//
// Config is kept for backward compatibility but will be removed in v2.0.
type Config struct {
	Model              string   `json:"model"`
	Temperature        *float32 `json:"temperature,omitempty"`
	MaxTokens          *int     `json:"max_tokens,omitempty"`
	TopP               *float32 `json:"top_p,omitempty"`
	TopK               *int     `json:"top_k,omitempty"`
	MinP               *float32 `json:"min_p,omitempty"`
	Stop               []string `json:"stop,omitempty"`
	BaseURL            string   `json:"base_url,omitempty"` // Custom base URL for OpenAI-compatible services
	InsecureSkipVerify bool     `json:"insecure_skip_verify,omitempty"`
}

// NewMessage creates a new message with the specified role and content.
func NewMessage(role types.Role, content string) types.Message {
	return types.Message{
		Role:    role,
		Content: content,
	}
}

// NewSystemMessage creates a new system message.
func NewSystemMessage(content string) types.Message {
	return NewMessage(RoleSystem, content)
}

// NewUserMessage creates a new user message.
func NewUserMessage(content string) types.Message {
	return NewMessage(RoleUser, content)
}

// NewAssistantMessage creates a new assistant message.
func NewAssistantMessage(content string) types.Message {
	return NewMessage(RoleAssistant, content)
}
