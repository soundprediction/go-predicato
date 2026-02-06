package types

type BadLlmCsvResponse struct {
	Error    error
	Response string
	Messages []*Message
}

// Role represents the role of a message sender.
type Role string

// Message represents a chat message.
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
}

// Response represents a chat completion response.
type Response struct {
	Content      string                 `json:"content"`
	TokensUsed   *TokenUsage            `json:"tokens_used,omitempty"`
	FinishReason string                 `json:"finish_reason,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	Model        string                 `json:"model,omitempty"`
}

// TokenUsage represents token usage statistics.
type TokenUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// PromptFunction is a function type that generates prompt messages from a context.
type PromptFunction func(context map[string]interface{}) ([]Message, error)

// PromptVersion is an interface for prompt versions that can be called with a context.
type PromptVersion interface {
	Call(context map[string]interface{}) ([]Message, error)
}
