package nlp

// ModelSize represents the size/complexity of the model to use
type ModelSize string

const (
	// ModelSizeSmall represents a small, fast model for simple tasks
	ModelSizeSmall ModelSize = "small"
	// ModelSizeMedium represents a medium model for more complex tasks
	ModelSizeMedium ModelSize = "medium"
)

// Default configuration values
const (
	DefaultMaxTokens   = 8192
	DefaultTemperature = 1.0
)

// LLMConfig holds configuration for LLM clients, matching Python LLMConfig structure
type LLMConfig struct {
	// APIKey is the authentication key for accessing the LLM API.
	// Excluded from JSON serialization to prevent accidental exposure in logs/responses.
	APIKey string `json:"-"`

	// Model is the specific LLM model to use for generating responses
	Model string `json:"model,omitempty"`

	// BaseURL is the base URL of the LLM API service
	BaseURL string `json:"base_url,omitempty"`

	// Temperature controls randomness in generation (0.0 to 2.0)
	// Recommended: 0.7 for non-thinking mode
	Temperature float32 `json:"temperature,omitempty"`

	// MaxTokens is the maximum number of tokens to generate
	MaxTokens int `json:"max_tokens,omitempty"`

	// TopP controls nucleus sampling (0.0 to 1.0)
	// Recommended: 0.8 for non-thinking mode
	TopP float32 `json:"top_p,omitempty"`

	// TopK controls top-k sampling (provider-specific, e.g., LM Studio)
	// Recommended: 20 for non-thinking mode
	TopK int `json:"top_k,omitempty"`

	// MinP controls minimum probability threshold (provider-specific, e.g., LM Studio)
	// Recommended: 0.0 for non-thinking mode
	MinP       float32 `json:"min_p,omitempty"`
	MaxRetries int     `json:"max_retries,omitempty"`

	// SmallModel is the model to use for simpler prompts
	SmallModel string `json:"small_model,omitempty"`

	// InsecureSkipVerify disables TLS certificate verification (useful for local self-signed certs)
	InsecureSkipVerify bool `json:"insecure_skip_verify,omitempty"`
}

// NewLLMConfig creates a new LLMConfig with default values
func NewLLMConfig() *LLMConfig {
	return &LLMConfig{
		Temperature: DefaultTemperature,
		MaxTokens:   DefaultMaxTokens,
	}
}

// WithAPIKey sets the API key
func (c *LLMConfig) WithAPIKey(apiKey string) *LLMConfig {
	c.APIKey = apiKey
	return c
}

// WithModel sets the model
func (c *LLMConfig) WithModel(model string) *LLMConfig {
	c.Model = model
	return c
}

// WithBaseURL sets the base URL
func (c *LLMConfig) WithBaseURL(baseURL string) *LLMConfig {
	c.BaseURL = baseURL
	return c
}

// WithTemperature sets the temperature
func (c *LLMConfig) WithTemperature(temperature float32) *LLMConfig {
	c.Temperature = temperature
	return c
}

// WithMaxTokens sets the max tokens
func (c *LLMConfig) WithMaxTokens(maxTokens int) *LLMConfig {
	c.MaxTokens = maxTokens
	return c
}

// WithSmallModel sets the small model
func (c *LLMConfig) WithSmallModel(smallModel string) *LLMConfig {
	c.SmallModel = smallModel
	return c
}

// WithTopP sets the top-p value
func (c *LLMConfig) WithTopP(topP float32) *LLMConfig {
	c.TopP = topP
	return c
}

// WithTopK sets the top-k value
func (c *LLMConfig) WithTopK(topK int) *LLMConfig {
	c.TopK = topK
	return c
}

// WithMinP sets the min-p value
func (c *LLMConfig) WithMinP(minP float32) *LLMConfig {
	c.MinP = minP
	return c
}
