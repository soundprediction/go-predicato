package handlers

import (
	"net/http"

	"github.com/soundprediction/predicato/pkg/config"
)

// ConfigHandler handles configuration endpoint requests
type ConfigHandler struct {
	config *config.Config
}

// NewConfigHandler creates a new config handler
func NewConfigHandler(cfg *config.Config) *ConfigHandler {
	return &ConfigHandler{
		config: cfg,
	}
}

// ModelsResponse represents the response for the models endpoint
type ModelsResponse struct {
	Embedding EmbeddingModelsResponse `json:"embedding"`
	NLP       NLPModelsResponse       `json:"nlp"`
}

// NLPModelsResponse represents NLP model configuration (without API keys)
type NLPModelsResponse struct {
	Models      map[string]NLPModelResponse `json:"models"`
	RouterRules []RouterRuleResponse        `json:"router_rules,omitempty"`
}

// NLPModelResponse represents a single NLP model configuration
type NLPModelResponse struct {
	Provider    string  `json:"provider"`
	Model       string  `json:"model"`
	BaseURL     string  `json:"base_url"`
	Temperature float32 `json:"temperature"`
	MaxTokens   int     `json:"max_tokens"`
}

// RouterRuleResponse represents a routing rule
type RouterRuleResponse struct {
	Usage    string `json:"usage"`
	Provider string `json:"provider"`
	Fallback string `json:"fallback,omitempty"`
}

// EmbeddingModelsResponse represents embedding model configuration
type EmbeddingModelsResponse struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
	BaseURL  string `json:"base_url"`
}

// GetModels handles GET /api/v1/config/models
// Returns the configured NLP and embedding models without exposing API keys
func (h *ConfigHandler) GetModels(w http.ResponseWriter, r *http.Request) {
	// Build NLP models response
	nlpModels := make(map[string]NLPModelResponse)
	for name, modelCfg := range h.config.NLP.Models {
		nlpModels[name] = NLPModelResponse{
			Provider:    modelCfg.Provider,
			Model:       modelCfg.Model,
			BaseURL:     modelCfg.BaseURL,
			Temperature: modelCfg.Temperature,
			MaxTokens:   modelCfg.MaxTokens,
		}
	}

	// Build router rules response
	var routerRules []RouterRuleResponse
	for _, rule := range h.config.NLP.RouterRules {
		routerRules = append(routerRules, RouterRuleResponse{
			Usage:    rule.Usage,
			Provider: rule.Provider,
			Fallback: rule.Fallback,
		})
	}

	response := ModelsResponse{
		NLP: NLPModelsResponse{
			Models:      nlpModels,
			RouterRules: routerRules,
		},
		Embedding: EmbeddingModelsResponse{
			Provider: h.config.Embedding.Provider,
			Model:    h.config.Embedding.Model,
			BaseURL:  h.config.Embedding.BaseURL,
		},
	}

	writeJSON(w, http.StatusOK, response)
}

// GetNLPModels handles GET /api/v1/config/nlp
// Returns just the NLP model configuration
func (h *ConfigHandler) GetNLPModels(w http.ResponseWriter, r *http.Request) {
	nlpModels := make(map[string]NLPModelResponse)
	for name, modelCfg := range h.config.NLP.Models {
		nlpModels[name] = NLPModelResponse{
			Provider:    modelCfg.Provider,
			Model:       modelCfg.Model,
			BaseURL:     modelCfg.BaseURL,
			Temperature: modelCfg.Temperature,
			MaxTokens:   modelCfg.MaxTokens,
		}
	}

	var routerRules []RouterRuleResponse
	for _, rule := range h.config.NLP.RouterRules {
		routerRules = append(routerRules, RouterRuleResponse{
			Usage:    rule.Usage,
			Provider: rule.Provider,
			Fallback: rule.Fallback,
		})
	}

	response := NLPModelsResponse{
		Models:      nlpModels,
		RouterRules: routerRules,
	}

	writeJSON(w, http.StatusOK, response)
}

// GetEmbeddingModels handles GET /api/v1/config/embedding
// Returns just the embedding model configuration
func (h *ConfigHandler) GetEmbeddingModels(w http.ResponseWriter, r *http.Request) {
	response := EmbeddingModelsResponse{
		Provider: h.config.Embedding.Provider,
		Model:    h.config.Embedding.Model,
		BaseURL:  h.config.Embedding.BaseURL,
	}

	writeJSON(w, http.StatusOK, response)
}
