package nlp

import (
	"context"
	"fmt"
	"strings"

	"github.com/soundprediction/predicato/pkg/config"
	"github.com/soundprediction/predicato/pkg/types"
)

// RouterClient routes requests to specific LLM providers based on rules
type RouterClient struct {
	defaultClient Client
	providers     map[string]Client
	rules         []config.RouterRule
}

// NewRouterClient creates a new router client
func NewRouterClient(providers map[string]Client, rules []config.RouterRule) (*RouterClient, error) {
	if len(providers) == 0 {
		return nil, fmt.Errorf("no providers configured")
	}

	// Determine default client (first one or specific "default" key)
	var defaultClient Client
	if client, ok := providers["default"]; ok {
		defaultClient = client
	} else {
		// Pick any
		for _, client := range providers {
			defaultClient = client
			break
		}
	}

	return &RouterClient{
		providers:     providers,
		rules:         rules,
		defaultClient: defaultClient,
	}, nil
}

// getClientForContext determines which client to use based on context
func (r *RouterClient) getClientForContext(ctx context.Context) (Client, string, Client) {
	usage, ok := ctx.Value(types.ContextKeyUsage).(string)
	if !ok || usage == "" {
		return r.defaultClient, "default", nil
	}

	// Find matching rule
	for _, rule := range r.rules {
		if strings.EqualFold(rule.Usage, usage) {
			primary, ok := r.providers[rule.Provider]
			if ok {
				var fallback Client
				if rule.Fallback != "" {
					fallback = r.providers[rule.Fallback]
				}
				return primary, rule.Provider, fallback
			}
		}
	}

	return r.defaultClient, "default", nil
}

// Chat implements Client with routing and fallback
func (r *RouterClient) Chat(ctx context.Context, messages []types.Message) (*types.Response, error) {
	primary, _, fallback := r.getClientForContext(ctx)

	resp, err := primary.Chat(ctx, messages)
	if err != nil {
		if fallback != nil {
			// Log routing fallback?
			// fmt.Printf("Routing fallback triggered: %v\n", err)
			return fallback.Chat(ctx, messages)
		}
		return nil, err
	}
	return resp, nil
}

// ChatWithStructuredOutput implements Client with routing and fallback
func (r *RouterClient) ChatWithStructuredOutput(ctx context.Context, messages []types.Message, schema any) (*types.Response, error) {
	primary, _, fallback := r.getClientForContext(ctx)

	resp, err := primary.ChatWithStructuredOutput(ctx, messages, schema)
	if err != nil {
		if fallback != nil {
			return fallback.ChatWithStructuredOutput(ctx, messages, schema)
		}
		return nil, err
	}
	return resp, nil
}

// GetModel returns the model identifier of the default client.
func (r *RouterClient) GetModel() string {
	if r.defaultClient != nil {
		return r.defaultClient.GetModel()
	}
	return "router"
}

// Close closes all providers
func (r *RouterClient) Close() error {
	var errs []string
	for id, provider := range r.providers {
		if err := provider.Close(); err != nil {
			errs = append(errs, fmt.Sprintf("%s: %v", id, err))
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("errors closing providers: %s", strings.Join(errs, "; "))
	}
	return nil
}

// ExtractEntities implements Client with routing and fallback
func (r *RouterClient) ExtractEntities(ctx context.Context, text string, entityTypes []string) ([]ExtractedEntity, error) {
	primary, _, fallback := r.getClientForContext(ctx)
	result, err := primary.ExtractEntities(ctx, text, entityTypes)
	if err != nil && fallback != nil {
		return fallback.ExtractEntities(ctx, text, entityTypes)
	}
	return result, err
}

// ExtractRelations implements Client with routing and fallback
func (r *RouterClient) ExtractRelations(ctx context.Context, text string, relationTypes []string) ([]ExtractedRelation, error) {
	primary, _, fallback := r.getClientForContext(ctx)
	result, err := primary.ExtractRelations(ctx, text, relationTypes)
	if err != nil && fallback != nil {
		return fallback.ExtractRelations(ctx, text, relationTypes)
	}
	return result, err
}

// ExtractExtended implements Client with routing and fallback
func (r *RouterClient) ExtractExtended(ctx context.Context, text string, entityTypes, relationTypes []string) (*ExtendedExtractionResult, error) {
	primary, _, fallback := r.getClientForContext(ctx)
	result, err := primary.ExtractExtended(ctx, text, entityTypes, relationTypes)
	if err != nil && fallback != nil {
		return fallback.ExtractExtended(ctx, text, entityTypes, relationTypes)
	}
	return result, err
}

// Summarize implements Client with routing and fallback
func (r *RouterClient) Summarize(ctx context.Context, text string) (string, error) {
	primary, _, fallback := r.getClientForContext(ctx)
	result, err := primary.Summarize(ctx, text)
	if err != nil && fallback != nil {
		return fallback.Summarize(ctx, text)
	}
	return result, err
}

// GenerateText implements Client with routing and fallback
func (r *RouterClient) GenerateText(ctx context.Context, prompt string) (string, error) {
	primary, _, fallback := r.getClientForContext(ctx)
	result, err := primary.GenerateText(ctx, prompt)
	if err != nil && fallback != nil {
		return fallback.GenerateText(ctx, prompt)
	}
	return result, err
}

// GetCapabilities returns the list of capabilities supported by this client.
func (r *RouterClient) GetCapabilities() []TaskCapability {
	// For router, we can return the union of capabilities, or just delegation to default.
	// safe approach: return default client's capabilities.
	if r.defaultClient != nil {
		return r.defaultClient.GetCapabilities()
	}
	return []TaskCapability{}
}
