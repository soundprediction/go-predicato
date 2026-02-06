package gliner2

import (
	"context"
	"fmt"

	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/types"
)

// Client provides unified access to GLInER2 functionality through different providers
type Client struct {
	httpClient *HTTPClient
	// Future: go-gline-rs GLInER2
	nativeClient *NativeClient // for go-gline-rs GLInER2 when available
	config       Config
	provider     Provider
}

func NewClient(config Config) (*Client, error) {
	switch config.Provider {
	case ProviderNative:
		nativeClient, err := NewNativeClient(config)
		if err != nil {
			return nil, fmt.Errorf("failed to create native client: %w", err)
		}
		return &Client{
			provider:     config.Provider,
			nativeClient: nativeClient,
			config:       config,
		}, nil
	case ProviderLocal, ProviderFastino:
		httpClient, err := NewHTTPClient(config)
		if err != nil {
			return nil, fmt.Errorf("failed to create HTTP client: %w", err)
		}
		return &Client{
			provider:   config.Provider,
			httpClient: httpClient,
			config:     config,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported provider: %v", config.Provider)
	}
}

func (c *Client) Close() error {
	if c.httpClient != nil {
		return c.httpClient.Close()
	}
	if c.nativeClient != nil {
		return c.nativeClient.Close()
	}
	return nil
}

func (c *Client) GetCapabilities() []nlp.TaskCapability {
	// GLInER2 supports these capabilities
	return []nlp.TaskCapability{
		nlp.TaskNamedEntityRecognition,
		nlp.TaskRelationExtraction,
	}
}

func (c *Client) GetModel() string {
	switch c.provider {
	case ProviderLocal:
		return "gliner-2-local"
	case ProviderFastino:
		return "gliner-2-fastino"
	case ProviderNative:
		return "gliner-2-native"
	default:
		return "gliner-2"
	}
}

func (c *Client) Chat(ctx context.Context, messages []types.Message) (*types.Response, error) {
	return nil, fmt.Errorf("GLInER2 does not support general Chat interface; use specific methods like ExtractEntities")
}

func (c *Client) ChatWithStructuredOutput(ctx context.Context, messages []types.Message, schema any) (*types.Response, error) {
	return nil, fmt.Errorf("GLInER2 does not support ChatWithStructuredOutput; use specific methods")
}

// ExtractEntities provides direct access to entity extraction
func (c *Client) ExtractEntities(ctx context.Context, text string, entityTypes []string) ([]nlp.ExtractedEntity, error) {
	switch c.provider {
	case ProviderNative:
		// TODO: specific mapping for native client if types differ
		// Assuming native client returns something standard we can map
		// Return type mismatch needs handling if nativeClient.ExtractEntitiesDirect returns local Entity type
		// For now, let's error or assume we map it here.
		// Since nativeClient is "future", let's focus on HTTP part which is verified.
		return nil, fmt.Errorf("native provider not yet implemented for standard interface")
	case ProviderLocal, ProviderFastino:
		if c.httpClient == nil {
			return nil, fmt.Errorf("HTTP client not available")
		}

		result, err := c.httpClient.ExtractEntities(ctx, text, entityTypes, 0.5)
		if err != nil {
			return nil, err
		}

		var entities []nlp.ExtractedEntity
		for label, entityList := range result.Entities {
			for _, entity := range entityList {
				entities = append(entities, nlp.ExtractedEntity{
					Text:       entity.Text,
					Label:      label,
					Confidence: entity.Confidence,
					Start:      entity.Start,
					End:        entity.End,
				})
			}
		}
		return entities, nil
	default:
		return nil, fmt.Errorf("unsupported provider: %v", c.provider)
	}
}

// ExtractRelations provides direct access to relationship extraction (GLInER2 relations)
func (c *Client) ExtractRelations(ctx context.Context, text string, relationTypes []string) ([]nlp.ExtractedRelation, error) {
	switch c.provider {
	case ProviderNative:
		return nil, fmt.Errorf("native provider not yet implemented for standard interface")
	case ProviderLocal, ProviderFastino:
		if c.httpClient == nil {
			return nil, fmt.Errorf("HTTP client not available")
		}

		schema := relationTypes
		// Check if we have ExtractRelations on httpClient or if we reuse ExtractFacts logic
		// http_client.go has ExtractRelations.
		// But it returns *RelationResult.
		// We need to map to []nlp.ExtractedRelation.

		// Let's use ExtractFacts logic from `http_client.go` effectively or call it directly?
		// Logic in `handleFactExtraction` calls `c.ExtractFacts`.
		// `ExtractFacts` in `client.go` (OLD) called `httpClient.ExtractFacts`.
		// Let's replace `ExtractFacts` with `ExtractRelations` implementing the interface.

		rels, err := c.httpClient.ExtractRelations(ctx, text, schema, 0.5)
		if err != nil {
			return nil, err
		}

		var relations []nlp.ExtractedRelation
		// RelationResult struct needs checking. From http_client.go:
		// type RelationResult struct { RelationExtraction map[string][]RelationTuple ... }
		// type RelationTuple struct { Head, Tail string ... }

		for relType, tuples := range rels.RelationExtraction {
			for _, tuple := range tuples {
				relations = append(relations, nlp.ExtractedRelation{
					Source:     tuple.Head.Text,
					Target:     tuple.Tail.Text,
					Type:       relType,
					Confidence: 1.0, // GLInER2 doesn't always provide confidence per relation yet
				})
			}
		}
		return relations, nil
	default:
		return nil, fmt.Errorf("unsupported provider: %v", c.provider)
	}
}

// Summarize is not supported by GLiNER2 models
func (c *Client) Summarize(ctx context.Context, text string) (string, error) {
	return "", fmt.Errorf("GLiNER2 does not support summarization")
}

// GenerateText is not supported by GLiNER2 models
func (c *Client) GenerateText(ctx context.Context, prompt string) (string, error) {
	return "", fmt.Errorf("GLiNER2 does not support text generation")
}

// ClassifyText provides direct access to text classification
func (c *Client) ClassifyText(ctx context.Context, text string, schema interface{}, threshold float64) (*ClassificationResult, error) {
	switch c.provider {
	case ProviderNative:
		return c.nativeClient.ClassifyText(ctx, text, schema, threshold)
	case ProviderLocal, ProviderFastino:
		if c.httpClient == nil {
			return nil, fmt.Errorf("HTTP client not available")
		}
		return c.httpClient.ClassifyText(ctx, text, schema, threshold)
	default:
		return nil, fmt.Errorf("unsupported provider: %v", c.provider)
	}
}

// Methods handleNodeExtraction and handleFactExtraction removed as they are no longer used.
// Direct extraction via ExtractEntities and ExtractRelations is now enforced.

func (c *Client) handleTextClassification(ctx context.Context, userMsg string) (*types.Response, error) {
	// Extract schema from the system message
	schema := extractClassificationSchema(userMsg)

	// Extract text content
	text := extractTextContent(userMsg)

	// Run text classification
	result, err := c.ClassifyText(ctx, text, schema, 0.5)
	if err != nil {
		return nil, fmt.Errorf("GLInER2 text classification failed: %w", err)
	}

	// Format as JSON response for Predicato
	content, err := formatClassificationResult(result)
	if err != nil {
		return nil, fmt.Errorf("failed to format classification result: %w", err)
	}

	return &types.Response{
		Content: content,
	}, nil
}
