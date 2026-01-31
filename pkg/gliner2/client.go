package gliner2

import (
	"context"
	"fmt"
	"strings"

	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/types"
)

// Client provides unified access to GLInER2 functionality through different providers
type Client struct {
	provider   Provider
	httpClient *HTTPClient
	// Future: go-gline-rs GLInER2
	nativeClient *NativeClient // for go-gline-rs GLInER2 when available
	config       Config
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
					Source:     tuple.Head,
					Target:     tuple.Tail,
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

func (c *Client) handleNodeExtraction(ctx context.Context, userMsg string) (*types.Response, error) {
	entityTypesTSV := parseSection(userMsg, "ENTITY TYPES")
	text := parseSection(userMsg, "TEXT")
	if text == "" {
		text = parseSection(userMsg, "CURRENT MESSAGE")
	}
	if text == "" {
		text = parseSection(userMsg, "JSON")
	}

	// Parse Entity Types to map Name -> ID
	typesRecords := parseTSV(entityTypesTSV)
	labelToID := make(map[string]string)
	var labels []string

	// Skip header if present
	startIndex := 0
	if len(typesRecords) > 0 && (strings.Contains(typesRecords[0][0], "entity_type_id") || strings.Contains(typesRecords[0][1], "entity_type_name")) {
		startIndex = 1
	}

	for i := startIndex; i < len(typesRecords); i++ {
		row := typesRecords[i]
		if len(row) < 2 {
			continue
		}
		id := row[0]
		name := row[1]
		labelToID[name] = id
		labels = append(labels, name)
	}

	// Run entity extraction
	entities, err := c.ExtractEntities(ctx, text, labels)
	if err != nil {
		return nil, fmt.Errorf("GLInER2 node extraction failed: %w", err)
	}

	// Format Output as TSV (like existing GLInER adapter)
	var sb strings.Builder
	sb.WriteString("entity\tentity_type_id\n")

	for _, e := range entities {
		id, ok := labelToID[e.Label]
		if !ok {
			id = "-1"
		}
		sb.WriteString(fmt.Sprintf("%s\t%s\n", e.Text, id))
	}

	return &types.Response{
		Content: sb.String(),
	}, nil
}

func (c *Client) handleFactExtraction(ctx context.Context, userMsg string) (*types.Response, error) {
	factTypesTSV := parseSection(userMsg, "FACT TYPES")
	extractedEntitiesTSV := parseSection(userMsg, "ENTITIES")
	text := parseSection(userMsg, "CURRENT_MESSAGE")
	if text == "" {
		text = parseSection(userMsg, "CURRENT MESSAGE")
	}

	// Parse Entities to map Text -> ID
	entitiesRecords := parseTSV(extractedEntitiesTSV)
	nameToID := make(map[string]string)

	// Skip header
	startIndex := 0
	if len(entitiesRecords) > 0 && (strings.Contains(entitiesRecords[0][0], "id") || strings.Contains(entitiesRecords[0][1], "name")) {
		startIndex = 1
	}

	for i := startIndex; i < len(entitiesRecords); i++ {
		row := entitiesRecords[i]
		if len(row) < 2 {
			continue
		}
		id := row[0]
		name := row[1]
		nameToID[name] = id
	}

	// Parse Fact Types (Relation Types)
	factTypesRecords := parseTSV(factTypesTSV)
	var relationTypes []string

	startFact := 0
	if len(factTypesRecords) > 0 && strings.Contains(factTypesRecords[0][0], "relation_type") {
		startFact = 1
	}

	for i := startFact; i < len(factTypesRecords); i++ {
		row := factTypesRecords[i]
		if len(row) > 0 {
			relationTypes = append(relationTypes, row[0])
		}
	}

	// Run fact extraction using GLInER2 relations
	facts, err := c.ExtractRelations(ctx, text, relationTypes)
	if err != nil {
		return nil, fmt.Errorf("GLInER2 fact extraction failed: %w", err)
	}

	// Format Output as TSV (like existing GLInER adapter)
	var sb strings.Builder
	sb.WriteString("source_id\trelation_type\ttarget_id\tfact\tsummary\tvalid_at\tinvalid_at\n")

	for _, f := range facts {
		srcID, okSrc := nameToID[f.Source]
		tgtID, okTgt := nameToID[f.Target]

		// Only emit if we can map back to IDs
		if okSrc && okTgt {
			fact := fmt.Sprintf("%s %s %s", f.Source, f.Type, f.Target)
			sb.WriteString(fmt.Sprintf("%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
				srcID, f.Type, tgtID, fact, "", "null", "null"))
		}
	}

	return &types.Response{
		Content: sb.String(),
	}, nil
}

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
