package gliner2

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type HTTPClient struct {
	httpClient *http.Client
	baseURL    string
	apiKey     string
	timeout    time.Duration
}

type LocalConfig struct {
	Endpoint string        `json:"endpoint"`
	Timeout  time.Duration `json:"timeout"`
}

type FastinoConfig struct {
	Endpoint string        `json:"endpoint"`
	APIKey   string        `json:"api_key"`
	Timeout  time.Duration `json:"timeout"`
}

// Future: NativeConfig for go-gline-rs GLInER2
// type NativeConfig struct {
// 	ModelPath string `json:"model_path"`
// }

type Config struct {
	Local   *LocalConfig   `json:"local,omitempty"`
	Fastino *FastinoConfig `json:"fastino,omitempty"`
	// Future:
	// Native *NativeConfig `json:"native,omitempty"`
	Provider Provider `json:"provider"`
}

func NewHTTPClient(config Config) (*HTTPClient, error) {
	var baseURL, apiKey string
	var timeout time.Duration

	switch config.Provider {
	case ProviderLocal:
		if config.Local == nil {
			return nil, fmt.Errorf("local config required for local provider")
		}
		baseURL = config.Local.Endpoint
		timeout = config.Local.Timeout
	case ProviderFastino:
		if config.Fastino == nil {
			return nil, fmt.Errorf("fastino config required for fastino provider")
		}
		baseURL = config.Fastino.Endpoint
		apiKey = config.Fastino.APIKey
		timeout = config.Fastino.Timeout
	default:
		return nil, fmt.Errorf("unsupported provider: %v", config.Provider)
	}

	if baseURL == "" {
		return nil, fmt.Errorf("endpoint URL is required")
	}

	client := &HTTPClient{
		baseURL: baseURL,
		apiKey:  apiKey,
		timeout: timeout,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}

	return client, nil
}

func (c *HTTPClient) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/health", nil)
	if err != nil {
		return fmt.Errorf("failed to create health request: %w", err)
	}

	// Add API key header for Fastino
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed with status: %d", resp.StatusCode)
	}

	return nil
}

func (c *HTTPClient) ExtractEntities(ctx context.Context, text string, schema interface{}, threshold float64) (*EntityResult, error) {
	request := ExtractRequest{
		Task:      "extract_entities",
		Text:      text,
		Schema:    schema,
		Threshold: threshold,
	}

	var result EntityResult
	err := c.makeRequest(ctx, request, &result)
	if err != nil {
		return nil, fmt.Errorf("entity extraction failed: %w", err)
	}

	return &result, nil
}

func (c *HTTPClient) ExtractRelations(ctx context.Context, text string, schema interface{}, threshold float64) (*RelationResult, error) {
	request := ExtractRequest{
		Task:      "extract_relations", // GLInER2 uses extract_relations
		Text:      text,
		Schema:    schema,
		Threshold: threshold,
	}

	var result RelationResult
	err := c.makeRequest(ctx, request, &result)
	if err != nil {
		return nil, fmt.Errorf("relation extraction failed: %w", err)
	}

	return &result, nil
}

func (c *HTTPClient) ExtractFacts(ctx context.Context, text string, schema interface{}, threshold float64) ([]Fact, error) {
	// GLInER2 uses relation extraction for facts
	relations, err := c.ExtractRelations(ctx, text, schema, threshold)
	if err != nil {
		return nil, err
	}

	// Convert GLInER2 relations to Predicato facts
	var facts []Fact
	for relationType, tuples := range relations.RelationExtraction {
		for _, tuple := range tuples {
			fact := Fact{
				Source: tuple.Head.Text,
				Target: tuple.Tail.Text,
				Type:   relationType,
				SourceSpan: &Span{
					Text:  tuple.Head.Text,
					Start: tuple.Head.Start,
					End:   tuple.Head.End,
				},
				TargetSpan: &Span{
					Text:  tuple.Tail.Text,
					Start: tuple.Tail.Start,
					End:   tuple.Tail.End,
				},
				Confidence: 1.0, // GLInER2 doesn't provide confidence in tuple format
			}
			facts = append(facts, fact)
		}
	}

	return facts, nil
}

func (c *HTTPClient) ClassifyText(ctx context.Context, text string, schema interface{}, threshold float64) (*ClassificationResult, error) {
	request := ExtractRequest{
		Task:      "classify_text",
		Text:      text,
		Schema:    schema,
		Threshold: threshold,
	}

	var result ClassificationResult
	err := c.makeRequest(ctx, request, &result)
	if err != nil {
		return nil, fmt.Errorf("text classification failed: %w", err)
	}

	return &result, nil
}

func (c *HTTPClient) ExtractStructured(ctx context.Context, text string, schema interface{}, threshold float64) (*StructuredResult, error) {
	request := ExtractRequest{
		Task:      "extract_json",
		Text:      text,
		Schema:    schema,
		Threshold: threshold,
	}

	var result StructuredResult
	err := c.makeRequest(ctx, request, &result)
	if err != nil {
		return nil, fmt.Errorf("structured extraction failed: %w", err)
	}

	return &result, nil
}

// ExtractExtended performs combined entity, relation, context, and rule extraction
// via the GLiNER2 server's two-pass pipeline.
//
// The server performs extraction in two internal passes:
//   - Pass 1: Entities + Relations using enriched type schemas (87 entity types,
//     44 relation types) via model.extract()
//   - Pass 2: Context metadata + Rules via model.extract_json() with a lightweight
//     structure schema
//
// Two passes are needed because GLiNER2's model produces zero structure results
// when entities, relations, and structures are all in the same schema — the
// model's attention is consumed by entity/relation extraction. The server merges
// results from both passes into a single response.
//
// entityTypes and relationTypes are accepted for interface compatibility but are
// not sent to the server; the server uses its own comprehensive enriched types
// that are a superset of graph-config.toml types.
func (c *HTTPClient) ExtractExtended(ctx context.Context, text string, _, _ []string) (*ExtendedExtractionResult, error) {
	// Signal extended extraction to the server. The server detects the
	// "entities" key and triggers its two-pass pipeline with enriched
	// entity/relation types and context/rule structure schemas.
	schema := map[string]interface{}{
		"entities":  true,
		"relations": true,
	}

	request := ExtractRequest{
		Task:      "extract_json",
		Text:      text,
		Schema:    schema,
		Threshold: 0.3,
	}

	// The raw response from extract_json is a StructuredResult with keys like
	// "entities", "relation_extraction", "fact", "rule".
	var rawResult map[string]interface{}
	err := c.makeRequest(ctx, request, &rawResult)
	if err != nil {
		return nil, fmt.Errorf("extended extraction failed: %w", err)
	}

	// Parse the combined response into ExtendedExtractionResult
	result := &ExtendedExtractionResult{
		SourceText: text,
		Entities:   make(map[string][]string),
	}

	// Parse entities: {"entities": {"condition": [{"text": "...", ...}], ...}}
	if entitiesRaw, ok := rawResult["entities"]; ok {
		if entitiesMap, ok := entitiesRaw.(map[string]interface{}); ok {
			for entityType, entityListRaw := range entitiesMap {
				if entityList, ok := entityListRaw.([]interface{}); ok {
					var names []string
					for _, e := range entityList {
						switch v := e.(type) {
						case string:
							names = append(names, v)
						case map[string]interface{}:
							if text, ok := v["text"].(string); ok {
								names = append(names, text)
							}
						}
					}
					if len(names) > 0 {
						result.Entities[entityType] = names
					}
				}
			}
		}
	}

	// Parse relations: {"relation_extraction": {"treats": [{"head": ..., "tail": ...}], ...}}
	if relRaw, ok := rawResult["relation_extraction"]; ok {
		if relMap, ok := relRaw.(map[string]interface{}); ok {
			for relType, tupleListRaw := range relMap {
				if tupleList, ok := tupleListRaw.([]interface{}); ok {
					for _, tupleRaw := range tupleList {
						if tuple, ok := tupleRaw.(map[string]interface{}); ok {
							rel := Relation{Relation: relType}
							rel.Head = parseSpanOrString(tuple["head"])
							rel.Tail = parseSpanOrString(tuple["tail"])
							if conf, ok := tuple["confidence"].(float64); ok {
								rel.Confidence = conf
							}
							result.Relations = append(result.Relations, rel)
						}
					}
				}
			}
		}
	}

	// Parse facts/triples: {"fact": [{"subject": "...", "predicate": "...", ...}]}
	if factsRaw, ok := rawResult["fact"]; ok {
		if factList, ok := factsRaw.([]interface{}); ok {
			for _, factRaw := range factList {
				if factMap, ok := factRaw.(map[string]interface{}); ok {
					triple := ExtendedTriple{
						Subject:           getStringField(factMap, "subject"),
						Predicate:         getStringField(factMap, "predicate"),
						Object:            getStringField(factMap, "object"),
						Condition:         getStringField(factMap, "condition"),
						Temporal:          getStringField(factMap, "temporal"),
						Location:          getStringField(factMap, "location"),
						Certainty:         getStringField(factMap, "certainty"),
						Scope:             getStringField(factMap, "scope"),
						SourceAttribution: getStringField(factMap, "source_attribution"),
					}
					if conf, ok := factMap["confidence"].(float64); ok {
						triple.Confidence = conf
					}
					result.Triples = append(result.Triples, triple)
				}
			}
		}
	}

	// Parse rules: {"rule": [{"antecedent": "...", "consequent": "...", ...}]}
	if rulesRaw, ok := rawResult["rule"]; ok {
		if ruleList, ok := rulesRaw.([]interface{}); ok {
			for _, ruleRaw := range ruleList {
				if ruleMap, ok := ruleRaw.(map[string]interface{}); ok {
					rule := Rule{
						Antecedent:        getStringField(ruleMap, "antecedent"),
						Consequent:        getStringField(ruleMap, "consequent"),
						Exception:         getStringField(ruleMap, "exception"),
						RuleType:          getStringField(ruleMap, "rule_type"),
						Scope:             getStringField(ruleMap, "scope"),
						SourceAttribution: getStringField(ruleMap, "source_attribution"),
					}
					if conf, ok := ruleMap["confidence"].(float64); ok {
						rule.Confidence = conf
					}
					result.Rules = append(result.Rules, rule)
				}
			}
		}
	}

	return result, nil
}

// getStringField safely extracts a string field from a map.
func getStringField(m map[string]interface{}, key string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// parseSpanOrString converts a raw interface{} to SpanOrString.
func parseSpanOrString(v interface{}) SpanOrString {
	switch val := v.(type) {
	case string:
		return SpanOrString{Text: val}
	case map[string]interface{}:
		s := SpanOrString{}
		if text, ok := val["text"].(string); ok {
			s.Text = text
		}
		if start, ok := val["start"].(float64); ok {
			s.Start = int(start)
		}
		if end, ok := val["end"].(float64); ok {
			s.End = int(end)
		}
		return s
	}
	return SpanOrString{}
}

func (c *HTTPClient) Close() error {
	// No explicit cleanup needed for HTTP client
	return nil
}

func (c *HTTPClient) makeRequest(ctx context.Context, request ExtractRequest, result interface{}) error {
	reqBody, err := json.Marshal(request)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/gliner-2", strings.NewReader(string(reqBody)))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	// Add API key header for Fastino
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var apiError struct {
			Detail string `json:"detail"`
		}
		_ = json.Unmarshal(body, &apiError)
		return fmt.Errorf("API error (status %d): %s", resp.StatusCode, apiError.Detail)
	}

	response := ExtractResponse{Result: result}
	return json.Unmarshal(body, &response)
}
