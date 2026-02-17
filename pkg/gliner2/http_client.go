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

// ExtractExtended extracts contextualized triples and rules from text using
// GLiNER2's structure extraction (extract_json).
//
// This only performs structure extraction — entities and relations are extracted
// separately by ExtractEntities/ExtractRelations in the ingestion pipeline
// (steps 4-6 of ExtractToFacts). Running entity+relation extraction here would
// be redundant and wasteful.
//
// entityTypes and relationTypes are accepted for interface compatibility but
// are not used.
func (c *HTTPClient) ExtractExtended(ctx context.Context, text string, _, _ []string) (*ExtendedExtractionResult, error) {
	// Structure-only schema: extract contextualized triples and rules.
	// Each field is "name::str" so the model predicts a single span per field
	// per instance, with the model's count predictor determining how many
	// instances to extract.
	schema := map[string]interface{}{
		"fact": []string{
			"subject::str",
			"predicate::str",
			"object::str",
			"condition::str",
			"temporal::str",
			"location::str",
			"certainty::str",
			"scope::str",
			"source_attribution::str",
		},
		"rule": []string{
			"antecedent::str",
			"consequent::str",
			"exception::str",
			"rule_type::str",
			"scope::str",
			"source_attribution::str",
		},
	}

	request := ExtractRequest{
		Task:      "extract_json",
		Text:      text,
		Schema:    schema,
		Threshold: 0.3,
	}

	var rawResult map[string]interface{}
	err := c.makeRequest(ctx, request, &rawResult)
	if err != nil {
		return nil, fmt.Errorf("extended extraction failed: %w", err)
	}

	result := &ExtendedExtractionResult{
		SourceText: text,
		Entities:   make(map[string][]string),
	}

	// Parse facts/triples: {"fact": [{"subject": "...", "predicate": "...", ...}]}
	// Field values may be plain strings or dicts ({"text": "...", "confidence": ...})
	// depending on server include_confidence/include_spans settings.
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
						Confidence:        getFieldConfidence(factMap),
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
						Confidence:        getFieldConfidence(ruleMap),
					}
					result.Rules = append(result.Rules, rule)
				}
			}
		}
	}

	return result, nil
}

// getStringField safely extracts a string field from a map.
// Handles both plain string values and dict values ({"text": "...", "confidence": ...})
// that GLiNER2 returns when include_confidence or include_spans is enabled.
func getStringField(m map[string]interface{}, key string) string {
	v, ok := m[key]
	if !ok {
		return ""
	}
	switch val := v.(type) {
	case string:
		return val
	case map[string]interface{}:
		if text, ok := val["text"].(string); ok {
			return text
		}
	}
	return ""
}

// getFieldConfidence extracts an aggregate confidence score from a structure instance.
// Checks for a top-level "confidence" key first, then averages per-field confidence
// values from dict-valued fields (include_confidence mode).
func getFieldConfidence(m map[string]interface{}) float64 {
	if conf, ok := m["confidence"].(float64); ok {
		return conf
	}
	var total float64
	var count int
	for _, v := range m {
		if d, ok := v.(map[string]interface{}); ok {
			if conf, ok := d["confidence"].(float64); ok {
				total += conf
				count++
			}
		}
	}
	if count > 0 {
		return total / float64(count)
	}
	return 0
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
