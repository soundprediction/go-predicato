package nlp_test

import (
	"context"
	"testing"

	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/types"
)

// Example usage of GenerateJSONResponseWithContinuation
func ExampleGenerateJSONResponseWithContinuation() {
	// This example shows how to use the function (requires actual LLM client)

	// Define your expected structure
	type PregnancyTips struct {
		Category string   `json:"category"`
		Tips     []string `json:"tips"`
	}
	_ = PregnancyTips{} // Example type for documentation

	// Create LLM client (example - replace with actual client)
	// nlProcessor, _ := nlp.NewOpenAIClient("your-api-key", nlp.Config{
	// 	Model: "gpt-4",
	// })

	// Use the function
	// var result PregnancyTips
	// jsonStr, err := GenerateJSONResponseWithContinuation(
	// 	context.Background(),
	// 	nlProcessor,
	// 	"You are a helpful assistant that returns only valid JSON.",
	// 	"Generate a JSON object with pregnancy nutrition tips",
	// 	&result,
	// 	5, // max retries
	// )
	//
	// if err != nil {
	// 	// Handle error - may have partial JSON
	// 	log.Printf("Error: %v, Partial JSON: %s", err, jsonStr)
	// } else {
	// 	// Success - result is populated with valid data
	// 	log.Printf("Success: %+v", result)
	// }
}

// Example showing simple JSON generation without struct validation
func ExampleGenerateJSONWithContinuation() {
	// This is simpler - just ensures valid JSON without schema validation

	// jsonStr, err := GenerateJSONWithContinuation(
	// 	context.Background(),
	// 	nlProcessor,
	// 	"Return only valid JSON.",
	// 	"List 5 pregnancy exercises as JSON array",
	// 	3,
	// )
	//
	// if err != nil {
	// 	log.Printf("Failed: %v", err)
	// } else {
	// 	// Parse the JSON as needed
	// 	var exercises []string
	// 	json.Unmarshal([]byte(jsonStr), &exercises)
	// }
}

func TestExtractJSONFromResponse(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "JSON in markdown code block",
			input:    "```json\n{\"name\": \"test\"}\n```",
			expected: "{\"name\": \"test\"}",
		},
		{
			name:     "JSON in generic code block",
			input:    "```\n{\"name\": \"test\"}\n```",
			expected: "{\"name\": \"test\"}",
		},
		{
			name:     "JSON object with surrounding text",
			input:    "Here is the result: {\"name\": \"test\"} Hope this helps!",
			expected: "{\"name\": \"test\"}",
		},
		{
			name:     "JSON array with surrounding text",
			input:    "The items are: [\"item1\", \"item2\"] as requested.",
			expected: "[\"item1\", \"item2\"]",
		},
		{
			name:     "Plain JSON object",
			input:    "{\"name\": \"test\"}",
			expected: "{\"name\": \"test\"}",
		},
		{
			name:     "Plain JSON array",
			input:    "[1, 2, 3]",
			expected: "[1, 2, 3]",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := nlp.ExtractJSONFromResponse(tt.input)
			if result != tt.expected {
				t.Errorf("ExtractJSONFromResponse() = %v, want %v", result, tt.expected)
			}
		})
	}
}

// Mock LLM client for testing
type mockLLMClient struct {
	responses []string
	callCount int
}

func (m *mockLLMClient) Chat(ctx context.Context, messages []types.Message) (*types.Response, error) {
	if m.callCount >= len(m.responses) {
		return &types.Response{Content: m.responses[len(m.responses)-1]}, nil
	}
	response := m.responses[m.callCount]
	m.callCount++
	return &types.Response{Content: response}, nil
}

func (m *mockLLMClient) ChatWithStructuredOutput(ctx context.Context, messages []types.Message, schema any) (*types.Response, error) {
	// Not used in these tests
	return nil, nil
}

func (m *mockLLMClient) Close() error {
	// Nothing to close in mock
	return nil
}

func (m *mockLLMClient) GetCapabilities() []nlp.TaskCapability {
	return []nlp.TaskCapability{nlp.TaskTextGeneration}
}

func TestGenerateJSONWithContinuation_Success(t *testing.T) {
	// Test successful JSON generation on first try
	mockClient := &mockLLMClient{
		responses: []string{
			`{"name": "Test", "value": 123}`,
		},
	}

	jsonStr, err := nlp.GenerateJSONWithContinuation(
		context.Background(),
		mockClient,
		"Return JSON",
		"Generate test data",
		3,
	)

	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}

	if jsonStr == "" {
		t.Error("Expected non-empty JSON string")
	}
}

func TestGenerateJSONWithContinuation_Continuation(t *testing.T) {
	// Test continuation when first response is incomplete
	mockClient := &mockLLMClient{
		responses: []string{
			`{"name": "Test", "items": [`,
			`"item1", "item2"]}`,
		},
	}

	jsonStr, err := nlp.GenerateJSONWithContinuation(
		context.Background(),
		mockClient,
		"Return JSON",
		"Generate test data",
		3,
	)

	if err != nil {
		t.Errorf("Expected no error after continuation, got: %v", err)
	}

	if jsonStr == "" {
		t.Error("Expected non-empty JSON string")
	}

	// Verify it contains both parts
	if !contains(jsonStr, "name") || !contains(jsonStr, "items") {
		t.Error("Expected JSON to contain combined response")
	}
}

func TestGenerateJSONResponseWithContinuation_StructValidation(t *testing.T) {
	type TestStruct struct {
		Name  string `json:"name"`
		Value int    `json:"value"`
	}

	mockClient := &mockLLMClient{
		responses: []string{
			`{"name": "Test", "value": 123}`,
		},
	}

	var result TestStruct
	jsonStr, err := nlp.GenerateJSONResponseWithContinuation(
		context.Background(),
		mockClient,
		"Return JSON",
		"Generate test data",
		&result,
		3,
	)

	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}

	if result.Name != "Test" || result.Value != 123 {
		t.Errorf("Expected struct to be populated correctly, got: %+v", result)
	}

	if jsonStr == "" {
		t.Error("Expected non-empty JSON string")
	}
}

func TestGenerateJSONResponseWithContinuationMessages_Success(t *testing.T) {
	type TestStruct struct {
		Name  string `json:"name"`
		Value int    `json:"value"`
	}

	mockClient := &mockLLMClient{
		responses: []string{
			`{"name": "Test", "value": 123}`,
		},
	}

	// Build messages manually
	messages := []types.Message{
		{Role: "system", Content: "Return JSON"},
		{Role: "user", Content: "Generate test data"},
	}

	var result TestStruct
	jsonStr, err := nlp.GenerateJSONResponseWithContinuationMessages(
		context.Background(),
		mockClient,
		messages,
		&result,
		3,
	)

	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}

	if result.Name != "Test" || result.Value != 123 {
		t.Errorf("Expected struct to be populated correctly, got: %+v", result)
	}

	if jsonStr == "" {
		t.Error("Expected non-empty JSON string")
	}
}

func TestGenerateJSONResponseWithContinuationMessages_WithHistory(t *testing.T) {
	type TestStruct struct {
		Items []string `json:"items"`
	}

	mockClient := &mockLLMClient{
		responses: []string{
			`{"items": ["item1", "item2", "item3"]}`,
		},
	}

	// Build messages with conversation history
	messages := []types.Message{
		{Role: "system", Content: "You are a helpful assistant"},
		{Role: "user", Content: "What are some items?"},
		{Role: "assistant", Content: "I can provide items in JSON format."},
		{Role: "user", Content: "Please provide them as JSON"},
	}

	var result TestStruct
	jsonStr, err := nlp.GenerateJSONResponseWithContinuationMessages(
		context.Background(),
		mockClient,
		messages,
		&result,
		3,
	)

	if err != nil {
		t.Errorf("Expected no error, got: %v", err)
	}

	if len(result.Items) != 3 {
		t.Errorf("Expected 3 items, got: %d", len(result.Items))
	}

	if jsonStr == "" {
		t.Error("Expected non-empty JSON string")
	}
}

func contains(s, substr string) bool {
	return len(s) > 0 && len(substr) > 0 && (s == substr || len(s) > len(substr) && (s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || containsMiddle(s, substr)))
}

func TestOpenAIGenericClient_LegacySignature(t *testing.T) {
	// Test that the legacy signature still works
	apiKey := "test-key"
	temp := float32(0.7)
	config := nlp.Config{
		Model:       "gpt-4o-mini",
		BaseURL:     "https://api.openai.com",
		Temperature: &temp,
	}

	client, err := nlp.NewOpenAIGenericClient(apiKey, config)
	if err != nil {
		t.Fatalf("Expected no error with legacy signature, got: %v", err)
	}

	if client == nil {
		t.Fatal("Expected non-nil client")
	}

	// Verify the client is properly configured
	if client.GetConfig().APIKey != apiKey {
		t.Errorf("Expected API key %s, got %s", apiKey, client.GetConfig().APIKey)
	}

	if client.GetConfig().Model != config.Model {
		t.Errorf("Expected model %s, got %s", config.Model, client.GetConfig().Model)
	}
}

func TestOpenAIGenericClient_NewSignature(t *testing.T) {
	// Test that the new signature works
	config := &nlp.LLMConfig{
		APIKey:      "test-key",
		Model:       "gpt-4o-mini",
		BaseURL:     "https://api.openai.com",
		Temperature: 0.7,
	}

	client, err := nlp.NewOpenAIGenericClient(config)
	if err != nil {
		t.Fatalf("Expected no error with new signature, got: %v", err)
	}

	if client == nil {
		t.Fatal("Expected non-nil client")
	}

	// Verify the client is properly configured
	if client.GetConfig().APIKey != config.APIKey {
		t.Errorf("Expected API key %s, got %s", config.APIKey, client.GetConfig().APIKey)
	}

	if client.GetConfig().Model != config.Model {
		t.Errorf("Expected model %s, got %s", config.Model, client.GetConfig().Model)
	}
}

func containsMiddle(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
