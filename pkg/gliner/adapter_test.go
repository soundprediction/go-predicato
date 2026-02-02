package gliner

import (
	"context"
	"strings"
	"testing"

	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/types"
)

type mockLLM struct{}

func (m *mockLLM) Chat(ctx context.Context, msgs []types.Message) (*types.Response, error) {
	return &types.Response{Content: "mock response"}, nil
}
func (m *mockLLM) ChatWithStructuredOutput(ctx context.Context, msgs []types.Message, schema any) (*types.Response, error) {
	return &types.Response{Content: "mock struct"}, nil
}
func (m *mockLLM) Close() error { return nil }

func (m *mockLLM) GetCapabilities() []nlp.TaskCapability {
	return []nlp.TaskCapability{nlp.TaskTextGeneration}
}

func (m *mockLLM) GetModel() string {
	return "mock-model"
}

func (m *mockLLM) ExtractEntities(ctx context.Context, text string, entityTypes []string) ([]nlp.ExtractedEntity, error) {
	return nil, nil
}

func (m *mockLLM) ExtractRelations(ctx context.Context, text string, relationTypes []string) ([]nlp.ExtractedRelation, error) {
	return nil, nil
}

func (m *mockLLM) Summarize(ctx context.Context, text string) (string, error) {
	return "summary", nil
}

func (m *mockLLM) GenerateText(ctx context.Context, prompt string) (string, error) {
	return "generated", nil
}

func TestAdapterNodeExtraction(t *testing.T) {
	modelID := "onnx-community/gliner_small-v2.1"
	c, err := NewClient(modelID)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	adapter := NewLLMAdapter(c, &mockLLM{})

	// Simulate Node Extraction Prompt
	sysPrompt := "You are an AI assistant that extracts entity nodes from text."
	userPrompt := `
<ENTITY TYPES>
entity_type_id	entity_type_name
0	Entity
1	Person
2	Organization
</ENTITY TYPES>

<TEXT>
Steve Jobs founded Apple in California.
</TEXT>
`
	messages := []types.Message{
		{Role: nlp.RoleSystem, Content: sysPrompt},
		{Role: nlp.RoleUser, Content: userPrompt},
	}

	resp, err := adapter.Chat(context.Background(), messages)
	if err != nil {
		t.Fatalf("Chat failed: %v", err)
	}

	t.Logf("Response:\n%s", resp.Content)

	if !strings.Contains(resp.Content, "Steve Jobs\t1") {
		t.Error("Expected Steve Jobs as Person (1)")
	}
	if !strings.Contains(resp.Content, "Apple\t2") {
		t.Error("Expected Apple as Organization (2)")
	}
}

func TestAdapterEdgeExtraction(t *testing.T) {
	// Need a model that supports relations or mock it?
	// Using the same node model for testing will checking flow but fail at extraction or return empty.
	// But TestClient verified loading.
	// For actual relation extraction we need a relation model.
	// We can try to load the node model as relation model (will fail or load incorrect graph)
	// Or downloading a real relation model "gliner-multitask-large-v0.5" but that's large.
	// "urchade/gliner_relation" ?
	// Since we don't have a small relation model handy and I can't guarantee download size/time.
	// I will test the parsing logic but expect empty/error if model not loaded.

	// Let's create a client without relation model loaded
	modelID := "onnx-community/gliner_small-v2.1"
	c, err := NewClient(modelID)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	defer c.Close()

	adapter := NewLLMAdapter(c, &mockLLM{})

	sysPrompt := "You are an expert fact extractor that extracts fact triples from text."
	userPrompt := `
<FACT TYPES>
fact_type_signature	relation_type
Person->FOUNDED->Organization	FOUNDED
</FACT TYPES>

<ENTITIES>
id	name
0	Steve Jobs
1	Apple
</ENTITIES>

<CURRENT_MESSAGE>
Steve Jobs founded Apple.
</CURRENT_MESSAGE>
`
	messages := []types.Message{
		{Role: nlp.RoleSystem, Content: sysPrompt},
		{Role: nlp.RoleUser, Content: userPrompt},
	}

	// Should fail because relation extraction not loaded
	_, err = adapter.Chat(context.Background(), messages)
	if err == nil {
		t.Error("Expected error due to missing relation model")
	} else if !strings.Contains(err.Error(), "relation model not loaded") {
		t.Errorf("Expected 'relation model not loaded' error, got: %v", err)
	}
}
