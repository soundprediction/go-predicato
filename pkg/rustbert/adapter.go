package rustbert

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/types"
)

// LLMAdapter adapts the RustBert client to the nlp.Client interface.
type LLMAdapter struct {
	client *Client
	task   string // "summarization", "text_generation", "ner", "qa"
}

// NewLLMAdapter creates a new LLM adapter for RustBert.
func NewLLMAdapter(client *Client, task string) *LLMAdapter {
	return &LLMAdapter{
		client: client,
		task:   task,
	}
}

// Chat implements the nlp.Client interface.
func (a *LLMAdapter) Chat(ctx context.Context, messages []types.Message) (*types.Response, error) {
	// Simple concatenation of user messages for now
	var inputBuilder strings.Builder
	for _, msg := range messages {
		if msg.Role == nlp.RoleUser {
			if inputBuilder.Len() > 0 {
				inputBuilder.WriteString("\n")
			}
			inputBuilder.WriteString(msg.Content)
		}
	}
	input := inputBuilder.String()

	var output string
	var err error

	switch a.task {
	case "summarization":
		output, err = a.client.Summarize(ctx, input)
		if err != nil {
			return nil, err
		}
	case "text_generation", "generation":
		output, err = a.client.GenerateText(ctx, input)
		if err != nil {
			return nil, err
		}
	case "ner":
		entities, err := a.client.ExtractEntities(ctx, input, nil)
		if err != nil {
			return nil, err
		}
		// Serialize to JSON for standard return
		b, err := json.Marshal(entities)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal NER entities: %w", err)
		}
		output = string(b)
	case "qa":
		// Naive protocol: Input format "Question: <q>\nContext: <ctx>"
		// Or try to parse last message?
		// For now, let's assume input is JUST the context, and we can't do QA easily without structured input.
		// Or assume json input?
		return nil, fmt.Errorf("QA task not fully implemented in chat interface yet")
	default:
		return nil, fmt.Errorf("unknown task: %s", a.task)
	}

	return &types.Response{
		Content: output,
	}, nil
}

// ChatWithStructuredOutput implements the nlp.Client interface.
func (a *LLMAdapter) ChatWithStructuredOutput(ctx context.Context, messages []types.Message, schema any) (*types.Response, error) {
	// RustBert models don't support structured output schema enforcement natively yet.
	// We just call chat and hope for the best (or the task inherently returns structure like NER).
	return a.Chat(ctx, messages)
}

// Close implements the nlp.Client interface.
func (a *LLMAdapter) Close() error {
	// The underlying client is shared, so we generally don't close it here
	// unless we own it. Application logic typically manages the rustbert.Client lifecycle.
	return nil
}

// ExtractEntities provides direct access to entity extraction
func (a *LLMAdapter) ExtractEntities(ctx context.Context, text string, entityTypes []string) ([]nlp.ExtractedEntity, error) {
	if a.task != "ner" {
		return nil, fmt.Errorf("task mismatch: adapter configured for %s, requested ExtractEntities", a.task)
	}

	return a.client.ExtractEntities(ctx, text, entityTypes)
}

// ExtractRelations provides direct access to relationship extraction
func (a *LLMAdapter) ExtractRelations(ctx context.Context, text string, relationTypes []string) ([]nlp.ExtractedRelation, error) {
	return nil, fmt.Errorf("RustBert does not support relationship extraction")
}

// Summarize provides direct access to summarization
func (a *LLMAdapter) Summarize(ctx context.Context, text string) (string, error) {
	if a.task != "summarization" {
		return "", fmt.Errorf("task mismatch: adapter configured for %s, requested Summarize", a.task)
	}

	return a.client.Summarize(ctx, text)
}

// GenerateText provides direct access to text generation
func (a *LLMAdapter) GenerateText(ctx context.Context, prompt string) (string, error) {
	if a.task != "text_generation" && a.task != "generation" {
		return "", fmt.Errorf("task mismatch: adapter configured for %s, requested GenerateText", a.task)
	}

	return a.client.GenerateText(ctx, prompt)
}

// GetCapabilities returns the list of capabilities supported by this client.
func (a *LLMAdapter) GetCapabilities() []nlp.TaskCapability {
	switch a.task {
	case "summarization":
		return []nlp.TaskCapability{nlp.TaskSummarization}
	case "text_generation", "generation":
		return []nlp.TaskCapability{nlp.TaskTextGeneration}
	case "ner":
		return []nlp.TaskCapability{nlp.TaskNamedEntityRecognition}
	case "qa":
		return []nlp.TaskCapability{nlp.TaskQuestionAnswering}
	default:
		return []nlp.TaskCapability{}
	}
}

// ExtractExtended implements the nlp.Client interface.
func (a *LLMAdapter) ExtractExtended(ctx context.Context, text string, entityTypes, relationTypes []string) (*nlp.ExtendedExtractionResult, error) {
	return nil, fmt.Errorf("RustBert does not support extended extraction")
}

// GetModel returns the model identifier.
func (a *LLMAdapter) GetModel() string {
	return "rust-bert-" + a.task
}
