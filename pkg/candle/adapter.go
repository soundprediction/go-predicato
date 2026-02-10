//go:build cgo

package candle

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/types"
)

// LLMAdapter adapts the candle Client to the nlp.Client interface,
// routing Chat messages to task-specific methods.
type LLMAdapter struct {
	client *Client
	task   string // "summarization", "text_generation", "ner", "translation"
}

// NewLLMAdapter creates a new LLM adapter for the candle client.
func NewLLMAdapter(client *Client, task string) *LLMAdapter {
	return &LLMAdapter{
		client: client,
		task:   task,
	}
}

// Chat implements nlp.Client by routing to task-specific methods.
func (a *LLMAdapter) Chat(ctx context.Context, messages []types.Message) (*types.Response, error) {
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
	case "text_generation", "generation":
		output, err = a.client.GenerateText(ctx, input)
	case "ner":
		entities, nerErr := a.client.ExtractEntities(ctx, input, nil)
		if nerErr != nil {
			return nil, nerErr
		}
		b, marshalErr := json.Marshal(entities)
		if marshalErr != nil {
			return nil, fmt.Errorf("failed to marshal NER entities: %w", marshalErr)
		}
		output = string(b)
	default:
		return nil, fmt.Errorf("unknown task: %s", a.task)
	}

	if err != nil {
		return nil, err
	}
	return &types.Response{Content: output}, nil
}

// ChatWithStructuredOutput delegates to Chat.
func (a *LLMAdapter) ChatWithStructuredOutput(ctx context.Context, messages []types.Message, _ any) (*types.Response, error) {
	return a.Chat(ctx, messages)
}

// ExtractEntities delegates to the underlying client.
func (a *LLMAdapter) ExtractEntities(ctx context.Context, text string, entityTypes []string) ([]nlp.ExtractedEntity, error) {
	return a.client.ExtractEntities(ctx, text, entityTypes)
}

// ExtractRelations delegates to the underlying client.
func (a *LLMAdapter) ExtractRelations(ctx context.Context, text string, relationTypes []string) ([]nlp.ExtractedRelation, error) {
	return a.client.ExtractRelations(ctx, text, relationTypes)
}

// ExtractExtended delegates to the underlying client.
func (a *LLMAdapter) ExtractExtended(ctx context.Context, text string, entityTypes, relationTypes []string) (*nlp.ExtendedExtractionResult, error) {
	return a.client.ExtractExtended(ctx, text, entityTypes, relationTypes)
}

// Summarize delegates to the underlying client.
func (a *LLMAdapter) Summarize(ctx context.Context, text string) (string, error) {
	return a.client.Summarize(ctx, text)
}

// GenerateText delegates to the underlying client.
func (a *LLMAdapter) GenerateText(ctx context.Context, prompt string) (string, error) {
	return a.client.GenerateText(ctx, prompt)
}

// GetModel returns the model identifier with task suffix.
func (a *LLMAdapter) GetModel() string {
	return "candle-" + a.task
}

// GetCapabilities returns task-specific capabilities.
func (a *LLMAdapter) GetCapabilities() []nlp.TaskCapability {
	switch a.task {
	case "summarization":
		return []nlp.TaskCapability{nlp.TaskSummarization}
	case "text_generation", "generation":
		return []nlp.TaskCapability{nlp.TaskTextGeneration}
	case "ner":
		return []nlp.TaskCapability{nlp.TaskNamedEntityRecognition}
	case "translation":
		return []nlp.TaskCapability{nlp.TaskTranslation}
	default:
		return []nlp.TaskCapability{}
	}
}

// Close is a no-op; the underlying Client lifecycle is managed externally.
func (a *LLMAdapter) Close() error {
	return nil
}
