package nlp

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/parquet-go/parquet-go"
	"github.com/soundprediction/predicato/pkg/cost"
	"github.com/soundprediction/predicato/pkg/types"
)

// TokenUsageRecord represents a single log entry for token usage
type TokenUsageRecord struct {
	Timestamp        time.Time `parquet:"timestamp"`
	ID               string    `parquet:"id"`
	Model            string    `parquet:"model"`
	UserID           string    `parquet:"user_id"`
	SessionID        string    `parquet:"session_id"`
	RequestSource    string    `parquet:"request_source"`
	IngestionSource  string    `parquet:"ingestion_source"`
	TotalTokens      int       `parquet:"total_tokens"`
	PromptTokens     int       `parquet:"prompt_tokens"`
	CompletionTokens int       `parquet:"completion_tokens"`
	EstimatedCost    float64   `parquet:"estimated_cost"`
	IsSystemCall     bool      `parquet:"is_system_call"`
}

// ParquetTokenTracker handles persistence of token usage stats to Parquet files
type ParquetTokenTracker struct {
	costCalculator *cost.CostCalculator
	outputDir      string
	buffer         []TokenUsageRecord
	batchSize      int
	mu             sync.Mutex
}

// NewTokenTracker creates a new token tracker writing to a directory
func NewTokenTracker(outputDir string) (*ParquetTokenTracker, error) {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create token tracking directory: %w", err)
	}

	tracker := &ParquetTokenTracker{
		outputDir:      outputDir,
		costCalculator: cost.NewCostCalculator(),
		buffer:         make([]TokenUsageRecord, 0, 100),
		batchSize:      100,
	}

	return tracker, nil
}

// AddUsage adds usage to the tracker
func (t *ParquetTokenTracker) AddUsage(ctx context.Context, usage *types.TokenUsage, model string) error {
	if usage == nil {
		return nil
	}

	costUSD := t.costCalculator.CalculateCost(model, usage.PromptTokens, usage.CompletionTokens)

	record := TokenUsageRecord{
		ID:               uuid.New().String(),
		Timestamp:        time.Now().UTC(),
		Model:            model,
		TotalTokens:      usage.TotalTokens,
		PromptTokens:     usage.PromptTokens,
		CompletionTokens: usage.CompletionTokens,
		EstimatedCost:    costUSD,
	}

	// Extract context
	if v, ok := ctx.Value(types.ContextKeyUserID).(string); ok {
		record.UserID = v
	}
	if v, ok := ctx.Value(types.ContextKeySessionID).(string); ok {
		record.SessionID = v
	}
	if v, ok := ctx.Value(types.ContextKeyRequestSource).(string); ok {
		record.RequestSource = v
	}
	if v, ok := ctx.Value(types.ContextKeyIngestionSource).(string); ok {
		record.IngestionSource = v
	}
	if v, ok := ctx.Value(types.ContextKeySystemCall).(bool); ok {
		record.IsSystemCall = v
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	t.buffer = append(t.buffer, record)

	if len(t.buffer) >= t.batchSize {
		return t.flush()
	}

	return nil
}

// flush writes the current buffer to a new Parquet file
// Caller must hold the lock
func (t *ParquetTokenTracker) flush() error {
	if len(t.buffer) == 0 {
		return nil
	}

	filename := fmt.Sprintf("token_usage_%s_%d.parquet", time.Now().Format("20060102_150405"), time.Now().UnixNano())
	filepath := filepath.Join(t.outputDir, filename)

	err := parquet.WriteFile(filepath, t.buffer)
	if err != nil {
		fmt.Printf("Failed to write token usage parquet file: %v\n", err)
		return err
	}

	// Clear buffer
	t.buffer = t.buffer[:0]
	return nil
}

// TokenTrackingClient wraps a Client to track usage
type TokenTrackingClient struct {
	client  Client
	tracker *ParquetTokenTracker
}

// NewTokenTrackingClient creates a wrapper client
func NewTokenTrackingClient(client Client, tracker *ParquetTokenTracker) *TokenTrackingClient {
	return &TokenTrackingClient{
		client:  client,
		tracker: tracker,
	}
}

// Chat implements Client
func (c *TokenTrackingClient) Chat(ctx context.Context, messages []types.Message) (*types.Response, error) {
	resp, err := c.client.Chat(ctx, messages)
	if err != nil {
		return nil, err
	}

	c.trackUsage(ctx, resp)

	return resp, nil
}

// ChatWithStructuredOutput implements Client
func (c *TokenTrackingClient) ChatWithStructuredOutput(ctx context.Context, messages []types.Message, schema any) (*types.Response, error) {
	resp, err := c.client.ChatWithStructuredOutput(ctx, messages, schema)
	if err != nil {
		return nil, err
	}

	c.trackUsage(ctx, resp)

	return resp, nil
}

// ExtractEntities implements Client
func (c *TokenTrackingClient) ExtractEntities(ctx context.Context, text string, entityTypes []string) ([]ExtractedEntity, error) {
	return c.client.ExtractEntities(ctx, text, entityTypes)
}

// ExtractRelations implements Client
func (c *TokenTrackingClient) ExtractRelations(ctx context.Context, text string, relationTypes []string) ([]ExtractedRelation, error) {
	return c.client.ExtractRelations(ctx, text, relationTypes)
}

// Summarize implements Client
func (c *TokenTrackingClient) Summarize(ctx context.Context, text string) (string, error) {
	return c.client.Summarize(ctx, text)
}

// GenerateText implements Client
func (c *TokenTrackingClient) GenerateText(ctx context.Context, prompt string) (string, error) {
	return c.client.GenerateText(ctx, prompt)
}

func (c *TokenTrackingClient) trackUsage(ctx context.Context, resp *types.Response) {
	if resp == nil || resp.TokensUsed == nil {
		return
	}

	// Use model from response if available
	model := resp.Model
	if model == "" {
		model = "unknown"
	}

	if err := c.tracker.AddUsage(ctx, resp.TokensUsed, model); err != nil {
		fmt.Printf("Warning: Failed to log token usage: %v\n", err)
	}
}

// GetModel returns the model identifier of the wrapped client.
func (c *TokenTrackingClient) GetModel() string {
	return c.client.GetModel()
}

// Close implements Client
func (c *TokenTrackingClient) Close() error {
	return c.client.Close()
}

// GetCapabilities returns the list of capabilities supported by this client.
func (c *TokenTrackingClient) GetCapabilities() []TaskCapability {
	return c.client.GetCapabilities()
}
