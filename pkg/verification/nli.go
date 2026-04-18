//go:build cgo

package verification

import (
	"context"
	"fmt"
	"sync"

	gocandle "github.com/soundprediction/go-candle/pkg/candle"
)

// CandleNLIConfig holds configuration for the candle-backed NLI client.
type CandleNLIConfig struct {
	// Model is the HuggingFace model ID (e.g., "cross-encoder/nli-deberta-v3-xsmall").
	Model string

	// CacheDir overrides the default HF cache directory.
	CacheDir string
}

// CandleNLIClient implements NLIClient using go-candle's sequence classification pipeline
// backed by a DeBERTa-v2 NLI cross-encoder.
type CandleNLIClient struct {
	pipeline *gocandle.SeqClassificationPipeline
	mu       sync.Mutex
}

// NewCandleNLIClient creates a new candle-backed NLI client.
// The model is automatically downloaded from HuggingFace Hub on first use.
func NewCandleNLIClient(cfg *CandleNLIConfig) (*CandleNLIClient, error) {
	if cfg.Model == "" {
		return nil, fmt.Errorf("model is required")
	}

	if err := gocandle.Init(); err != nil {
		return nil, fmt.Errorf("failed to init candle: %w", err)
	}

	pipeline, err := gocandle.NewSeqClassificationPipeline(gocandle.SeqClassificationConfig{
		ModelID:  cfg.Model,
		CacheDir: cfg.CacheDir,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create NLI pipeline: %w", err)
	}

	return &CandleNLIClient{pipeline: pipeline}, nil
}

// Classify returns NLI scores for a single premise-hypothesis pair.
// The input is formatted as "premise [SEP] hypothesis" for the cross-encoder tokenizer.
func (c *CandleNLIClient) Classify(_ context.Context, premise, hypothesis string) (*NLIScores, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.pipeline == nil {
		return nil, fmt.Errorf("NLI pipeline is closed")
	}

	// Cross-encoder models expect premise [SEP] hypothesis as a single input.
	// The tokenizer handles the separator token.
	input := premise + " [SEP] " + hypothesis

	logits, err := c.pipeline.Classify(input, true)
	if err != nil {
		return nil, fmt.Errorf("NLI classification failed: %w", err)
	}

	return logitsToNLIScores(logits), nil
}

// ClassifyBatch returns NLI scores for multiple premise-hypothesis pairs.
func (c *CandleNLIClient) ClassifyBatch(_ context.Context, premises, hypotheses []string) ([]NLIScores, error) {
	if len(premises) != len(hypotheses) {
		return nil, fmt.Errorf("premises and hypotheses must have same length")
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if c.pipeline == nil {
		return nil, fmt.Errorf("NLI pipeline is closed")
	}

	inputs := make([]string, len(premises))
	for i := range premises {
		inputs[i] = premises[i] + " [SEP] " + hypotheses[i]
	}

	batchLogits, err := c.pipeline.ClassifyBatch(inputs, true)
	if err != nil {
		return nil, fmt.Errorf("NLI batch classification failed: %w", err)
	}

	results := make([]NLIScores, len(batchLogits))
	for i, logits := range batchLogits {
		scores := logitsToNLIScores(logits)
		if scores != nil {
			results[i] = *scores
		}
	}

	return results, nil
}

// Close cleans up resources.
func (c *CandleNLIClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.pipeline != nil {
		c.pipeline.Close()
		c.pipeline = nil
	}
	return nil
}

// logitsToNLIScores converts raw model logits to NLI scores.
// The cross-encoder/nli-deberta-v3-* models output [contradiction, neutral, entailment].
func logitsToNLIScores(logits []float32) *NLIScores {
	if len(logits) < 3 {
		return nil
	}
	return &NLIScores{
		Contradiction: logits[0],
		Neutral:       logits[1],
		Entailment:    logits[2],
	}
}
