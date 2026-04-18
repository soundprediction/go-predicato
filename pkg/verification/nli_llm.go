package verification

import (
	"context"
	"fmt"
	"strings"

	"github.com/soundprediction/predicato/pkg/nlp"
)

const nliPromptTemplate = `You are a precise Natural Language Inference (NLI) classifier. Your task is to determine whether a hypothesis is supported by, contradicted by, or neutral with respect to a given premise.

Premise (source evidence):
%s

Hypothesis (claim to verify):
%s

Classify the relationship as exactly one of:
- ENTAILMENT: The hypothesis is clearly supported by the premise.
- CONTRADICTION: The hypothesis directly contradicts the premise.
- NEUTRAL: The hypothesis is neither supported nor contradicted by the premise; there is not enough information.

Respond with ONLY one word: ENTAILMENT, CONTRADICTION, or NEUTRAL.`

const nliBatchPromptTemplate = `You are a precise Natural Language Inference (NLI) classifier. For each numbered pair below, classify the relationship between the premise and hypothesis as exactly one of: ENTAILMENT, CONTRADICTION, or NEUTRAL.

%s

Respond with ONLY the classifications, one per line, in order (e.g.):
1. ENTAILMENT
2. NEUTRAL
3. CONTRADICTION`

// LLMNLIClient implements NLIClient using an LLM for entailment classification.
// This is a fallback when the dedicated NLI cross-encoder model is unavailable.
type LLMNLIClient struct {
	llm nlp.Client
}

// NewLLMNLIClient creates a new LLM-backed NLI client.
func NewLLMNLIClient(llm nlp.Client) *LLMNLIClient {
	return &LLMNLIClient{llm: llm}
}

// Classify returns NLI scores for a single premise-hypothesis pair via LLM.
func (c *LLMNLIClient) Classify(ctx context.Context, premise, hypothesis string) (*NLIScores, error) {
	prompt := fmt.Sprintf(nliPromptTemplate, premise, hypothesis)

	resp, err := c.llm.GenerateText(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("LLM NLI classification failed: %w", err)
	}

	return parseLLMVerdict(resp), nil
}

// ClassifyBatch returns NLI scores for multiple premise-hypothesis pairs.
// Uses a single batched prompt for efficiency.
func (c *LLMNLIClient) ClassifyBatch(ctx context.Context, premises, hypotheses []string) ([]NLIScores, error) {
	if len(premises) != len(hypotheses) {
		return nil, fmt.Errorf("premises and hypotheses must have same length")
	}

	if len(premises) == 0 {
		return nil, nil
	}

	// For small batches, use individual calls for reliability
	if len(premises) <= 3 {
		results := make([]NLIScores, len(premises))
		for i := range premises {
			scores, err := c.Classify(ctx, premises[i], hypotheses[i])
			if err != nil {
				// Default to neutral on error
				results[i] = NLIScores{Neutral: 1.0}
				continue
			}
			results[i] = *scores
		}
		return results, nil
	}

	// Build batched prompt
	var pairs strings.Builder
	for i := range premises {
		fmt.Fprintf(&pairs, "%d. Premise: %s\n   Hypothesis: %s\n\n", i+1, premises[i], hypotheses[i])
	}

	prompt := fmt.Sprintf(nliBatchPromptTemplate, pairs.String())
	resp, err := c.llm.GenerateText(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("LLM batch NLI failed: %w", err)
	}

	return parseBatchLLMVerdicts(resp, len(premises)), nil
}

// Close is a no-op since we don't own the LLM client.
func (c *LLMNLIClient) Close() error {
	return nil
}

// parseLLMVerdict converts an LLM text response to NLI scores.
func parseLLMVerdict(response string) *NLIScores {
	upper := strings.ToUpper(strings.TrimSpace(response))

	// Check for the verdict keywords
	switch {
	case strings.Contains(upper, "ENTAILMENT"):
		return &NLIScores{Entailment: 1.0, Neutral: 0.0, Contradiction: 0.0}
	case strings.Contains(upper, "CONTRADICTION"):
		return &NLIScores{Entailment: 0.0, Neutral: 0.0, Contradiction: 1.0}
	default:
		return &NLIScores{Entailment: 0.0, Neutral: 1.0, Contradiction: 0.0}
	}
}

// parseBatchLLMVerdicts parses a multi-line LLM response into NLI scores.
func parseBatchLLMVerdicts(response string, expected int) []NLIScores {
	lines := strings.Split(strings.TrimSpace(response), "\n")
	results := make([]NLIScores, expected)

	lineIdx := 0
	for i := 0; i < expected; i++ {
		if lineIdx < len(lines) {
			scores := parseLLMVerdict(lines[lineIdx])
			results[i] = *scores
			lineIdx++
		} else {
			// Default to neutral if we ran out of lines
			results[i] = NLIScores{Neutral: 1.0}
		}
	}

	return results
}
