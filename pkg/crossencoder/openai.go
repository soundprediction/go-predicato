package crossencoder

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/types"
)

// OpenAIRerankerClient implements cross-encoder functionality using OpenAI's API
// This reranker uses the OpenAI API to run a simple boolean classifier prompt concurrently
// for each passage. Log-probabilities are used to rank the passages.
type OpenAIRerankerClient struct {
	client    nlp.Client
	semaphore chan struct{} // Controls concurrency
	config    Config
}

// NewOpenAIRerankerClient creates a new OpenAI-based reranker client
func NewOpenAIRerankerClient(nlProcessor nlp.Client, config Config) *OpenAIRerankerClient {
	if config.Model == "" {
		config.Model = "gpt-4o-mini"
	}
	if config.MaxConcurrency <= 0 {
		config.MaxConcurrency = 10
	}

	return &OpenAIRerankerClient{
		client:    nlProcessor,
		config:    config,
		semaphore: make(chan struct{}, config.MaxConcurrency),
	}
}

// Rank ranks the given passages based on their relevance to the query
func (c *OpenAIRerankerClient) Rank(ctx context.Context, query string, passages []string) ([]RankedPassage, error) {
	if len(passages) == 0 {
		return []RankedPassage{}, nil
	}

	// Create a slice to hold results with original indices
	type passageResult struct {
		err     error
		passage string
		score   float64
		index   int
	}

	results := make([]passageResult, len(passages))
	var wg sync.WaitGroup

	// Process passages concurrently with semaphore for rate limiting
	for i, passage := range passages {
		wg.Add(1)
		go func(idx int, p string) {
			defer wg.Done()

			// Acquire semaphore
			c.semaphore <- struct{}{}
			defer func() { <-c.semaphore }()

			score, err := c.scorePassage(ctx, query, p)
			results[idx] = passageResult{
				passage: p,
				score:   score,
				index:   idx,
				err:     err,
			}
		}(i, passage)
	}

	wg.Wait()

	// Check for errors and collect successful results
	var rankedPassages []RankedPassage
	for _, result := range results {
		if result.err != nil {
			return nil, fmt.Errorf("error scoring passage %d: %w", result.index, result.err)
		}
		rankedPassages = append(rankedPassages, RankedPassage{
			Passage: result.passage,
			Score:   result.score,
		})
	}

	// Sort by score descending
	sort.Slice(rankedPassages, func(i, j int) bool {
		return rankedPassages[i].Score > rankedPassages[j].Score
	})

	return rankedPassages, nil
}

// scorePassage scores a single passage against the query using OpenAI's logprobs
func (c *OpenAIRerankerClient) scorePassage(ctx context.Context, query, passage string) (float64, error) {
	messages := []types.Message{
		nlp.NewSystemMessage("You are an expert tasked with determining whether the passage is relevant to the query"),
		nlp.NewUserMessage(fmt.Sprintf(`Respond with "True" if PASSAGE is relevant to QUERY and "False" otherwise.
<PASSAGE>
%s
</PASSAGE>
<QUERY>
%s
</QUERY>`, passage, query)),
	}

	// For now, we'll use a simplified scoring approach since we don't have
	// direct access to OpenAI's logprobs in our LLM abstraction
	// This is a placeholder implementation that would need to be enhanced
	// with actual logprob support or use the raw OpenAI client

	response, err := c.client.Chat(ctx, messages)
	if err != nil {
		return 0, fmt.Errorf("failed to get response: %w", err)
	}

	// Simple heuristic scoring based on response content
	// In a real implementation, you'd want to use logprobs for better accuracy
	content := response.Content
	if len(content) == 0 {
		return 0.5, nil // neutral score if no response
	}

	// Check if response starts with "True" (case-insensitive)
	firstWord := ""
	for i, r := range content {
		if r == ' ' || r == '\n' || r == '\t' {
			firstWord = content[:i]
			break
		}
	}
	if firstWord == "" {
		firstWord = content
	}

	switch firstWord {
	case "True", "true", "TRUE", "Yes", "yes", "YES":
		return 0.8, nil // High relevance score
	case "False", "false", "FALSE", "No", "no", "NO":
		return 0.2, nil // Low relevance score
	default:
		return 0.5, nil // Neutral score for ambiguous responses
	}
}

// Close cleans up any resources used by the client
func (c *OpenAIRerankerClient) Close() error {
	if c.client != nil {
		return c.client.Close()
	}
	return nil
}
