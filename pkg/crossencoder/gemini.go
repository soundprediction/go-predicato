package crossencoder

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/types"
)

// GeminiRerankerClient implements cross-encoder functionality using Google Gemini models
// This reranker uses the Gemini API to score passage relevance through classification
type GeminiRerankerClient struct {
	client    nlp.Client
	semaphore chan struct{} // Controls concurrency
	config    Config
}

// NewGeminiRerankerClient creates a new Gemini-based reranker client
func NewGeminiRerankerClient(nlProcessor nlp.Client, config Config) *GeminiRerankerClient {
	if config.Model == "" {
		config.Model = "gemini-1.5-flash"
	}
	if config.MaxConcurrency <= 0 {
		config.MaxConcurrency = 10
	}

	return &GeminiRerankerClient{
		client:    nlProcessor,
		config:    config,
		semaphore: make(chan struct{}, config.MaxConcurrency),
	}
}

// Rank ranks the given passages based on their relevance to the query
func (c *GeminiRerankerClient) Rank(ctx context.Context, query string, passages []string) ([]RankedPassage, error) {
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

// scorePassage scores a single passage against the query using Gemini's classification
func (c *GeminiRerankerClient) scorePassage(ctx context.Context, query, passage string) (float64, error) {
	messages := []types.Message{
		nlp.NewSystemMessage("You are an expert document relevance scorer. Your task is to determine how relevant a passage is to a given query. Respond with a single number between 0 and 1, where 0 means completely irrelevant and 1 means perfectly relevant."),
		nlp.NewUserMessage(fmt.Sprintf(`Rate the relevance of this PASSAGE to the QUERY on a scale from 0.0 to 1.0.

QUERY: %s

PASSAGE: %s

Respond with only a decimal number between 0.0 and 1.0 (e.g., 0.85):`, query, passage)),
	}

	response, err := c.client.Chat(ctx, messages)
	if err != nil {
		return 0, fmt.Errorf("failed to get response: %w", err)
	}

	// Parse the score from the response
	content := response.Content
	if len(content) == 0 {
		return 0.5, nil // neutral score if no response
	}

	// Try to extract a numeric score from the response
	score, err := c.parseScore(content)
	if err != nil {
		// Fallback to keyword-based scoring if parsing fails
		return c.fallbackScore(content), nil
	}

	// Ensure score is within valid range
	if score < 0 {
		score = 0
	} else if score > 1 {
		score = 1
	}

	return score, nil
}

// parseScore attempts to parse a numeric score from the response
func (c *GeminiRerankerClient) parseScore(content string) (float64, error) {
	// Simple parsing - look for the first decimal number in the response
	var score float64
	var hasDecimal bool
	var numStr string

	for i, r := range content {
		if r >= '0' && r <= '9' {
			numStr += string(r)
		} else if r == '.' && !hasDecimal {
			hasDecimal = true
			numStr += string(r)
		} else if numStr != "" {
			// We've found the end of a number
			break
		}

		// If we've read more than 10 characters, something's wrong
		if i > 10 {
			break
		}
	}

	if numStr == "" {
		return 0, fmt.Errorf("no numeric score found")
	}

	// Parse the extracted number
	multiplier := 1.0
	decimalPos := -1

	// Find decimal position
	for i, r := range numStr {
		if r == '.' {
			decimalPos = i
			break
		}
	}

	// Convert to float
	for i, r := range numStr {
		if r >= '0' && r <= '9' {
			digit := int(r - '0')
			if decimalPos >= 0 && i > decimalPos {
				multiplier /= 10
				score += float64(digit) * multiplier
			} else if decimalPos < 0 || i < decimalPos {
				score = score*10 + float64(digit)
			}
		}
	}

	return score, nil
}

// fallbackScore provides keyword-based scoring when numeric parsing fails
func (c *GeminiRerankerClient) fallbackScore(content string) float64 {
	content = strings.ToLower(content)

	// High relevance indicators
	if strings.Contains(content, "very relevant") || strings.Contains(content, "highly relevant") ||
		strings.Contains(content, "perfect") || strings.Contains(content, "excellent") {
		return 0.9
	}

	// Medium-high relevance
	if strings.Contains(content, "relevant") || strings.Contains(content, "related") ||
		strings.Contains(content, "good") || strings.Contains(content, "strong") {
		return 0.7
	}

	// Medium relevance
	if strings.Contains(content, "somewhat") || strings.Contains(content, "partially") ||
		strings.Contains(content, "moderate") || strings.Contains(content, "fair") {
		return 0.5
	}

	// Low relevance
	if strings.Contains(content, "not relevant") || strings.Contains(content, "irrelevant") ||
		strings.Contains(content, "unrelated") || strings.Contains(content, "poor") {
		return 0.2
	}

	// Default neutral score
	return 0.5
}

// Close cleans up any resources used by the client
func (c *GeminiRerankerClient) Close() error {
	if c.client != nil {
		return c.client.Close()
	}
	return nil
}
