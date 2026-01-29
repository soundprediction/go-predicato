package predicato

import (
	"context"
	"fmt"
	"strings"

	"github.com/soundprediction/predicato/pkg/types"
)

// AnalyzeContentRelevance uses the configured LLM to determine if content is
// relevant to the specified topics.
func (c *Client) AnalyzeContentRelevance(ctx context.Context, content string, topics []string) (bool, error) {
	if c.nlProcessor == nil {
		return false, fmt.Errorf("NLP processor not configured")
	}

	// Build the prompt
	topicsStr := strings.Join(topics, "\n- ")
	prompt := fmt.Sprintf(`Analyze the following text content and determine if it is primarily related to ANY of these topics:
- %s

Respond with "YES" if it is relevant to any of these topics, and "NO" if it is not.
Provide only "YES" or "NO" as your answer, with no additional explanation.

Text content to analyze:
---
%s
---`, topicsStr, content)

	// Call the NLP processor
	messages := []types.Message{
		{Role: "user", Content: prompt},
	}

	response, err := c.nlProcessor.Chat(ctx, messages)
	if err != nil {
		return false, fmt.Errorf("LLM request failed: %w", err)
	}

	// Parse the response
	responseText := strings.ToUpper(strings.TrimSpace(response.Content))
	return strings.Contains(responseText, "YES"), nil
}

// ExtractSourceInfo uses the configured LLM to extract source metadata from
// web content.
func (c *Client) ExtractSourceInfo(ctx context.Context, content string, url string) (string, string, error) {
	if c.nlProcessor == nil {
		return "Unknown", "other", fmt.Errorf("NLP processor not configured")
	}

	prompt := fmt.Sprintf(`Analyze the following web page content and extract the source information.

URL: %s

Text content:
---
%s
---

Please identify:
1. Source Name: The name of the organization, website, or publisher (e.g., "CDC", "Mayo Clinic", "WHO", "WebMD")
2. Source Type: The type of source (choose one: "government_health_agency", "medical_institution", "academic", "health_information_website", "news", "other")

Respond in this exact format:
SOURCE_NAME: [name here]
SOURCE_TYPE: [type here]

If you cannot determine the source, respond with:
SOURCE_NAME: Unknown
SOURCE_TYPE: other`, url, content)

	messages := []types.Message{
		{Role: "user", Content: prompt},
	}

	response, err := c.nlProcessor.Chat(ctx, messages)
	if err != nil {
		return "Unknown", "other", fmt.Errorf("LLM request failed: %w", err)
	}

	// Parse the response
	sourceName := "Unknown"
	sourceType := "other"

	for _, line := range strings.Split(response.Content, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "SOURCE_NAME:") {
			sourceName = strings.TrimSpace(strings.TrimPrefix(line, "SOURCE_NAME:"))
		} else if strings.HasPrefix(line, "SOURCE_TYPE:") {
			sourceType = strings.ToLower(strings.TrimSpace(strings.TrimPrefix(line, "SOURCE_TYPE:")))
		}
	}

	return sourceName, sourceType, nil
}
