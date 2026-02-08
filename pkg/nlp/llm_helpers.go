package nlp

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math/rand"
	"regexp"
	"strings"
	"time"

	jsonrepair "github.com/kaptinlin/jsonrepair"
	"github.com/soundprediction/predicato/pkg/types"
)

// calculateProgressiveTimeout returns a timeout duration that increases with each attempt.
// Starts at 90s, increases by 45s per attempt, with ±20% jitter.
// Examples: attempt 0: 72-108s, attempt 1: 108-162s, attempt 2: 144-216s, attempt 8: 432-648s
func calculateProgressiveTimeout(attempt int) time.Duration {
	// Base timeout: 90s + (attempt * 45s)
	baseTimeout := time.Duration(90+attempt*45) * time.Second

	// Add ±20% jitter
	jitterPercent := 0.2
	jitterRange := float64(baseTimeout) * jitterPercent
	jitter := time.Duration(rand.Float64()*jitterRange*2 - jitterRange)

	timeout := baseTimeout + jitter
	// Ensure minimum timeout of 30s
	if timeout < 30*time.Second {
		timeout = 30 * time.Second
	}
	return timeout
}

// GenerateJSONResponseWithContinuation makes repeated LLM calls with continuation prompts
// until valid JSON is received or max retries is reached.
//
// Parameters:
//   - ctx: Context for the LLM call
//   - nlProcessor: The LLM client to use
//   - systemPrompt: The initial system/instruction prompt
//   - userPrompt: The user's request prompt
//   - targetStruct: A pointer to the struct to unmarshal JSON into (for validation)
//   - maxRetries: Maximum number of continuation attempts (default 3 if <= 0)
//
// Returns:
//   - The final JSON string (may be partial if all retries exhausted)
//   - Error if all retries fail or if there's a critical error
//
// Example:
//
//	type MyStruct struct {
//	    Name string `json:"name"`
//	    Items []string `json:"items"`
//	}
//	var result MyStruct
//	jsonStr, err := GenerateJSONResponseWithContinuation(
//	    ctx, nlProcessor,
//	    "You are a JSON generator. Return only valid JSON.",
//	    "Generate a list of 10 pregnancy tips",
//	    &result,
//	    5,
//	)

// RemoveThinkTags removes <think> tags and everything in between them from a string.
func RemoveThinkTags(input string) string {
	re := regexp.MustCompile(`(?s)<think>.*?</think>`)
	return re.ReplaceAllString(input, "")
}
func StripHtmlTags(s string) string {
	// Regular expression to match HTML tags.
	// <[^>]*> matches any character between '<' and '>'
	const tagRegex = "<[^>]*>"

	// Compile the regex
	r := regexp.MustCompile(tagRegex)

	// Replace all matches with an empty string
	return r.ReplaceAllString(s, "")
}
func GenerateJSONResponseWithContinuation(
	ctx context.Context,
	nlProcessor Client,
	systemPrompt string,
	userPrompt string,
	targetStruct interface{},
	maxRetries int,
) (string, error) {
	// Build initial messages
	messages := []types.Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	return GenerateJSONResponseWithContinuationMessages(ctx, nlProcessor, messages, targetStruct, maxRetries)
}

func isValidJson(s string) (bool, error) {
	var js json.RawMessage
	err := json.Unmarshal([]byte(s), &js)
	ok := (err == nil)
	return ok, err
}

// AppendOverlap appends s2 to s1, removing any overlapping part.
// It finds the longest suffix of s1 that is also a prefix of s2 and
// combines them to avoid duplicating the overlapping section.
func AppendOverlap(s1, s2 string) string {
	len1 := len(s1)
	len2 := len(s2)

	// Determine the maximum possible overlap length to check.
	// This can't be longer than the shorter of the two strings.
	maxOverlap := len1
	if len2 < len1 {
		maxOverlap = len2
	}

	// Iterate backwards from the longest possible overlap.
	// The first match found will be the longest one.
	for i := maxOverlap; i > 0; i-- {
		// Check if the suffix of s1 matches the prefix of s2.
		if s1[len1-i:] == s2[:i] {
			// If a match is found, append the non-overlapping part of s2 and return.
			return s1 + s2[i:]
		}
	}

	// If no overlap is found after checking all possibilities,
	// simply concatenate the two strings.
	return s1 + s2
}
func truncateToLastCloseBrace(s string) string {
	lastIndex := strings.LastIndex(s, "}")
	if lastIndex == -1 {
		return "" // No closing brace found
	}
	return s[:lastIndex+1]
}

// GenerateJSONResponseWithContinuationMessages makes repeated LLM calls with continuation prompts
// until valid JSON is received or max retries is reached. This version accepts pre-built messages.
//
// Parameters:
//   - ctx: Context for the LLM call
//   - nlProcessor: The LLM client to use
//   - messages: The initial message history
//   - targetStruct: A pointer to the struct to unmarshal JSON into (for validation)
//   - maxRetries: Maximum number of continuation attempts (default 3 if <= 0)
//
// Returns:
//   - The final JSON string (may be partial if all retries exhausted)
//   - Error if all retries fail or if there's a critical error
func GenerateJSONResponseWithContinuationMessages(
	ctx context.Context,
	nlProcessor Client,
	messages []types.Message,
	targetStruct interface{},
	maxRetries int,
) (string, error) {
	if maxRetries <= 0 {
		maxRetries = 8
	}

	// Make a copy of messages to avoid modifying the original slice
	workingMessages := make([]types.Message, len(messages))
	copy(workingMessages, messages)
	var accumulatedResponse string
	var lastError error

	for attempt := 0; attempt <= maxRetries; attempt++ {
		// Make LLM call
		if attempt > 0 {
			workingMessages[1].Content = messages[1].Content + "\nFinish your work:\n" + strings.TrimSpace(accumulatedResponse)
		}

		// Create context with progressive timeout (increases with each attempt + jitter)
		timeout := calculateProgressiveTimeout(attempt)
		attemptCtx, cancel := context.WithTimeout(ctx, timeout)

		// fmt.Printf("workingMessages[1].Content: %v\n", workingMessages[1].Content)
		response, err := nlProcessor.Chat(attemptCtx, workingMessages)
		cancel()

		if err != nil {
			lastError = fmt.Errorf("LLM call failed on attempt %d: %w", attempt+1, err)
			continue
		}

		if response == nil || response.Content == "" {
			lastError = fmt.Errorf("empty response from LLM on attempt %d", attempt+1)
			// ask the LLM to fix the output
			continue
		}
		startLen := len(accumulatedResponse)
		accumulatedResponse = AppendOverlap(strings.TrimSpace(accumulatedResponse), strings.TrimSpace((response.Content)))
		afterLen := len(accumulatedResponse)
		gap := afterLen - startLen
		ok, err := isValidJson(RemoveThinkTags(accumulatedResponse))

		if ok {
			cleanJSON := RemoveThinkTags(accumulatedResponse)
			if targetStruct != nil {
				// Try to unmarshal the JSON into the target struct.
				// If it's valid JSON but doesn't fit the struct, we still return
				// the JSON and let the caller handle any unmarshal issues.
				_ = json.Unmarshal([]byte(cleanJSON), targetStruct)
			}
			return cleanJSON, nil
		}

		if attempt > 1 && gap == 0 {
			accumulatedResponse = truncateToLastCloseBrace(accumulatedResponse)

			return accumulatedResponse, err
		}

	}

	if lastError != nil {
		accumulatedResponse = truncateToLastCloseBrace(accumulatedResponse)
		resp, _ := jsonrepair.JSONRepair(RemoveThinkTags(accumulatedResponse))
		return resp, fmt.Errorf("failed after %d attempts: %w", maxRetries+1, lastError)
	}

	return RemoveThinkTags(accumulatedResponse), fmt.Errorf("failed to generate valid JSON after %d attempts", maxRetries+1)
}

// GenerateJSONWithContinuation is a simpler version that doesn't validate against a struct
// and just ensures valid JSON is returned.
func GenerateJSONWithContinuation(
	ctx context.Context,
	nlProcessor Client,
	systemPrompt string,
	userPrompt string,
	maxRetries int,
) (string, error) {
	if maxRetries <= 0 {
		maxRetries = 8
	}

	// Build initial messages
	messages := []types.Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	var accumulatedResponse string
	var lastError error

	for attempt := 0; attempt <= maxRetries; attempt++ {
		// Create context with progressive timeout (increases with each attempt + jitter)
		timeout := calculateProgressiveTimeout(attempt)
		attemptCtx, cancel := context.WithTimeout(ctx, timeout)

		// Make LLM call
		response, err := nlProcessor.Chat(attemptCtx, messages)
		cancel()
		if err != nil {
			lastError = fmt.Errorf("LLM call failed on attempt %d: %w", attempt+1, err)
			continue
		}

		if response == nil || response.Content == "" {
			lastError = fmt.Errorf("empty response from LLM on attempt %d", attempt+1)
			continue
		}

		// Accumulate the response
		if attempt == 0 {
			accumulatedResponse = strings.TrimSpace(response.Content)
		} else {
			// For continuation, append the new content
			accumulatedResponse += strings.TrimSpace(response.Content)
		}

		// Try to repair JSON
		repairedJSON, _ := jsonrepair.JSONRepair(accumulatedResponse)

		// Validate it's proper JSON
		var testJSON interface{}
		err = json.Unmarshal([]byte(repairedJSON), &testJSON)
		if err != nil {
			// JSON is invalid or incomplete, try continuation
			lastError = fmt.Errorf("invalid JSON on attempt %d: %w", attempt+1, err)

			if attempt < maxRetries {
				// Add continuation prompt
				messages = append(messages, types.Message{
					Role:    "assistant",
					Content: accumulatedResponse,
				})
				messages = append(messages, types.Message{
					Role:    "user",
					Content: "The JSON response was incomplete or invalid. Please continue from where you left off and complete the JSON:",
				})
			}
			continue
		}

		// Success! Valid JSON
		return repairedJSON, nil
	}

	// All retries exhausted
	if lastError != nil {
		return accumulatedResponse, fmt.Errorf("failed after %d attempts: %w", maxRetries+1, lastError)
	}

	return accumulatedResponse, fmt.Errorf("failed to generate valid JSON after %d attempts", maxRetries+1)
}

// ExtractJSONFromResponse attempts to extract JSON from LLM responses that may contain
// markdown code blocks or other surrounding text.
func ExtractJSONFromResponse(response string) string {
	// Remove markdown code blocks if present
	response = strings.TrimSpace(response)

	// Check for ```json ... ``` pattern
	if strings.Contains(response, "```json") {
		start := strings.Index(response, "```json")
		end := strings.Index(response[start+7:], "```")
		if end != -1 {
			return strings.TrimSpace(response[start+7 : start+7+end])
		}
	}

	// Check for ``` ... ``` pattern
	if strings.HasPrefix(response, "```") {
		lines := strings.Split(response, "\n")
		if len(lines) > 2 {
			// Remove first and last line (the ``` markers)
			return strings.TrimSpace(strings.Join(lines[1:len(lines)-1], "\n"))
		}
	}

	// Try to find JSON object boundaries
	jsonStart := strings.Index(response, "{")
	jsonEnd := strings.LastIndex(response, "}")
	if jsonStart != -1 && jsonEnd != -1 && jsonEnd > jsonStart {
		return response[jsonStart : jsonEnd+1]
	}

	// Try to find JSON array boundaries
	jsonStart = strings.Index(response, "[")
	jsonEnd = strings.LastIndex(response, "]")
	if jsonStart != -1 && jsonEnd != -1 && jsonEnd > jsonStart {
		return response[jsonStart : jsonEnd+1]
	}

	// Return as-is if no extraction possible
	return response
}

// CSVParserFunc is a function type for parsing CSV/TSV strings into a slice of type T.
type CSVParserFunc[T any] func(csvContent string) ([]*T, error)

// GenerateCSVResponse generates a CSV response from an LLM and parses it into a slice of type T.
// It handles retries with continuation prompts when parsing fails.
//
// Parameters:
//   - ctx: Context for the LLM call
//   - nlProcessor: The LLM client to use
//   - logger: Logger for debugging (can be nil)
//   - messages: The initial message history
//   - csvParser: Function to parse CSV content into []T
//   - maxRetries: Maximum number of retry attempts (default 3 if <= 0)
//
// Returns:
//   - []T: Successfully parsed CSV records
//   - *types.BadLlmCsvResponse: Error information including messages, response, and error
//   - error: Error if all retries fail or if there's a critical error
//
// The function:
//   - Makes LLM calls with the provided messages
//   - Strips HTML tags and cleans up the response
//   - Parses CSV using the provided parser function
//   - Retries with continuation prompts if parsing fails
//   - Returns detailed error information via BadLlmCsvResponse
//
// Example:
//
//	type Entity struct {
//	    Name string `csv:"name"`
//	    Type string `csv:"type"`
//	}
//
//	// Create a parser function
//	parser := func(csvContent string) ([]*Entity, error) {
//	    return utils.UnmarshalCSV[Entity](csvContent, '\t')
//	}
//
//	entities, badResp, err := GenerateCSVResponse[Entity](
//	    ctx, nlProcessor, logger,
//	    []types.Message{
//	        {Role: "system", Content: "You are a CSV generator."},
//	        {Role: "user", Content: "Generate entity data in TSV format."},
//	    },
//	    parser,
//	    3,
//	)
func GenerateCSVResponse[T any](
	ctx context.Context,
	nlProcessor Client,
	logger *slog.Logger,
	messages []types.Message,
	csvParser CSVParserFunc[T],
	maxRetries int,
) ([]T, *types.BadLlmCsvResponse, error) {
	if maxRetries <= 0 {
		maxRetries = 8
	}

	// Make a copy of messages to avoid modifying the original slice
	workingMessages := make([]types.Message, len(messages))
	copy(workingMessages, messages)

	var lastResponse *types.Response
	var lastError error

	for attempt := 0; attempt <= maxRetries; attempt++ {
		// Create context with progressive timeout (increases with each attempt + jitter)
		timeout := calculateProgressiveTimeout(attempt)
		attemptCtx, cancel := context.WithTimeout(ctx, timeout)

		// Make LLM call
		response, err := nlProcessor.Chat(attemptCtx, workingMessages)
		cancel()
		if err != nil {
			lastError = fmt.Errorf("LLM call failed on attempt %d: %w", attempt+1, err)
			lastResponse = response

			// Add continuation prompt for next attempt
			if attempt < maxRetries {
				workingMessages = append(workingMessages, types.Message{
					Role:    RoleAssistant,
					Content: "",
				})
				workingMessages = append(workingMessages, types.Message{
					Role:    RoleUser,
					Content: "The previous response failed. Please try again with valid CSV/TSV format:",
				})
			}
			continue
		}

		if response == nil || response.Content == "" {
			lastError = fmt.Errorf("empty response from LLM on attempt %d", attempt+1)
			lastResponse = response

			// Add continuation prompt for next attempt
			if attempt < maxRetries {
				workingMessages = append(workingMessages, types.Message{
					Role:    RoleAssistant,
					Content: "",
				})
				workingMessages = append(workingMessages, types.Message{
					Role:    RoleUser,
					Content: "No response received. Please provide the CSV/TSV data:",
				})
			}
			continue
		}

		lastResponse = response

		// Log response if logger is provided
		if logger != nil {
			logger.Debug("LLM CSV response received", "attempt", attempt+1, "length", len(response.Content))
		}

		// Clean up the response
		cleanedResponse := StripHtmlTags(response.Content)
		if strings.HasSuffix(cleanedResponse, "\n") {
			lines := strings.Split(cleanedResponse, "\n")
			if len(lines) > 1 {
				cleanedResponse = strings.Join(lines[:len(lines)-1], "\n")
			}
		}

		// Try to parse the CSV/TSV using the provided parser
		resultPtrs, err := csvParser(cleanedResponse)
		if err != nil {
			lastError = fmt.Errorf("failed to parse CSV on attempt %d: %w", attempt+1, err)

			if logger != nil {
				logger.Debug("CSV parsing failed", "attempt", attempt+1, "error", err, "response", cleanedResponse)
			}

			// Add continuation prompt for next attempt
			if attempt < maxRetries {
				workingMessages = append(workingMessages, types.Message{
					Role:    RoleAssistant,
					Content: response.Content,
				})
				workingMessages = append(workingMessages, types.Message{
					Role:    RoleUser,
					Content: fmt.Sprintf("The CSV/TSV format was invalid: %v. Please provide valid TSV data with tab-separated values:", err),
				})
			}
			continue
		}

		// Convert pointer slice to value slice
		results := make([]T, 0, len(resultPtrs))
		for _, ptr := range resultPtrs {
			if ptr != nil {
				results = append(results, *ptr)
			}
		}

		// Success!
		if logger != nil {
			logger.Debug("CSV parsing successful", "attempt", attempt+1, "records", len(results))
		}

		return results, nil, nil
	}

	// All retries exhausted - return error information
	badResponse := &types.BadLlmCsvResponse{
		Messages: make([]*types.Message, 0, len(workingMessages)),
		Response: "",
		Error:    lastError,
	}

	// Copy messages for error reporting
	for i := range workingMessages {
		msg := workingMessages[i]
		badResponse.Messages = append(badResponse.Messages, &msg)
	}

	if lastResponse != nil {
		badResponse.Response = lastResponse.Content
	}

	if logger != nil {
		logger.Error("CSV generation failed after all retries",
			"attempts", maxRetries+1,
			"error", lastError,
		)
	}

	if lastError != nil {
		return nil, badResponse, fmt.Errorf("failed after %d attempts: %w", maxRetries+1, lastError)
	}

	return nil, badResponse, fmt.Errorf("failed to generate valid CSV after %d attempts", maxRetries+1)
}

// YAMLParserFunc is a function type for parsing YAML strings into a slice of type T.
type YAMLParserFunc[T any] func(yamlContent string) ([]*T, error)

// GenerateYAMLResponse generates a YAML response from an LLM and parses it into a slice of type T.
// It handles retries with continuation prompts when parsing fails.
func GenerateYAMLResponse[T any](
	ctx context.Context,
	nlProcessor Client,
	logger *slog.Logger,
	messages []types.Message,
	yamlParser YAMLParserFunc[T],
	maxRetries int,
) ([]T, *types.BadLlmCsvResponse, error) {
	if maxRetries <= 0 {
		maxRetries = 3
	}

	// Make a copy of messages to avoid modifying the original slice
	workingMessages := make([]types.Message, len(messages))
	copy(workingMessages, messages)

	var lastResponse *types.Response
	var lastError error

	for attempt := 0; attempt <= maxRetries; attempt++ {
		// Make LLM call
		response, err := nlProcessor.Chat(ctx, workingMessages)
		if err != nil {
			lastError = fmt.Errorf("LLM call failed on attempt %d: %w", attempt+1, err)
			lastResponse = response

			// Add continuation prompt for next attempt
			if attempt < maxRetries {
				workingMessages = append(workingMessages, types.Message{
					Role:    RoleAssistant,
					Content: "",
				})
				workingMessages = append(workingMessages, types.Message{
					Role:    RoleUser,
					Content: "The previous response failed. Please try again with valid YAML format:",
				})
			}
			continue
		}

		if response == nil || response.Content == "" {
			lastError = fmt.Errorf("empty response from LLM on attempt %d", attempt+1)
			lastResponse = response

			// Add continuation prompt for next attempt
			if attempt < maxRetries {
				workingMessages = append(workingMessages, types.Message{
					Role:    RoleAssistant,
					Content: "",
				})
				workingMessages = append(workingMessages, types.Message{
					Role:    RoleUser,
					Content: "No response received. Please provide the YAML data:",
				})
			}
			continue
		}

		lastResponse = response

		// Clean up the response (remove code blocks)
		cleanedResponse := response.Content

		// Remove markdown code blocks if present
		if strings.Contains(cleanedResponse, "```") {
			cleanedResponse = ExtractJSONFromResponse(cleanedResponse) // Reusing this helper as it strips backticks well enough or we can make a specific one
		}

		// Also strip HTML tags just in case
		cleanedResponse = StripHtmlTags(cleanedResponse)

		// Try to parse using the provided parser
		resultPtrs, err := yamlParser(cleanedResponse)
		if err != nil {
			lastError = fmt.Errorf("failed to parse YAML on attempt %d: %w", attempt+1, err)

			if logger != nil {
				logger.Debug("YAML parsing failed", "attempt", attempt+1, "error", err, "response", cleanedResponse)
			}

			// Add continuation prompt for next attempt
			if attempt < maxRetries {
				workingMessages = append(workingMessages, types.Message{
					Role:    RoleAssistant,
					Content: response.Content,
				})
				workingMessages = append(workingMessages, types.Message{
					Role:    RoleUser,
					Content: fmt.Sprintf("The YAML format was invalid: %v. Please provide valid YAML data:", err),
				})
			}
			continue
		}

		// Convert pointer slice to value slice
		results := make([]T, 0, len(resultPtrs))
		for _, ptr := range resultPtrs {
			if ptr != nil {
				results = append(results, *ptr)
			}
		}

		return results, nil, nil
	}

	// All retries exhausted - return error information
	// We reuse BadLlmCsvResponse for now as it has the right fields, maybe rename it later?
	// It contains Messages, Response, Error.
	badResponse := &types.BadLlmCsvResponse{
		Messages: make([]*types.Message, 0, len(workingMessages)),
		Response: "",
		Error:    lastError,
	}

	for i := range workingMessages {
		msg := workingMessages[i]
		badResponse.Messages = append(badResponse.Messages, &msg)
	}

	if lastResponse != nil {
		badResponse.Response = lastResponse.Content
	}

	return nil, badResponse, fmt.Errorf("failed to generate valid YAML after %d attempts", maxRetries+1)
}

// ExtractExtendedHelperFunc is a helper for LLM clients to implement ExtractExtended.
// It constructs a prompt with the expected schema and uses the client's ChatWithStructuredOutput or generation capabilities.
func ExtractExtendedHelper(ctx context.Context, client Client, text string) (*ExtendedExtractionResult, error) {
	// Define the schema/structure we want the LLM to populate
	// We use the JSON tags from the structs in client_types.go
	// Since we can't easily pass the Go struct definition to the prompt as a schema object in all cases,
	// we'll describe it in the system prompt or rely on the client's ability to handle the struct.
	
	systemPrompt := `You are an expert medical information extraction system.
Your task is to extract structured information from the provided text, including:
1. Entities: Named entities grouped by type (e.g., condition, medication, symptom).
2. Relations: Relationships between entities (e.g., medication treats condition).
3. Triples: Extended facts with context (subject, predicate, object, plus condition, temporal, certainty, etc.).
4. Rules: Conditional logic found in the text (IF antecedent THEN consequent).

Return the output as a JSON object matching this structure:
{
  "source_text": "Original text segment",
  "entities": { "type": ["entity1", "entity2"] },
  "relations": [ { "source": "A", "target": "B", "type": "treats", "confidence": 1.0 } ],
  "triples": [ 
    { 
      "subject": "Metformin", "predicate": "reduces", "object": "glucose", 
      "condition": "in T2DM", "temporal": "", "certainty": "high" 
    } 
  ],
  "rules": [
    {
      "antecedent": "fasting glucose > 95",
      "consequent": "initiate insulin",
      "exception": "unless contraindicated",
      "rule_type": "clinical_decision"
    }
  ]
}
`

	userPrompt := fmt.Sprintf("Extract structured information from the following text:\n\n%s", text)

	messages := []types.Message{
		NewSystemMessage(systemPrompt),
		NewUserMessage(userPrompt),
	}

	// We use a temporary struct to capture the LLM output, ensuring it aligns with ExtendedExtractionResult
	var result ExtendedExtractionResult

	// Use the continuation helper to get valid JSON
	_, err := GenerateJSONResponseWithContinuationMessages(ctx, client, messages, &result, 3)
	if err != nil {
		return nil, fmt.Errorf("failed to extract extended info with LLM: %w", err)
	}

	// If unmarshalling inside the helper succeeded (it populated result if targetStruct was passed), we are good.
	// But GenerateJSONResponseWithContinuationMessages takes interface{}, and populates it if possible.
	// Let's manually unmarshal again to be sure or just trust the helper.
	// The helper does: _ = json.Unmarshal([]byte(cleanJSON), targetStruct)
	// So 'result' should be populated.
	
	// Ensure SourceText is set if the LLM didn't return it
	if result.SourceText == "" {
		result.SourceText = text
	}

	return &result, nil
}
