package nlp

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/soundprediction/predicato/pkg/types"
)

// RetryConfig holds configuration for retry behavior
type RetryConfig struct {
	// MaxRetries is the maximum number of retry attempts (default: 3)
	MaxRetries int
	// InitialDelay is the initial delay before the first retry (default: 1 second)
	InitialDelay time.Duration
	// MaxDelay is the maximum delay between retries (default: 60 seconds)
	MaxDelay time.Duration
	// BackoffMultiplier is the multiplier for exponential backoff (default: 2.0)
	BackoffMultiplier float64
	// JitterFraction is the fraction of delay to use as random jitter (default: 0.1 = 10%)
	// Setting to 0 disables jitter
	JitterFraction float64
}

// DefaultRetryConfig returns the default retry configuration
func DefaultRetryConfig() *RetryConfig {
	return &RetryConfig{
		MaxRetries:        3,
		InitialDelay:      1 * time.Second,
		MaxDelay:          60 * time.Second,
		BackoffMultiplier: 2.0,
		JitterFraction:    0.1, // 10% jitter by default
	}
}

// RetryClient wraps an LLM client and adds retry logic with exponential backoff
type RetryClient struct {
	client Client
	config *RetryConfig
}

// NewRetryClient creates a new retry client wrapper.
// Returns an error if client is nil.
func NewRetryClient(client Client, config *RetryConfig) (*RetryClient, error) {
	if client == nil {
		return nil, fmt.Errorf("client cannot be nil")
	}
	if config == nil {
		config = DefaultRetryConfig()
	}
	// Ensure sensible defaults
	if config.MaxRetries < 0 {
		config.MaxRetries = 3
	}
	if config.InitialDelay <= 0 {
		config.InitialDelay = 1 * time.Second
	}
	if config.MaxDelay <= 0 {
		config.MaxDelay = 60 * time.Second
	}
	if config.BackoffMultiplier <= 0 {
		config.BackoffMultiplier = 2.0
	}

	return &RetryClient{
		client: client,
		config: config,
	}, nil
}

// MustNewRetryClient creates a new retry client wrapper, panicking if client is nil.
// Use this when you're certain the client is not nil.
func MustNewRetryClient(client Client, config *RetryConfig) *RetryClient {
	rc, err := NewRetryClient(client, config)
	if err != nil {
		panic(err)
	}
	return rc
}

// Chat implements the Client interface with retry logic
func (r *RetryClient) Chat(ctx context.Context, messages []types.Message) (*types.Response, error) {
	var lastErr error

	for attempt := 0; attempt <= r.config.MaxRetries; attempt++ {
		// If this is a retry, wait with exponential backoff
		if attempt > 0 {
			delay := r.calculateDelay(attempt)
			select {
			case <-time.After(delay):
				// Continue with retry
			case <-ctx.Done():
				return nil, fmt.Errorf("context cancelled during retry backoff: %w", ctx.Err())
			}
		}

		// Make the LLM call
		resp, err := r.client.Chat(ctx, messages)
		if err == nil {
			return resp, nil
		}

		// Store the error
		lastErr = err

		// Check if the error is retryable
		if !isRetryableError(err) {
			// Non-retryable error, fail immediately
			return nil, err
		}

		// Log retry attempt (in production, this should use a logger)
		// For now, we just continue to retry
	}

	// All retries exhausted
	return nil, fmt.Errorf("failed after %d retries: %w", r.config.MaxRetries, lastErr)
}

// ChatWithStructuredOutput implements the Client interface with retry logic
func (r *RetryClient) ChatWithStructuredOutput(ctx context.Context, messages []types.Message, schema any) (*types.Response, error) {
	var lastErr error

	for attempt := 0; attempt <= r.config.MaxRetries; attempt++ {
		// If this is a retry, wait with exponential backoff
		if attempt > 0 {
			delay := r.calculateDelay(attempt)
			select {
			case <-time.After(delay):
				// Continue with retry
			case <-ctx.Done():
				return nil, fmt.Errorf("context cancelled during retry backoff: %w", ctx.Err())
			}
		}

		// Make the LLM call
		result, err := r.client.ChatWithStructuredOutput(ctx, messages, schema)
		if err == nil {
			return result, nil
		}

		// Store the error
		lastErr = err

		// Check if the error is retryable
		if !isRetryableError(err) {
			// Non-retryable error, fail immediately
			return nil, err
		}

		// Log retry attempt (in production, this should use a logger)
		// For now, we just continue to retry
	}

	// All retries exhausted
	return nil, fmt.Errorf("failed after %d retries: %w", r.config.MaxRetries, lastErr)
}

// ExtractEntities implements the Client interface with retry logic
func (r *RetryClient) ExtractEntities(ctx context.Context, text string, entityTypes []string) ([]ExtractedEntity, error) {
	var lastErr error

	for attempt := 0; attempt <= r.config.MaxRetries; attempt++ {
		if attempt > 0 {
			delay := r.calculateDelay(attempt)
			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return nil, fmt.Errorf("context cancelled during retry backoff: %w", ctx.Err())
			}
		}

		result, err := r.client.ExtractEntities(ctx, text, entityTypes)
		if err == nil {
			return result, nil
		}
		lastErr = err

		if !isRetryableError(err) {
			return nil, err
		}
	}
	return nil, fmt.Errorf("failed after %d retries: %w", r.config.MaxRetries, lastErr)
}

// ExtractRelations implements the Client interface with retry logic
func (r *RetryClient) ExtractRelations(ctx context.Context, text string, relationTypes []string) ([]ExtractedRelation, error) {
	var lastErr error

	for attempt := 0; attempt <= r.config.MaxRetries; attempt++ {
		if attempt > 0 {
			delay := r.calculateDelay(attempt)
			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return nil, fmt.Errorf("context cancelled during retry backoff: %w", ctx.Err())
			}
		}

		result, err := r.client.ExtractRelations(ctx, text, relationTypes)
		if err == nil {
			return result, nil
		}
		lastErr = err

		if !isRetryableError(err) {
			return nil, err
		}
	}
	return nil, fmt.Errorf("failed after %d retries: %w", r.config.MaxRetries, lastErr)
}

// Summarize implements the Client interface with retry logic
func (r *RetryClient) Summarize(ctx context.Context, text string) (string, error) {
	var lastErr error

	for attempt := 0; attempt <= r.config.MaxRetries; attempt++ {
		if attempt > 0 {
			delay := r.calculateDelay(attempt)
			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return "", fmt.Errorf("context cancelled during retry backoff: %w", ctx.Err())
			}
		}

		result, err := r.client.Summarize(ctx, text)
		if err == nil {
			return result, nil
		}
		lastErr = err

		if !isRetryableError(err) {
			return "", err
		}
	}
	return "", fmt.Errorf("failed after %d retries: %w", r.config.MaxRetries, lastErr)
}

// GenerateText implements the Client interface with retry logic
func (r *RetryClient) GenerateText(ctx context.Context, prompt string) (string, error) {
	var lastErr error

	for attempt := 0; attempt <= r.config.MaxRetries; attempt++ {
		if attempt > 0 {
			delay := r.calculateDelay(attempt)
			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return "", fmt.Errorf("context cancelled during retry backoff: %w", ctx.Err())
			}
		}

		result, err := r.client.GenerateText(ctx, prompt)
		if err == nil {
			return result, nil
		}
		lastErr = err

		if !isRetryableError(err) {
			return "", err
		}
	}
	return "", fmt.Errorf("failed after %d retries: %w", r.config.MaxRetries, lastErr)
}

// Close implements the Client interface
func (r *RetryClient) Close() error {
	return r.client.Close()
}

// GetCapabilities returns the list of capabilities supported by this client.
func (r *RetryClient) GetCapabilities() []TaskCapability {
	return r.client.GetCapabilities()
}

// calculateDelay calculates the delay for a given retry attempt using exponential backoff with jitter
func (r *RetryClient) calculateDelay(attempt int) time.Duration {
	// Calculate exponential backoff: InitialDelay * (BackoffMultiplier ^ (attempt - 1))
	delay := float64(r.config.InitialDelay) * math.Pow(r.config.BackoffMultiplier, float64(attempt-1))

	// Cap at MaxDelay
	if delay > float64(r.config.MaxDelay) {
		delay = float64(r.config.MaxDelay)
	}

	// Add jitter to prevent thundering herd problem
	// Jitter adds a random value between -JitterFraction*delay and +JitterFraction*delay
	if r.config.JitterFraction > 0 {
		jitterRange := delay * r.config.JitterFraction
		// rand.Float64() returns [0.0, 1.0), so (rand.Float64()*2 - 1) returns [-1.0, 1.0)
		jitter := jitterRange * (rand.Float64()*2 - 1)
		delay += jitter
		// Ensure delay doesn't go negative
		if delay < 0 {
			delay = 0
		}
	}

	return time.Duration(delay)
}

// isRetryableError determines if an error is retryable
func isRetryableError(err error) bool {
	if err == nil {
		return false
	}

	// Rate limit errors should be retried
	var rateLimitErr *RateLimitError
	if errors.As(err, &rateLimitErr) {
		return true
	}

	// Check for standard rate limit error
	if errors.Is(err, ErrRateLimit) {
		return true
	}

	// Check error message for common retryable patterns
	errMsg := strings.ToLower(err.Error())

	// HTTP 5xx errors (server errors)
	retryablePatterns := []string{
		"500", "internal server error",
		"502", "bad gateway",
		"503", "service unavailable",
		"504", "gateway timeout",
		"timeout",
		"connection reset",
		"connection refused",
		"temporary failure",
		"rate limit",
		"too many requests",
		"429",
	}

	for _, pattern := range retryablePatterns {
		if strings.Contains(errMsg, pattern) {
			return true
		}
	}

	// Check for HTTP status codes if available
	// This will work with errors that wrap http.Response
	type httpErrorWithStatusCode interface {
		HTTPStatusCode() int
	}

	if httpErr, ok := err.(httpErrorWithStatusCode); ok {
		statusCode := httpErr.HTTPStatusCode()
		// Retry on 5xx errors and 429 (rate limit)
		if statusCode >= 500 || statusCode == http.StatusTooManyRequests {
			return true
		}
	}

	return false
}
