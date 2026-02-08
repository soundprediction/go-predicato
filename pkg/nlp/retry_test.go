package nlp

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/soundprediction/predicato/pkg/types"
)

// mockClient is a mock LLM client for testing
type mockClient struct {
	callCount        int
	failUntilCall    int
	errorToReturn    error
	responseToReturn *types.Response
}

func (m *mockClient) Chat(ctx context.Context, messages []types.Message) (*types.Response, error) {
	m.callCount++
	if m.callCount <= m.failUntilCall {
		return nil, m.errorToReturn
	}
	if m.responseToReturn != nil {
		return m.responseToReturn, nil
	}
	return &types.Response{Content: "success"}, nil
}

func (m *mockClient) ChatWithStructuredOutput(ctx context.Context, messages []types.Message, schema any) (*types.Response, error) {
	m.callCount++
	if m.callCount <= m.failUntilCall {
		return nil, m.errorToReturn
	}
	return &types.Response{Content: `{"status": "success"}`}, nil
}

func (m *mockClient) Close() error {
	return nil
}

func (m *mockClient) ExtractEntities(ctx context.Context, text string, entityTypes []string) ([]ExtractedEntity, error) {
	m.callCount++
	if m.callCount <= m.failUntilCall {
		return nil, m.errorToReturn
	}
	return []ExtractedEntity{}, nil
}

func (m *mockClient) ExtractRelations(ctx context.Context, text string, relationTypes []string) ([]ExtractedRelation, error) {
	m.callCount++
	if m.callCount <= m.failUntilCall {
		return nil, m.errorToReturn
	}
	return []ExtractedRelation{}, nil
}

func (m *mockClient) Summarize(ctx context.Context, text string) (string, error) {
	m.callCount++
	if m.callCount <= m.failUntilCall {
		return "", m.errorToReturn
	}
	if m.responseToReturn != nil {
		return m.responseToReturn.Content, nil
	}
	return "summary", nil
}

func (m *mockClient) GenerateText(ctx context.Context, prompt string) (string, error) {
	m.callCount++
	if m.callCount <= m.failUntilCall {
		return "", m.errorToReturn
	}
	if m.responseToReturn != nil {
		return m.responseToReturn.Content, nil
	}
	return "generated text", nil
}

func (m *mockClient) ExtractExtended(ctx context.Context, text string, entityTypes, relationTypes []string) (*ExtendedExtractionResult, error) {
	m.callCount++
	if m.callCount <= m.failUntilCall {
		return nil, m.errorToReturn
	}
	return &ExtendedExtractionResult{}, nil
}

func (m *mockClient) GetCapabilities() []TaskCapability {
	return []TaskCapability{TaskTextGeneration}
}

func (m *mockClient) GetModel() string {
	return "mock-model"
}

func TestRetryClient_SuccessOnFirstAttempt(t *testing.T) {
	mock := &mockClient{
		failUntilCall: 0,
	}

	config := &RetryConfig{
		MaxRetries:        3,
		InitialDelay:      10 * time.Millisecond,
		MaxDelay:          100 * time.Millisecond,
		BackoffMultiplier: 2.0,
	}

	retryClient, _ := NewRetryClient(mock, config)

	resp, err := retryClient.Chat(context.Background(), []types.Message{{Role: RoleUser, Content: "test"}})
	if err != nil {
		t.Fatalf("expected success, got error: %v", err)
	}

	if resp.Content != "success" {
		t.Errorf("expected content 'success', got '%s'", resp.Content)
	}

	if mock.callCount != 1 {
		t.Errorf("expected 1 call, got %d", mock.callCount)
	}
}

func TestRetryClient_SuccessAfterRetries(t *testing.T) {
	mock := &mockClient{
		failUntilCall: 2,
		errorToReturn: errors.New("500 internal server error"),
	}

	config := &RetryConfig{
		MaxRetries:        3,
		InitialDelay:      10 * time.Millisecond,
		MaxDelay:          100 * time.Millisecond,
		BackoffMultiplier: 2.0,
	}

	retryClient, _ := NewRetryClient(mock, config)

	start := time.Now()
	resp, err := retryClient.Chat(context.Background(), []types.Message{{Role: RoleUser, Content: "test"}})
	duration := time.Since(start)

	if err != nil {
		t.Fatalf("expected success after retries, got error: %v", err)
	}

	if resp.Content != "success" {
		t.Errorf("expected content 'success', got '%s'", resp.Content)
	}

	if mock.callCount != 3 {
		t.Errorf("expected 3 calls (1 initial + 2 retries), got %d", mock.callCount)
	}

	// Should have waited at least for the backoff delays
	// First retry: 10ms, Second retry: 20ms = total 30ms minimum
	if duration < 30*time.Millisecond {
		t.Errorf("expected at least 30ms duration for backoff, got %v", duration)
	}
}

func TestRetryClient_FailAfterMaxRetries(t *testing.T) {
	mock := &mockClient{
		failUntilCall: 10, // More than max retries
		errorToReturn: errors.New("503 service unavailable"),
	}

	config := &RetryConfig{
		MaxRetries:        3,
		InitialDelay:      10 * time.Millisecond,
		MaxDelay:          100 * time.Millisecond,
		BackoffMultiplier: 2.0,
	}

	retryClient, _ := NewRetryClient(mock, config)

	_, err := retryClient.Chat(context.Background(), []types.Message{{Role: RoleUser, Content: "test"}})
	if err == nil {
		t.Fatal("expected error after max retries, got nil")
	}

	if mock.callCount != 4 {
		t.Errorf("expected 4 calls (1 initial + 3 retries), got %d", mock.callCount)
	}

	if !errors.Is(err, mock.errorToReturn) {
		t.Errorf("expected error to wrap original error")
	}
}

func TestRetryClient_NonRetryableError(t *testing.T) {
	mock := &mockClient{
		failUntilCall: 10,
		errorToReturn: errors.New("400 bad request"), // Non-retryable
	}

	config := &RetryConfig{
		MaxRetries:        3,
		InitialDelay:      10 * time.Millisecond,
		MaxDelay:          100 * time.Millisecond,
		BackoffMultiplier: 2.0,
	}

	retryClient, _ := NewRetryClient(mock, config)

	_, err := retryClient.Chat(context.Background(), []types.Message{{Role: RoleUser, Content: "test"}})
	if err == nil {
		t.Fatal("expected error, got nil")
	}

	// Should fail immediately without retries
	if mock.callCount != 1 {
		t.Errorf("expected 1 call (no retries for non-retryable error), got %d", mock.callCount)
	}
}

func TestRetryClient_RateLimitError(t *testing.T) {
	mock := &mockClient{
		failUntilCall: 2,
		errorToReturn: NewRateLimitError("rate limit exceeded"),
	}

	config := &RetryConfig{
		MaxRetries:        3,
		InitialDelay:      10 * time.Millisecond,
		MaxDelay:          100 * time.Millisecond,
		BackoffMultiplier: 2.0,
	}

	retryClient, _ := NewRetryClient(mock, config)

	resp, err := retryClient.Chat(context.Background(), []types.Message{{Role: RoleUser, Content: "test"}})
	if err != nil {
		t.Fatalf("expected success after retries, got error: %v", err)
	}

	if resp.Content != "success" {
		t.Errorf("expected content 'success', got '%s'", resp.Content)
	}

	if mock.callCount != 3 {
		t.Errorf("expected 3 calls, got %d", mock.callCount)
	}
}

func TestRetryClient_ContextCancellation(t *testing.T) {
	mock := &mockClient{
		failUntilCall: 10,
		errorToReturn: errors.New("500 internal server error"),
	}

	config := &RetryConfig{
		MaxRetries:        5,
		InitialDelay:      100 * time.Millisecond,
		MaxDelay:          1 * time.Second,
		BackoffMultiplier: 2.0,
	}

	retryClient, _ := NewRetryClient(mock, config)

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := retryClient.Chat(ctx, []types.Message{{Role: RoleUser, Content: "test"}})
	if err == nil {
		t.Fatal("expected error due to context cancellation, got nil")
	}

	// Should have attempted at least once, but not completed all retries
	if mock.callCount >= 6 {
		t.Errorf("expected fewer than 6 calls due to context cancellation, got %d", mock.callCount)
	}
}

func TestIsRetryableError(t *testing.T) {
	tests := []struct {
		name      string
		err       error
		retryable bool
	}{
		{"nil error", nil, false},
		{"500 error", errors.New("500 internal server error"), true},
		{"502 error", errors.New("502 bad gateway"), true},
		{"503 error", errors.New("503 service unavailable"), true},
		{"504 error", errors.New("504 gateway timeout"), true},
		{"timeout", errors.New("connection timeout"), true},
		{"rate limit", errors.New("rate limit exceeded"), true},
		{"429 error", errors.New("429 too many requests"), true},
		{"400 error", errors.New("400 bad request"), false},
		{"401 error", errors.New("401 unauthorized"), false},
		{"403 error", errors.New("403 forbidden"), false},
		{"404 error", errors.New("404 not found"), false},
		{"rate limit error type", NewRateLimitError(), true},
		{"refusal error", NewRefusalError("refused"), false},
		{"connection reset", errors.New("connection reset by peer"), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isRetryableError(tt.err)
			if result != tt.retryable {
				t.Errorf("isRetryableError(%v) = %v, want %v", tt.err, result, tt.retryable)
			}
		})
	}
}

func TestRetryClient_ExponentialBackoff(t *testing.T) {
	config := &RetryConfig{
		MaxRetries:        5,
		InitialDelay:      100 * time.Millisecond,
		MaxDelay:          1 * time.Second,
		BackoffMultiplier: 2.0,
		JitterFraction:    0, // Disable jitter for predictable testing
	}

	// Use a mock client since NewRetryClient requires non-nil client
	mock := &mockClient{}
	retryClient, _ := NewRetryClient(mock, config)

	// Test delay calculation
	delays := []time.Duration{
		retryClient.calculateDelay(1), // First retry
		retryClient.calculateDelay(2), // Second retry
		retryClient.calculateDelay(3), // Third retry
		retryClient.calculateDelay(4), // Fourth retry
		retryClient.calculateDelay(5), // Fifth retry
	}

	expected := []time.Duration{
		100 * time.Millisecond,  // 100 * 2^0
		200 * time.Millisecond,  // 100 * 2^1
		400 * time.Millisecond,  // 100 * 2^2
		800 * time.Millisecond,  // 100 * 2^3
		1000 * time.Millisecond, // 100 * 2^4 = 1600, capped at MaxDelay
	}

	for i, delay := range delays {
		if delay != expected[i] {
			t.Errorf("delay[%d] = %v, want %v", i, delay, expected[i])
		}
	}
}

func TestRetryClient_ChatWithStructuredOutput(t *testing.T) {
	mock := &mockClient{
		failUntilCall: 2,
		errorToReturn: errors.New("500 internal server error"),
	}

	config := &RetryConfig{
		MaxRetries:        3,
		InitialDelay:      10 * time.Millisecond,
		MaxDelay:          100 * time.Millisecond,
		BackoffMultiplier: 2.0,
	}

	retryClient, _ := NewRetryClient(mock, config)

	result, err := retryClient.ChatWithStructuredOutput(
		context.Background(),
		[]types.Message{{Role: RoleUser, Content: "test"}},
		map[string]interface{}{"type": "object"},
	)

	if err != nil {
		t.Fatalf("expected success after retries, got error: %v", err)
	}

	if result.Content != `{"status": "success"}` {
		t.Errorf("unexpected result: %s", result.Content)
	}

	if mock.callCount != 3 {
		t.Errorf("expected 3 calls, got %d", mock.callCount)
	}
}

func TestDefaultRetryConfig(t *testing.T) {
	config := DefaultRetryConfig()

	if config.MaxRetries != 3 {
		t.Errorf("expected MaxRetries = 3, got %d", config.MaxRetries)
	}

	if config.InitialDelay != 1*time.Second {
		t.Errorf("expected InitialDelay = 1s, got %v", config.InitialDelay)
	}

	if config.MaxDelay != 60*time.Second {
		t.Errorf("expected MaxDelay = 60s, got %v", config.MaxDelay)
	}

	if config.BackoffMultiplier != 2.0 {
		t.Errorf("expected BackoffMultiplier = 2.0, got %f", config.BackoffMultiplier)
	}

	if config.JitterFraction != 0.1 {
		t.Errorf("expected JitterFraction = 0.1, got %f", config.JitterFraction)
	}
}

func TestNewRetryClient_NilClient(t *testing.T) {
	_, err := NewRetryClient(nil, nil)
	if err == nil {
		t.Error("expected error for nil client, got nil")
	}
	if err.Error() != "client cannot be nil" {
		t.Errorf("unexpected error message: %s", err.Error())
	}
}

// httpError implements httpErrorWithStatusCode for testing
type httpError struct {
	statusCode int
	message    string
}

func (e httpError) Error() string {
	return fmt.Sprintf("%d: %s", e.statusCode, e.message)
}

func (e httpError) HTTPStatusCode() int {
	return e.statusCode
}

func TestIsRetryableError_HTTPStatusCode(t *testing.T) {
	tests := []struct {
		name       string
		statusCode int
		retryable  bool
	}{
		{"500 internal server error", 500, true},
		{"502 bad gateway", 502, true},
		{"503 service unavailable", 503, true},
		{"504 gateway timeout", 504, true},
		{"429 rate limit", 429, true},
		{"400 bad request", 400, false},
		{"401 unauthorized", 401, false},
		{"403 forbidden", 403, false},
		{"404 not found", 404, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := httpError{statusCode: tt.statusCode, message: tt.name}
			result := isRetryableError(err)
			if result != tt.retryable {
				t.Errorf("isRetryableError(%v) = %v, want %v", err, result, tt.retryable)
			}
		})
	}
}
