package crossencoder

import (
	"context"
	"testing"
)

func TestMockRerankerClient(t *testing.T) {
	client := NewMockRerankerClient(DefaultConfig(ProviderMock))
	defer client.Close()

	ctx := context.Background()
	query := "artificial intelligence machine learning"
	passages := []string{
		"Machine learning is a subset of artificial intelligence",
		"The weather is nice today",
		"Deep learning models use neural networks",
		"Cats are cute animals",
		"AI and ML are transforming technology",
	}

	results, err := client.Rank(ctx, query, passages)
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if len(results) != len(passages) {
		t.Fatalf("Expected %d results, got %d", len(passages), len(results))
	}

	// Verify results are sorted by score (descending)
	for i := 1; i < len(results); i++ {
		if results[i-1].Score < results[i].Score {
			t.Errorf("Results not sorted by score: %f < %f", results[i-1].Score, results[i].Score)
		}
	}

	// The first passage should likely rank highest due to keyword overlap
	if results[0].Passage != "Machine learning is a subset of artificial intelligence" &&
		results[0].Passage != "AI and ML are transforming technology" {
		t.Logf("Top result: %s (score: %f)", results[0].Passage, results[0].Score)
		t.Logf("All results:")
		for i, result := range results {
			t.Logf("  %d. %s (%.3f)", i+1, result.Passage, result.Score)
		}
	}
}

func TestLocalRerankerClient(t *testing.T) {
	client := NewLocalRerankerClient(DefaultConfig(ProviderLocal))
	defer client.Close()

	ctx := context.Background()
	query := "machine learning algorithms"
	passages := []string{
		"Machine learning algorithms are used in data science",
		"Cooking recipes for dinner tonight",
		"Neural networks and deep learning",
		"Sports scores and statistics",
		"Supervised learning algorithms like decision trees",
	}

	results, err := client.Rank(ctx, query, passages)
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if len(results) != len(passages) {
		t.Fatalf("Expected %d results, got %d", len(passages), len(results))
	}

	// Verify results are sorted by score (descending)
	for i := 1; i < len(results); i++ {
		if results[i-1].Score < results[i].Score {
			t.Errorf("Results not sorted by score: %f < %f", results[i-1].Score, results[i].Score)
		}
	}
}

func TestEmptyPassages(t *testing.T) {
	client := NewMockRerankerClient(DefaultConfig(ProviderMock))
	defer client.Close()

	ctx := context.Background()
	results, err := client.Rank(ctx, "test query", []string{})
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	if len(results) != 0 {
		t.Fatalf("Expected 0 results for empty passages, got %d", len(results))
	}
}

func TestNewClient(t *testing.T) {
	tests := []struct {
		name         string
		config       ClientConfig
		expectError  bool
		expectedType string
	}{
		{
			name: "mock provider",
			config: ClientConfig{
				Provider: ProviderMock,
				Config:   DefaultConfig(ProviderMock),
			},
			expectError:  false,
			expectedType: "*crossencoder.MockRerankerClient",
		},
		{
			name: "local provider",
			config: ClientConfig{
				Provider: ProviderLocal,
				Config:   DefaultConfig(ProviderLocal),
			},
			expectError:  false,
			expectedType: "*crossencoder.LocalRerankerClient",
		},
		{
			name: "openai provider without llm client",
			config: ClientConfig{
				Provider: ProviderOpenAI,
				Config:   DefaultConfig(ProviderOpenAI),
			},
			expectError: true,
		},
		{
			name: "candle provider without embedder client",
			config: ClientConfig{
				Provider: ProviderCandle,
				Config:   DefaultConfig(ProviderCandle),
			},
			expectError: true,
		},
		{
			name: "unknown provider",
			config: ClientConfig{
				Provider: "unknown",
				Config:   Config{},
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewClient(tt.config)

			if tt.expectError {
				if err == nil {
					t.Errorf("Expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Errorf("Expected no error, got: %v", err)
				return
			}

			if client == nil {
				t.Errorf("Expected client, got nil")
				return
			}

			// Clean up
			client.Close()
		})
	}
}

func TestDefaultConfig(t *testing.T) {
	tests := []struct {
		provider Provider
		expected Config
	}{
		{
			provider: ProviderOpenAI,
			expected: Config{
				Model:          "gpt-4o-mini",
				BatchSize:      10,
				MaxConcurrency: 5,
			},
		},
		{
			provider: ProviderLocal,
			expected: Config{
				BatchSize: 100,
			},
		},
		{
			provider: ProviderMock,
			expected: Config{
				BatchSize: 100,
			},
		},
		{
			provider: ProviderCandle,
			expected: Config{
				Model:          "qwen/qwen3-embedding-0.6b",
				BatchSize:      100,
				MaxConcurrency: 1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(string(tt.provider), func(t *testing.T) {
			config := DefaultConfig(tt.provider)

			if config.Model != tt.expected.Model {
				t.Errorf("Expected model %s, got %s", tt.expected.Model, config.Model)
			}
			if config.BatchSize != tt.expected.BatchSize {
				t.Errorf("Expected batch size %d, got %d", tt.expected.BatchSize, config.BatchSize)
			}
			if config.MaxConcurrency != tt.expected.MaxConcurrency {
				t.Errorf("Expected max concurrency %d, got %d", tt.expected.MaxConcurrency, config.MaxConcurrency)
			}
		})
	}
}

// containsAny checks if s contains any of the substrings
func containsAny(s string, substrings []string) bool {
	for _, sub := range substrings {
		if contains(s, sub) {
			return true
		}
	}
	return false
}

// contains checks if s contains substr (case-insensitive)
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr ||
		len(substr) == 0 ||
		(len(s) > 0 && len(substr) > 0 && containsLower(toLower(s), toLower(substr))))
}

func toLower(s string) string {
	b := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		b[i] = c
	}
	return string(b)
}

func containsLower(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// Benchmark tests
func BenchmarkMockReranker(b *testing.B) {
	client := NewMockRerankerClient(DefaultConfig(ProviderMock))
	defer client.Close()

	ctx := context.Background()
	query := "machine learning artificial intelligence"
	passages := []string{
		"Machine learning is a subset of artificial intelligence",
		"Deep learning models use neural networks",
		"Natural language processing applications",
		"Computer vision and image recognition",
		"Reinforcement learning algorithms",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := client.Rank(ctx, query, passages)
		if err != nil {
			b.Fatalf("Benchmark failed: %v", err)
		}
	}
}

func BenchmarkLocalReranker(b *testing.B) {
	client := NewLocalRerankerClient(DefaultConfig(ProviderLocal))
	defer client.Close()

	ctx := context.Background()
	query := "machine learning artificial intelligence"
	passages := []string{
		"Machine learning is a subset of artificial intelligence",
		"Deep learning models use neural networks",
		"Natural language processing applications",
		"Computer vision and image recognition",
		"Reinforcement learning algorithms",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := client.Rank(ctx, query, passages)
		if err != nil {
			b.Fatalf("Benchmark failed: %v", err)
		}
	}
}
