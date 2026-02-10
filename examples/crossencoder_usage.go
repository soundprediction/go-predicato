package main

import (
	"context"
	"fmt"
	"log"

	candleAdapter "github.com/soundprediction/predicato/pkg/candle"
	"github.com/soundprediction/predicato/pkg/crossencoder"
)

// Example demonstrating how to use cross-encoder reranking in predicato.
//
// This example prioritizes internal reranking (go-candle) which requires
// no external services, followed by external API options for production deployments.
//
// Internal Services (Recommended for getting started):
// - Candle with qwen/qwen3-embedding-0.6b (no API key, runs locally)
//
// External APIs (For production/cloud deployments):
// - vLLM, LocalAI, Jina AI (require running servers or API keys)
func main() {
	ctx := context.Background()

	// Example passages to rank
	query := "What is machine learning?"
	passages := []string{
		"Machine learning is a subset of artificial intelligence that uses algorithms to learn patterns from data.",
		"Cooking involves combining ingredients and applying heat to create delicious meals.",
		"Deep learning uses neural networks with multiple layers to solve complex problems.",
		"Weather patterns are influenced by atmospheric pressure, temperature, and humidity.",
		"Supervised learning trains models on labeled data to make predictions on new data.",
	}

	fmt.Println("================================================================================")
	fmt.Println("Cross-Encoder Reranking Examples")
	fmt.Println("================================================================================")
	fmt.Printf("\nQuery: %s\n", query)
	fmt.Printf("Passages to rank: %d\n\n", len(passages))

	// ========================================
	// Example 1: RECOMMENDED - Using Candle Embeddings (Internal, No API Required)
	// ========================================
	fmt.Println("1. [RECOMMENDED] Candle Embedding-Based Reranker (Internal - No API Required)")
	fmt.Println("   Model: qwen/qwen3-embedding-0.6b")
	fmt.Println("   (First run will download the model from HuggingFace Hub)")
	fmt.Println()

	embedder, err := candleAdapter.NewCandleEmbedderClient(&candleAdapter.CandleEmbedderConfig{
		Model:      "qwen/qwen3-embedding-0.6b",
		Dimensions: 1024,
		Normalize:  true,
	})
	if err != nil {
		fmt.Printf("   Warning: Failed to create Candle embedder: %v\n", err)
		fmt.Println("   This may happen if CGO is not enabled or Rust libraries are not available.")
		fmt.Println("   Enable CGO with: export CGO_ENABLED=1")
	} else {
		defer embedder.Close()

		candleReranker := candleAdapter.NewCandleRerankerClient(embedder)

		results, err := candleReranker.Rank(ctx, query, passages)
		if err != nil {
			fmt.Printf("   Warning: Candle reranking failed: %v\n", err)
		} else {
			fmt.Println("   Candle reranking successful:")
			for i, result := range results {
				passagePreview := result.Passage
				if len(passagePreview) > 60 {
					passagePreview = passagePreview[:60] + "..."
				}
				fmt.Printf("   %d. (Score: %.3f) %s\n", i+1, result.Score, passagePreview)
			}
		}
	}
	fmt.Println()

	// ========================================
	// Example 2: Local Fallback Reranker (No ML, Term Frequency Based)
	// ========================================
	fmt.Println("2. Local Fallback Reranker (No External Dependencies)")
	fmt.Println("   Uses term frequency / cosine similarity (no ML model)")
	fmt.Println()

	localReranker := crossencoder.NewLocalRerankerClient(crossencoder.Config{
		BatchSize: 100,
	})
	defer localReranker.Close()

	results, err := localReranker.Rank(ctx, query, passages)
	if err != nil {
		fmt.Printf("   Warning: Local reranking failed: %v\n", err)
	} else {
		fmt.Println("   Local reranking successful:")
		for i, result := range results {
			passagePreview := result.Passage
			if len(passagePreview) > 60 {
				passagePreview = passagePreview[:60] + "..."
			}
			fmt.Printf("   %d. (Score: %.3f) %s\n", i+1, result.Score, passagePreview)
		}
	}
	fmt.Println()

	// ========================================
	// External API Examples (Require Running Servers or API Keys)
	// ========================================
	fmt.Println("================================================================================")
	fmt.Println("External API Rerankers (For Production/Cloud Deployments)")
	fmt.Println("================================================================================")
	fmt.Println()

	// Example 3: Using vLLM with a cross-encoder model
	fmt.Println("3. vLLM Reranker (Requires vLLM Server)")
	fmt.Println("   Start with: vllm serve BAAI/bge-reranker-large")
	vllmReranker := crossencoder.NewVLLMRerankerClient(
		"http://localhost:8000/v1", // vLLM server URL
		"BAAI/bge-reranker-large",  // Cross-encoder model
	)
	defer vllmReranker.Close()

	results, err = vllmReranker.Rank(ctx, query, passages)
	if err != nil {
		fmt.Printf("   Warning: vLLM reranking failed (server may not be running): %v\n", err)
	} else {
		fmt.Println("   vLLM reranking successful:")
		for i, result := range results {
			passagePreview := result.Passage
			if len(passagePreview) > 60 {
				passagePreview = passagePreview[:60] + "..."
			}
			fmt.Printf("   %d. (Score: %.3f) %s\n", i+1, result.Score, passagePreview)
		}
	}
	fmt.Println()

	// Example 4: Using Jina AI reranking service
	fmt.Println("4. Jina AI Reranker (Requires API Key)")
	fmt.Println("   Get API key from: https://jina.ai/")
	jinaReranker := crossencoder.NewJinaRerankerClient(
		"your-jina-api-key",        // API key from Jina AI
		"jina-reranker-v1-base-en", // Jina reranker model
	)
	defer jinaReranker.Close()

	results, err = jinaReranker.Rank(ctx, query, passages)
	if err != nil {
		fmt.Printf("   Warning: Jina reranking failed (API key may be invalid): %v\n", err)
	} else {
		fmt.Println("   Jina reranking successful:")
		for i, result := range results {
			passagePreview := result.Passage
			if len(passagePreview) > 60 {
				passagePreview = passagePreview[:60] + "..."
			}
			fmt.Printf("   %d. (Score: %.3f) %s\n", i+1, result.Score, passagePreview)
		}
	}
	fmt.Println()

	// Example 5: Using LocalAI reranking service
	fmt.Println("5. LocalAI Reranker (Requires LocalAI Server)")
	fmt.Println("   Start with: local-ai run reranker")
	localAIReranker := crossencoder.NewLocalAIRerankerClient(
		"http://localhost:8080/v1", // LocalAI server URL
		"reranker",                 // Model name configured in LocalAI
		"",                         // API key (empty if not required)
	)
	defer localAIReranker.Close()

	results, err = localAIReranker.Rank(ctx, query, passages)
	if err != nil {
		fmt.Printf("   Warning: LocalAI reranking failed (server may not be running): %v\n", err)
	} else {
		fmt.Println("   LocalAI reranking successful:")
		for i, result := range results {
			passagePreview := result.Passage
			if len(passagePreview) > 60 {
				passagePreview = passagePreview[:60] + "..."
			}
			fmt.Printf("   %d. (Score: %.3f) %s\n", i+1, result.Score, passagePreview)
		}
	}

	// Summary
	fmt.Println()
	fmt.Println("================================================================================")
	fmt.Println("Summary")
	fmt.Println("================================================================================")
	fmt.Println()
	fmt.Println("Recommended Approach:")
	fmt.Println("  1. Start with Candle (qwen/qwen3-embedding-0.6b) - no API needed")
	fmt.Println("  2. Use Local Fallback if CGO is not available")
	fmt.Println("  3. Upgrade to external APIs (vLLM, Jina) for production scale")
	fmt.Println()
	fmt.Println("For more examples, see:")
	fmt.Println("  - examples/basic/ - Full internal services example")
	fmt.Println("  - examples/chat/ - Interactive chat with reranking")
	fmt.Println("  - examples/external_apis/ - External API integration")
}

// demonstrateSearchIntegration shows how the reranker would be integrated
// into a larger search system
func demonstrateSearchIntegration() {
	ctx := context.Background()

	query := "artificial intelligence applications"
	initialResults := []string{
		"AI is used in healthcare for medical diagnosis and drug discovery.",
		"The weather today is sunny with a chance of rain.",
		"Machine learning algorithms power recommendation systems.",
		"Cooking recipes often include step-by-step instructions.",
		"Natural language processing enables chatbots and translation.",
	}

	// Recommended: Use Candle for internal reranking
	embedder, err := candleAdapter.NewCandleEmbedderClient(&candleAdapter.CandleEmbedderConfig{
		Model:      "qwen/qwen3-embedding-0.6b",
		Dimensions: 1024,
		Normalize:  true,
	})

	var reranker crossencoder.Client
	if err != nil {
		// Fallback to local reranker if Candle is not available
		log.Printf("Candle not available, using local fallback: %v", err)
		reranker = crossencoder.NewLocalRerankerClient(crossencoder.Config{})
	} else {
		reranker = candleAdapter.NewCandleRerankerClient(embedder)
	}
	defer reranker.Close()

	// Rerank the initial results
	rerankedResults, err := reranker.Rank(ctx, query, initialResults)
	if err != nil {
		log.Printf("Reranking failed, using original order: %v", err)
		return
	}

	fmt.Println("Reranked results (most relevant first):")
	for i, result := range rerankedResults {
		fmt.Printf("%d. (Score: %.3f) %s\n", i+1, result.Score, result.Passage)
	}
}
