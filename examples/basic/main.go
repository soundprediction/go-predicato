// Package main demonstrates basic usage of predicato with internal services.
//
// This example shows how to:
// - Create and configure a Predicato client using only internal services
// - Use Ladybug embedded database (no external database server)
// - Use go-candle SmolLM2 for text generation (no external LLM API)
// - Use go-candle with qwen3-embedding-0.6b for embeddings (no external API)
// - Use go-candle embedding-based reranking for reranking search results
// - Add episodes (data) to the knowledge graph
// - Search and rerank results from the knowledge graph
//
// Prerequisites:
// - CGO enabled (required for Rust FFI bindings)
// - No API keys or external services required!
//
// First run will download models from HuggingFace Hub.
//
// Memory Requirements:
// - Minimum 4GB RAM recommended
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/soundprediction/predicato"
	candleAdapter "github.com/soundprediction/predicato/pkg/candle"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/types"
)

func main() {
	ctx := context.Background()

	fmt.Println("================================================================================")
	fmt.Println("Predicato Basic Example - Internal Services Stack")
	fmt.Println("================================================================================")
	fmt.Println()
	fmt.Println("This example uses predicato's internal services:")
	fmt.Println("  - Ladybug: embedded graph database (no server required)")
	fmt.Println("  - Candle SmolLM2: local text generation (no API required)")
	fmt.Println("  - Candle: local embeddings with qwen/qwen3-embedding-0.6b")
	fmt.Println("  - Candle: local embedding-based reranking")
	fmt.Println()
	fmt.Println("No API keys or external services needed!")
	fmt.Println()

	// ========================================
	// 1. Create Ladybug Driver (Embedded Graph Database)
	// ========================================
	fmt.Println("[1/5] Setting up Ladybug embedded graph database...")

	ladybugDriver, err := driver.NewLadybugDriver("./example_graph.db", 1)
	if err != nil {
		log.Fatalf("Failed to create Ladybug driver: %v", err)
	}
	defer func() {
		if err := ladybugDriver.Close(); err != nil {
			log.Printf("Error closing Ladybug driver: %v", err)
		}
	}()
	fmt.Println("      Ladybug driver created (embedded database at ./example_graph.db)")

	// ========================================
	// 2. Create Candle Client (Local Text Generation)
	// ========================================
	fmt.Println("[2/5] Setting up Candle SmolLM2 for text generation...")
	fmt.Println("      (First run will download the model, please wait...)")

	candleClient := candleAdapter.NewClient(candleAdapter.CandleNLPConfig{
		TextGenModelID: "HuggingFaceTB/SmolLM2-360M-Instruct",
	})
	defer candleClient.Close()

	// Create LLM adapter for nlp.Client interface
	llmClient := candleAdapter.NewLLMAdapter(candleClient, "text_generation")
	fmt.Println("      Candle text generation client created")

	// ========================================
	// 3. Create Embedder Client (Local Embeddings)
	// ========================================
	fmt.Println("[3/5] Setting up Candle embedder with qwen/qwen3-embedding-0.6b...")
	fmt.Println("      (First run will download the model, please wait...)")

	embedderClient, err := candleAdapter.NewCandleEmbedderClient(&candleAdapter.CandleEmbedderConfig{
		Model:      "qwen/qwen3-embedding-0.6b",
		Dimensions: 1024,
		Normalize:  true,
	})
	if err != nil {
		log.Fatalf("Failed to create embedder client: %v", err)
	}
	defer embedderClient.Close()
	fmt.Println("      Candle embedder created (model: qwen/qwen3-embedding-0.6b)")

	// ========================================
	// 4. Create Reranker Client (Local Reranking)
	// ========================================
	fmt.Println("[4/5] Setting up Candle embedding-based reranker...")

	rerankerClient := candleAdapter.NewCandleRerankerClient(embedderClient)
	// rerankerClient shares the embedder, so don't close it separately
	fmt.Println("      Candle reranker created (embedding-based cosine similarity)")

	// ========================================
	// 5. Create Predicato Client
	// ========================================
	fmt.Println("[5/5] Creating Predicato client...")

	predicatoConfig := &predicato.Config{
		GroupID:  "example-group",
		TimeZone: time.UTC,
	}

	client, err := predicato.NewClient(ladybugDriver, llmClient, embedderClient, predicatoConfig, nil)
	if err != nil {
		log.Fatalf("Failed to create Predicato client: %v", err)
	}
	defer client.Close(ctx)
	fmt.Printf("      Predicato client created (group: %s)\n", predicatoConfig.GroupID)

	fmt.Println()
	fmt.Println("================================================================================")
	fmt.Println("All components initialized successfully!")
	fmt.Println("================================================================================")
	fmt.Println()

	// ========================================
	// Example: Add Episodes
	// ========================================
	fmt.Println("Adding sample episodes to the knowledge graph...")

	episodes := []types.Episode{
		{
			ID:        "episode-1",
			Name:      "Meeting with Alice",
			Content:   "Had a productive meeting with Alice about the new project. She mentioned that the deadline is next month and we need to focus on the API design.",
			Reference: time.Now().Add(-24 * time.Hour),
			CreatedAt: time.Now(),
			GroupID:   "example-group",
			Metadata: map[string]interface{}{
				"type": "meeting",
			},
		},
		{
			ID:        "episode-2",
			Name:      "Project Research",
			Content:   "Researched various approaches for implementing the API. Found that GraphQL might be a good fit for our use case due to its flexibility.",
			Reference: time.Now().Add(-12 * time.Hour),
			CreatedAt: time.Now(),
			GroupID:   "example-group",
			Metadata: map[string]interface{}{
				"type": "research",
			},
		},
		{
			ID:        "episode-3",
			Name:      "Technical Discussion",
			Content:   "Discussed REST vs GraphQL with the team. Bob prefers REST for simplicity, but Carol thinks GraphQL's type system would help with API documentation.",
			Reference: time.Now().Add(-6 * time.Hour),
			CreatedAt: time.Now(),
			GroupID:   "example-group",
			Metadata: map[string]interface{}{
				"type": "discussion",
			},
		},
	}

	if _, err := client.Add(ctx, episodes, nil); err != nil {
		fmt.Printf("Warning: Add operation failed: %v\n", err)
		fmt.Println("This may be expected if Ladybug implementation is still in development.")
	} else {
		fmt.Printf("Added %d episodes to the knowledge graph\n", len(episodes))
	}

	// ========================================
	// Example: Search the Knowledge Graph
	// ========================================
	fmt.Println()
	fmt.Println("Searching the knowledge graph for: \"API design and deadlines\"")

	searchConfig := &types.SearchConfig{
		Limit:              10,
		CenterNodeDistance: 2,
		MinScore:           0.0,
		IncludeEdges:       true,
		Rerank:             false, // We'll do manual reranking to demonstrate
	}

	results, err := client.Search(ctx, "API design and deadlines", searchConfig)
	if err != nil {
		fmt.Printf("Warning: Search operation failed: %v\n", err)
		fmt.Println("This may be expected if Ladybug implementation is still in development.")
	} else if results != nil && len(results.Nodes) > 0 {
		fmt.Printf("Found %d nodes and %d edges\n", len(results.Nodes), len(results.Edges))

		fmt.Println()
		fmt.Println("Search results (before reranking):")
		fmt.Println("----------------------------------")
		for i, node := range results.Nodes {
			if i >= 5 {
				break
			}
			summary := truncateString(node.Summary, 80)
			fmt.Printf("  %d. %s (%s)\n", i+1, node.Name, node.Type)
			fmt.Printf("     %s\n", summary)
		}

		// ========================================
		// Example: Rerank Search Results
		// ========================================
		fmt.Println()
		fmt.Println("Reranking results with Candle embedding-based reranker...")

		// Extract passages for reranking
		passages := make([]string, len(results.Nodes))
		for i, node := range results.Nodes {
			passages[i] = node.Summary
		}

		// Rerank
		rankedResults, err := rerankerClient.Rank(ctx, "API design and deadlines", passages)
		if err != nil {
			fmt.Printf("Warning: Reranking failed: %v\n", err)
		} else {
			fmt.Println()
			fmt.Println("Search results (after reranking):")
			fmt.Println("---------------------------------")
			for i, ranked := range rankedResults {
				if i >= 5 {
					break
				}
				passage := truncateString(ranked.Passage, 80)
				fmt.Printf("  %d. (score: %.3f) %s\n", i+1, ranked.Score, passage)
			}
		}
	} else {
		fmt.Println("No results found (knowledge graph may be empty)")
	}

	// ========================================
	// Example: Text Generation
	// ========================================
	fmt.Println()
	fmt.Println("Demonstrating text generation with Candle SmolLM2...")

	prompt := "The advantages of using a knowledge graph are"
	generated, err := candleClient.GenerateText(ctx, prompt)
	if err != nil {
		fmt.Printf("Warning: Text generation failed: %v\n", err)
	} else {
		fmt.Printf("Prompt: %s\n", prompt)
		fmt.Printf("Generated: %s\n", truncateString(generated, 200))
	}

	fmt.Println()
	fmt.Println("================================================================================")
	fmt.Println("Example completed successfully!")
	fmt.Println("================================================================================")
	fmt.Println()
	fmt.Println("Summary:")
	fmt.Println("  - Used Ladybug embedded database (no Neo4j server)")
	fmt.Println("  - Used Candle SmolLM2 for text generation (no OpenAI API)")
	fmt.Println("  - Used Candle qwen/qwen3-embedding-0.6b for embeddings (no API)")
	fmt.Println("  - Used Candle embedding-based reranking (no API)")
	fmt.Println()
	fmt.Println("For external API examples, see: examples/external_apis/")
}

// truncateString truncates a string to a maximum length with ellipsis
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
