package crossencoder_test

import (
	"context"
	"fmt"
	"log"

	"github.com/soundprediction/predicato/pkg/crossencoder"
	"github.com/soundprediction/predicato/pkg/nlp"
)

// ExampleNewClient demonstrates how to create different types of cross-encoder clients
func ExampleNewClient() {
	// Mock client for testing
	mockClient, err := crossencoder.NewClient(crossencoder.ClientConfig{
		Provider: crossencoder.ProviderMock,
		Config:   crossencoder.DefaultConfig(crossencoder.ProviderMock),
	})
	if err != nil {
		log.Fatal(err)
	}
	defer mockClient.Close()

	// Local similarity client
	localClient, err := crossencoder.NewClient(crossencoder.ClientConfig{
		Provider: crossencoder.ProviderLocal,
		Config:   crossencoder.DefaultConfig(crossencoder.ProviderLocal),
	})
	if err != nil {
		log.Fatal(err)
	}
	defer localClient.Close()

	fmt.Println("Created mock and local cross-encoder clients")
	// Output: Created mock and local cross-encoder clients
}

// ExampleMockRerankerClient demonstrates basic usage of the mock reranker
func ExampleMockRerankerClient() {
	client := crossencoder.NewMockRerankerClient(crossencoder.DefaultConfig(crossencoder.ProviderMock))
	defer client.Close()

	ctx := context.Background()
	query := "machine learning algorithms"
	passages := []string{
		"Machine learning algorithms are used for data analysis",
		"Cooking recipes and meal preparation",
		"Deep learning neural networks and AI",
		"Weather forecast and climate data",
		"Supervised learning and decision trees",
	}

	results, err := client.Rank(ctx, query, passages)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Ranked %d passages\n", len(results))
	fmt.Printf("Top result has score > 0: %t\n", results[0].Score > 0)
	fmt.Printf("Results are sorted: %t\n", results[0].Score >= results[1].Score)
	// Output:
	// Ranked 5 passages
	// Top result has score > 0: true
	// Results are sorted: true
}

// ExampleLocalRerankerClient demonstrates the local similarity reranker
func ExampleLocalRerankerClient() {
	client := crossencoder.NewLocalRerankerClient(crossencoder.DefaultConfig(crossencoder.ProviderLocal))
	defer client.Close()

	ctx := context.Background()
	query := "artificial intelligence neural networks"
	passages := []string{
		"Artificial intelligence uses neural networks for complex tasks",
		"Traditional cooking methods and recipes",
		"Machine learning and deep neural networks",
		"Sports statistics and game analysis",
	}

	results, err := client.Rank(ctx, query, passages)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Top passage has positive score: %t\n", results[0].Score > 0)
	// Output: Top passage has positive score: true
}

// ExampleOpenAIRerankerClient demonstrates how to use OpenAI reranker (requires API key)
func ExampleOpenAIRerankerClient() {
	// Note: This example requires a valid OpenAI API key
	// In practice, you would get this from environment variables or configuration

	// Create LLM client (example - replace with actual implementation)
	llmConfig := nlp.Config{
		Model: "gpt-4o-mini",
	}

	// This is a conceptual example - actual implementation depends on your LLM client
	// nlProcessor := nlp.NewOpenAIClient("your-api-key", llmConfig)
	//
	// reranker := crossencoder.NewOpenAIRerankerClient(nlProcessor, crossencoder.Config{
	//     MaxConcurrency: 3,
	// })
	// defer reranker.Close()
	//
	// ctx := context.Background()
	// results, err := reranker.Rank(ctx, "search query", passages)

	fmt.Printf("OpenAI reranker config: %+v\n", llmConfig)
	// Output: OpenAI reranker config: {Temperature:<nil> MaxTokens:<nil> TopP:<nil> TopK:<nil> MinP:<nil> Model:gpt-4o-mini BaseURL: Stop:[] InsecureSkipVerify:false}
}

// ExampleRankedPassage demonstrates working with ranked results
func ExampleRankedPassage() {
	client := crossencoder.NewMockRerankerClient(crossencoder.DefaultConfig(crossencoder.ProviderMock))
	defer client.Close()

	ctx := context.Background()
	query := "data science"
	passages := []string{
		"Data science involves statistical analysis",
		"Cooking involves following recipes",
		"Machine learning is part of data science",
	}

	results, err := client.Rank(ctx, query, passages)
	if err != nil {
		log.Fatal(err)
	}

	// Access individual results
	topResult := results[0]
	fmt.Printf("Top result score: %.3f\n", topResult.Score)
	fmt.Printf("Results are sorted: %t\n", results[0].Score >= results[1].Score)

	// Output:
	// Top result score: 0.900
	// Results are sorted: true
}
