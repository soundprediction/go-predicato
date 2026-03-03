// Package main demonstrates factstore RAG (Retrieval-Augmented Generation) usage.
//
// This example shows how to:
// - Configure a PostgreSQL factstore with VectorChord
// - Extract knowledge from text documents
// - Perform hybrid search (vector + keyword) with RRF fusion
// - Use search results for RAG applications
//
// Prerequisites:
// - PostgreSQL 15+ with VectorChord extension
// - An embedding model (this example uses a mock for simplicity)
//
// For production PostgreSQL setup, see docs/FACTSTORE_RAG.md
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/soundprediction/predicato/pkg/factstore"
	"github.com/soundprediction/predicato/pkg/types"
)

func main() {
	ctx := context.Background()

	fmt.Println("================================================================================")
	fmt.Println("Predicato FactStore RAG Example")
	fmt.Println("================================================================================")
	fmt.Println()
	fmt.Println("This example demonstrates:")
	fmt.Println("  - Storing extracted knowledge in a factstore")
	fmt.Println("  - Performing hybrid search (vector + keyword)")
	fmt.Println("  - Using search results for RAG applications")
	fmt.Println()

	// ========================================
	// 1. Configure FactStore Backend
	// ========================================
	// PostgreSQL with VectorChord for vector search

	var db factstore.FactsDB
	var err error

	// Check if PostgreSQL connection string is provided
	pgConnString := os.Getenv("FACTSTORE_POSTGRES_URL")
	if pgConnString != "" {
		fmt.Println("[1/4] Connecting to PostgreSQL with VectorChord...")
		db, err = factstore.NewPostgresDB(pgConnString, 1024) // 1024 dimensions for qwen3-embedding
		if err != nil {
			log.Fatalf("Failed to connect to PostgreSQL: %v", err)
		}
		fmt.Println("      Connected to PostgreSQL (VectorChord enabled)")
	} else {
		fmt.Println("[1/4] Using in-memory factstore (no PostgreSQL configured)...")
		fmt.Println("      Set FACTSTORE_POSTGRES_URL for production PostgreSQL backend")
		fmt.Println("      Example: FACTSTORE_POSTGRES_URL='postgres://user:pass@localhost:5432/facts'")
		fmt.Println()

		// For this example, we'll demonstrate with a mock/nil database
		// In real usage, you'd use PostgreSQL
		fmt.Println("      (Skipping database operations - demo mode)")
		demonstrateMockUsage()
		return
	}
	defer db.Close()

	// Initialize schema
	fmt.Println("[2/4] Initializing database schema...")
	if err := db.Initialize(ctx); err != nil {
		log.Fatalf("Failed to initialize database: %v", err)
	}
	fmt.Println("      Schema initialized")

	// ========================================
	// 2. Extract and Store Knowledge
	// ========================================
	fmt.Println("[3/4] Storing extracted knowledge...")

	// Sample source document
	source := &factstore.Source{
		ID:        "doc-001",
		Name:      "Company Overview",
		Content:   "Acme Corporation was founded in 2010 by Alice Smith and Bob Johnson. The company specializes in cloud computing and AI solutions. Alice serves as CEO while Bob leads the engineering team. Their flagship product, CloudAI, has over 10,000 customers worldwide.",
		GroupID:   "example-group",
		Metadata:  map[string]interface{}{"type": "document", "category": "company"},
		CreatedAt: time.Now(),
	}

	if err := db.SaveSource(ctx, source); err != nil {
		log.Fatalf("Failed to save source: %v", err)
	}

	// Extracted nodes (entities) - in real usage, these would come from LLM extraction
	nodes := []*types.ExtractedNode{
		{
			ID:          "node-001",
			SourceID:    source.ID,
			GroupID:     source.GroupID,
			Name:        "Acme Corporation",
			Type:        "organization",
			Description: "A technology company specializing in cloud computing and AI solutions, founded in 2010",
			Embedding:   generateMockEmbedding("Acme Corporation technology company cloud AI"),
			ChunkIndex:  0,
		},
		{
			ID:          "node-002",
			SourceID:    source.ID,
			GroupID:     source.GroupID,
			Name:        "Alice Smith",
			Type:        "person",
			Description: "Co-founder and CEO of Acme Corporation",
			Embedding:   generateMockEmbedding("Alice Smith CEO founder executive"),
			ChunkIndex:  0,
		},
		{
			ID:          "node-003",
			SourceID:    source.ID,
			GroupID:     source.GroupID,
			Name:        "Bob Johnson",
			Type:        "person",
			Description: "Co-founder and head of engineering at Acme Corporation",
			Embedding:   generateMockEmbedding("Bob Johnson engineer founder CTO"),
			ChunkIndex:  0,
		},
		{
			ID:          "node-004",
			SourceID:    source.ID,
			GroupID:     source.GroupID,
			Name:        "CloudAI",
			Type:        "product",
			Description: "Flagship AI product with over 10,000 customers worldwide",
			Embedding:   generateMockEmbedding("CloudAI product AI software customers"),
			ChunkIndex:  0,
		},
	}

	// Extracted triples (relationships)
	triples := []*types.ExtractedTriple{
		{
			ID:          "triple-001",
			SourceID:    source.ID,
			GroupID:     source.GroupID,
			Subject:     "Alice Smith",
			SubjectType: "person",
			Object:      "Acme Corporation",
			ObjectType:  "organization",
			Predicate:   "FOUNDED",
			Description: "Alice Smith co-founded Acme Corporation in 2010",
			Embedding:   generateMockEmbedding("founded created started company"),
			Confidence:  1.0,
			ChunkIndex:  0,
		},
		{
			ID:          "triple-002",
			SourceID:    source.ID,
			GroupID:     source.GroupID,
			Subject:     "Bob Johnson",
			SubjectType: "person",
			Object:      "Acme Corporation",
			ObjectType:  "organization",
			Predicate:   "FOUNDED",
			Description: "Bob Johnson co-founded Acme Corporation in 2010",
			Embedding:   generateMockEmbedding("founded created started company"),
			Confidence:  1.0,
			ChunkIndex:  0,
		},
		{
			ID:          "triple-003",
			SourceID:    source.ID,
			GroupID:     source.GroupID,
			Subject:     "Alice Smith",
			SubjectType: "person",
			Object:      "Acme Corporation",
			ObjectType:  "organization",
			Predicate:   "CEO_OF",
			Description: "Alice Smith serves as CEO of Acme Corporation",
			Embedding:   generateMockEmbedding("CEO chief executive officer leads"),
			Confidence:  1.0,
			ChunkIndex:  0,
		},
		{
			ID:          "triple-004",
			SourceID:    source.ID,
			GroupID:     source.GroupID,
			Subject:     "Acme Corporation",
			SubjectType: "organization",
			Object:      "CloudAI",
			ObjectType:  "product",
			Predicate:   "PRODUCES",
			Description: "Acme Corporation produces CloudAI as their flagship product",
			Embedding:   generateMockEmbedding("produces creates makes product"),
			Confidence:  1.0,
			ChunkIndex:  0,
		},
	}

	if err := db.SaveExtractedKnowledge(ctx, source.ID, nodes, triples); err != nil {
		log.Fatalf("Failed to save extracted knowledge: %v", err)
	}
	fmt.Printf("      Stored %d nodes and %d triples\n", len(nodes), len(triples))

	// ========================================
	// 3. Perform Hybrid Search
	// ========================================
	fmt.Println("[4/4] Performing hybrid search...")
	fmt.Println()

	// Search for information about founders
	query := "Who founded the company?"
	queryEmbedding := generateMockEmbedding("founder created started company CEO")

	searchConfig := &factstore.FactSearchConfig{
		GroupID:  source.GroupID,
		Limit:    5,
		MinScore: 0.0, // Include all results for demo
		SearchMethods: []factstore.SearchMethod{
			factstore.VectorSearch,
			factstore.KeywordSearch,
		},
	}

	results, err := db.HybridSearch(ctx, query, queryEmbedding, searchConfig)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	// Display results
	fmt.Println("================================================================================")
	fmt.Printf("Search Query: %q\n", query)
	fmt.Println("================================================================================")
	fmt.Println()

	fmt.Printf("Found %d nodes:\n", len(results.Nodes))
	fmt.Println("-" + "---------------------------------------")
	for i, node := range results.Nodes {
		score := 0.0
		if i < len(results.NodeScores) {
			score = results.NodeScores[i]
		}
		fmt.Printf("  %d. [%.3f] %s (%s)\n", i+1, score, node.Name, node.Type)
		fmt.Printf("     %s\n", truncate(node.Description, 70))
		fmt.Println()
	}

	if len(results.Triples) > 0 {
		fmt.Printf("Found %d triples:\n", len(results.Triples))
		fmt.Println("----------------------------------------")
		for i, triple := range results.Triples {
			score := 0.0
			if i < len(results.TripleScores) {
				score = results.TripleScores[i]
			}
			fmt.Printf("  %d. [%.3f] %s -[%s]-> %s\n", i+1, score, triple.Subject, triple.Predicate, triple.Object)
			fmt.Printf("     %s\n", truncate(triple.Description, 70))
			fmt.Println()
		}
	}

	// ========================================
	// 4. RAG Application Example
	// ========================================
	fmt.Println("================================================================================")
	fmt.Println("RAG Context Generation")
	fmt.Println("================================================================================")
	fmt.Println()
	fmt.Println("Based on the search results, you can construct RAG context:")
	fmt.Println()

	ragContext := buildRAGContext(results)
	fmt.Println(ragContext)
	fmt.Println()

	fmt.Println("================================================================================")
	fmt.Println("Example completed successfully!")
	fmt.Println("================================================================================")
}

// generateMockEmbedding creates a simple mock embedding for demonstration.
// In production, use a real embedding model like qwen3-embedding or OpenAI.
func generateMockEmbedding(text string) []float32 {
	// Create a deterministic embedding based on text hash
	// This is NOT suitable for real similarity search!
	embedding := make([]float32, 1024)
	for i := 0; i < len(embedding); i++ {
		// Simple hash-based value
		hash := 0
		for j, c := range text {
			hash += int(c) * (i + j + 1)
		}
		embedding[i] = float32(hash%1000) / 1000.0
	}
	return embedding
}

// truncate shortens a string to maxLen characters
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}

// buildRAGContext constructs context from search results for LLM prompting
func buildRAGContext(results *factstore.FactSearchResults) string {
	var context string

	context += "Relevant entities:\n"
	for _, node := range results.Nodes {
		context += fmt.Sprintf("- %s (%s): %s\n", node.Name, node.Type, node.Description)
	}

	if len(results.Triples) > 0 {
		context += "\nRelationships:\n"
		for _, triple := range results.Triples {
			context += fmt.Sprintf("- %s %s %s: %s\n", triple.Subject, triple.Predicate, triple.Object, triple.Description)
		}
	}

	return context
}

// demonstrateMockUsage shows the API without a real database connection
func demonstrateMockUsage() {
	fmt.Println()
	fmt.Println("================================================================================")
	fmt.Println("FactStore RAG API Overview (Demo Mode)")
	fmt.Println("================================================================================")
	fmt.Println()
	fmt.Println("To use the factstore, you would:")
	fmt.Println()
	fmt.Println("1. Create a factstore connection:")
	fmt.Println("   db, err := factstore.NewPostgresDB(connString, 1024)")
	fmt.Println()
	fmt.Println("2. Initialize the schema:")
	fmt.Println("   db.Initialize(ctx)")
	fmt.Println()
	fmt.Println("3. Store extracted knowledge:")
	fmt.Println("   db.SaveSource(ctx, source)")
	fmt.Println("   db.SaveExtractedKnowledge(ctx, sourceID, nodes, triples)")
	fmt.Println()
	fmt.Println("4. Search with hybrid (vector + keyword) search:")
	fmt.Println("   results, err := db.HybridSearch(ctx, query, embedding, config)")
	fmt.Println()
	fmt.Println("5. Use results for RAG:")
	fmt.Println("   - results.Nodes: Matching entities with scores")
	fmt.Println("   - results.Triples: Matching relationships with scores")
	fmt.Println("   - Build context from results for LLM prompts")
	fmt.Println()
	fmt.Println("For a complete example with a real database, set:")
	fmt.Println("  FACTSTORE_POSTGRES_URL='postgres://user:pass@localhost:5432/facts'")
	fmt.Println()
	fmt.Println("See docs/FACTSTORE_RAG.md for setup instructions.")
	fmt.Println()
	fmt.Println("================================================================================")
	fmt.Println("Demo completed!")
	fmt.Println("================================================================================")
}
