// Package main demonstrates how to build an interactive chat application using predicato
// with internal services only - no external APIs required.
//
// This example shows how to:
// - Create separate Predicato clients for global knowledge and user-specific data
// - Use Candle SmolLM2 for text generation (local, no API)
// - Use go-candle with qwen3-embedding-0.6b for embeddings (local, no API)
// - Use go-candle embedding-based reranking (local, no API)
// - Use episodes to track conversation history with AddToEpisode
// - Apply reranking to improve search result quality
// - Maintain conversation continuity with UUID v7 episode IDs
// - Build an interactive chat loop with conversation history
//
// Prerequisites:
// - CGO enabled (required for Rust FFI bindings)
// - ~4GB RAM minimum
// - No API keys or external services required!
//
// First run will download models from HuggingFace Hub.
//
// Usage:
//
//	go run main.go --user-id alice
//	go run main.go --user-id alice --global-db ./knowledge_db --user-db-dir ./user_dbs
package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/soundprediction/predicato"
	candleAdapter "github.com/soundprediction/predicato/pkg/candle"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/types"
)

// ChatSession holds the conversation state
type ChatSession struct {
	SessionID string
	EpisodeID string
	Messages  []Message
}

// Message represents a single conversation turn
type Message struct {
	UserQuery   string
	LLMResponse string
	Timestamp   time.Time
}

// ChatClients holds all the clients needed for the chat application
type ChatClients struct {
	GlobalPredicato *predicato.Client                   // Global knowledge base (can be nil)
	UserPredicato   *predicato.Client                   // User-specific episodic memory
	LLM             nlp.Client                          // LLM for text generation
	Candle          *candleAdapter.Client               // Candle client for direct generation
	Reranker        *candleAdapter.CandleRerankerClient // Reranker for improving search quality
	Context         context.Context
}

func main() {
	// Command-line flags
	userID := flag.String("user-id", "alice", "User ID for the chat session")
	globalDBPath := flag.String("global-db", "./knowledge_db/content_graph.ladybugdb", "Path to global knowledge base (ladybug database)")
	userDBDir := flag.String("user-db-dir", "./user_dbs", "Directory for user-specific databases")
	skipGlobal := flag.Bool("skip-global", false, "Skip loading global knowledge base")
	flag.Parse()

	fmt.Println("================================================================================")
	fmt.Println("Predicato Interactive Chat - Internal Services Stack")
	fmt.Println("================================================================================")
	fmt.Println()
	fmt.Println("This chat uses predicato's internal services:")
	fmt.Println("  - Ladybug: embedded graph database (no server required)")
	fmt.Println("  - Candle SmolLM2: local text generation (no API required)")
	fmt.Println("  - Candle: local embeddings with qwen/qwen3-embedding-0.6b")
	fmt.Println("  - Candle: local embedding-based reranking")
	fmt.Println()
	fmt.Println("No API keys or external services needed!")
	fmt.Printf("User ID: %s\n", *userID)
	fmt.Println()

	// Initialize clients
	clients, err := initializeClients(*userID, *globalDBPath, *userDBDir, *skipGlobal)
	if err != nil {
		log.Fatalf("Failed to initialize clients: %v", err)
	}
	defer clients.Close()

	// Run the chat loop
	runChatLoop(clients, *userID)
}

func initializeClients(userID, globalDBPath, userDBDir string, skipGlobal bool) (*ChatClients, error) {
	ctx := context.Background()

	fmt.Println("Initializing internal services...")
	fmt.Println("(First run will download models, please wait...)")
	fmt.Println()

	// ========================================
	// 1. Create Candle Client (Local Text Generation)
	// ========================================
	fmt.Println("[1/4] Setting up Candle SmolLM2 for text generation...")

	candleClient := candleAdapter.NewClient(candleAdapter.CandleNLPConfig{
		TextGenModelID: "HuggingFaceTB/SmolLM2-360M-Instruct",
	})

	// Create LLM adapter for nlp.Client interface
	llmClient := candleAdapter.NewLLMAdapter(candleClient, "text_generation")
	fmt.Println("      Candle SmolLM2 loaded")

	// ========================================
	// 2. Create Embedder Client (Local Embeddings)
	// ========================================
	fmt.Println("[2/4] Setting up Candle embedder with qwen/qwen3-embedding-0.6b...")

	embedderClient, err := candleAdapter.NewCandleEmbedderClient(&candleAdapter.CandleEmbedderConfig{
		Model:      "qwen/qwen3-embedding-0.6b",
		Dimensions: 1024,
		Normalize:  true,
	})
	if err != nil {
		candleClient.Close()
		return nil, fmt.Errorf("failed to create embedder client: %w", err)
	}
	fmt.Println("      Candle embedder loaded")

	// ========================================
	// 3. Create Reranker Client (Local Reranking)
	// ========================================
	fmt.Println("[3/4] Setting up Candle embedding-based reranker...")

	rerankerClient := candleAdapter.NewCandleRerankerClient(embedderClient)
	fmt.Println("      Candle reranker loaded")

	// ========================================
	// 4. Create Predicato Clients
	// ========================================
	fmt.Println("[4/4] Setting up Predicato clients...")

	// Create global Predicato client (if enabled)
	var globalPredicatoClient *predicato.Client
	if !skipGlobal {
		// Check if global database exists
		if _, err := os.Stat(globalDBPath); err == nil {
			ladybugDriver, err := driver.NewLadybugDriver(globalDBPath, 1)
			if err != nil {
				fmt.Printf("      Warning: Failed to load global database: %v\n", err)
				fmt.Println("      Continuing without global knowledge base...")
			} else {
				globalConfig := &predicato.Config{
					GroupID:  "global-knowledge",
					TimeZone: time.UTC,
				}
				globalPredicatoClient, err = predicato.NewClient(ladybugDriver, llmClient, embedderClient, globalConfig, nil)
				if err != nil {
					fmt.Printf("      Warning: Failed to create global client: %v\n", err)
					globalPredicatoClient = nil
				} else {
					fmt.Printf("      Global knowledge base loaded from %s\n", globalDBPath)
				}
			}
		} else {
			fmt.Printf("      Global knowledge base not found at %s\n", globalDBPath)
			fmt.Println("      Continuing without global knowledge base...")
		}
	}

	// Create user-specific Predicato client
	userDBPath := filepath.Join(userDBDir, fmt.Sprintf("user_%s.ladybugdb", userID))

	// Create parent directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(userDBPath), 0755); err != nil {
		candleClient.Close()
		embedderClient.Close()
		return nil, fmt.Errorf("failed to create user database directory: %w", err)
	}

	userLadybugDriver, err := driver.NewLadybugDriver(userDBPath, 1)
	if err != nil {
		candleClient.Close()
		embedderClient.Close()
		return nil, fmt.Errorf("failed to create user ladybug driver: %w", err)
	}

	userConfig := &predicato.Config{
		GroupID:  fmt.Sprintf("user-%s-chat", userID),
		TimeZone: time.UTC,
	}
	userPredicatoClient, err := predicato.NewClient(userLadybugDriver, llmClient, embedderClient, userConfig, nil)
	if err != nil {
		candleClient.Close()
		embedderClient.Close()
		return nil, fmt.Errorf("failed to create user client: %w", err)
	}
	fmt.Printf("      User database initialized at %s\n", userDBPath)

	fmt.Println()
	fmt.Println("All components initialized successfully!")
	fmt.Println()

	return &ChatClients{
		GlobalPredicato: globalPredicatoClient,
		UserPredicato:   userPredicatoClient,
		LLM:             llmClient,
		Candle:          candleClient,
		Reranker:        rerankerClient,
		Context:         ctx,
	}, nil
}

func (c *ChatClients) Close() {
	if c.GlobalPredicato != nil {
		c.GlobalPredicato.Close(c.Context)
	}
	if c.UserPredicato != nil {
		c.UserPredicato.Close(c.Context)
	}
	if c.Candle != nil {
		c.Candle.Close()
	}
	// Reranker shares the embedder, which is cleaned up via Candle close
}

func runChatLoop(clients *ChatClients, userID string) {
	// Initialize chat session
	sessionID, err := uuid.NewV7()
	if err != nil {
		sessionID = uuid.New()
	}

	session := &ChatSession{
		SessionID: sessionID.String(),
		Messages:  []Message{},
	}

	// Print welcome banner
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println("Predicato Interactive Chat")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println()
	fmt.Println("Commands:")
	fmt.Println("  Type your question and press Enter")
	fmt.Println("  Type 'exit' or 'quit' to end the session")
	fmt.Println("  Type 'history' to view conversation history")
	fmt.Println("  Type 'search <query>' to search the global knowledge base")
	fmt.Println(strings.Repeat("=", 70))
	fmt.Println()

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())

		// Handle commands
		switch {
		case input == "exit" || input == "quit":
			fmt.Println("\nGoodbye!")
			return
		case input == "history":
			showHistory(session)
			continue
		case strings.HasPrefix(input, "search "):
			query := strings.TrimPrefix(input, "search ")
			searchGlobalKnowledge(clients, query)
			continue
		case input == "":
			continue
		}

		// Process the query
		processQuery(clients, session, userID, input)
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("\nError reading input: %v\n", err)
	}
}

func processQuery(clients *ChatClients, session *ChatSession, userID, input string) {
	ctx := clients.Context

	// Format conversation turn
	conversationTurn := fmt.Sprintf("User: %s\n", input)

	// Add to episode (create or append)
	// Run in background to avoid freezing the chat
	if session.EpisodeID == "" {
		// First message - create initial episode with UUID v7
		episodeID, err := uuid.NewV7()
		if err != nil {
			episodeID = uuid.New()
		}

		episode := types.Episode{
			ID:        episodeID.String(),
			Name:      fmt.Sprintf("Chat with %s", userID),
			Content:   conversationTurn,
			GroupID:   fmt.Sprintf("user-%s-chat", userID),
			Metadata:  map[string]interface{}{"session_id": session.SessionID, "type": "chat"},
			Reference: time.Now(),
		}

		// Set the episode ID immediately so subsequent messages can reference it
		session.EpisodeID = episode.ID

		// Create episode in background
		go func(ep types.Episode) {
			result, err := clients.UserPredicato.Add(ctx, []types.Episode{ep}, nil)
			if err != nil {
				fmt.Printf("Warning: Failed to create episode: %v\n", err)
			} else if result != nil && len(result.Episodes) > 0 {
				fmt.Printf("Created episode %s\n", result.Episodes[0].Uuid)
			}
		}(episode)
	} else {
		// Subsequent messages - append to existing episode in background
		go func(episodeID, content string) {
			_, err := clients.UserPredicato.AddToEpisode(ctx, episodeID, content, nil)
			if err != nil {
				fmt.Printf("Warning: Failed to append to episode: %v\n", err)
			}
		}(session.EpisodeID, conversationTurn)
	}

	// Search global knowledge base if available and rerank results
	var contextNodes []*types.Node
	if clients.GlobalPredicato != nil {
		fmt.Println("Searching global knowledge base...")
		searchConfig := &types.SearchConfig{
			Limit:              10, // Get more results for reranking
			CenterNodeDistance: 2,
			MinScore:           0.0,
			IncludeEdges:       true,
		}

		results, err := clients.GlobalPredicato.Search(ctx, input, searchConfig)
		if err != nil {
			fmt.Printf("Warning: Search failed: %v\n", err)
		} else if results != nil && len(results.Nodes) > 0 {
			// Rerank the results
			passages := make([]string, len(results.Nodes))
			for i, node := range results.Nodes {
				passages[i] = node.Summary
			}

			rankedResults, err := clients.Reranker.Rank(ctx, input, passages)
			if err != nil {
				fmt.Printf("Warning: Reranking failed, using original order: %v\n", err)
				contextNodes = results.Nodes
			} else {
				// Reorder nodes based on reranking
				fmt.Printf("Found %d relevant nodes (reranked)\n", len(rankedResults))

				// Create a map for quick lookup
				nodeMap := make(map[string]*types.Node)
				for _, node := range results.Nodes {
					nodeMap[node.Summary] = node
				}

				// Take top 3 reranked results
				for i, ranked := range rankedResults {
					if i >= 3 {
						break
					}
					if node, ok := nodeMap[ranked.Passage]; ok {
						contextNodes = append(contextNodes, node)
						fmt.Printf("  %d. (score: %.3f) %s\n", i+1, ranked.Score, truncate(node.Name, 50))
					}
				}
			}
		} else {
			fmt.Println("No relevant context found")
		}
	}

	// Build prompt with context
	prompt := buildPrompt(input, session.Messages, contextNodes)

	// Generate response using Candle SmolLM2
	fmt.Println()
	fmt.Println("Assistant:")
	fmt.Println(strings.Repeat("-", 70))

	response, err := clients.Candle.GenerateText(ctx, prompt)
	if err != nil {
		fmt.Printf("Failed to generate response: %v\n", err)
		return
	}

	// Clean up the response (model might include the prompt)
	response = strings.TrimPrefix(response, prompt)
	response = strings.TrimSpace(response)

	if response == "" {
		response = "I'm sorry, I couldn't generate a response. Please try rephrasing your question."
	}

	fmt.Println(response)
	fmt.Println(strings.Repeat("-", 70))
	fmt.Println()

	// Save message to session
	message := Message{
		UserQuery:   input,
		LLMResponse: response,
		Timestamp:   time.Now(),
	}
	session.Messages = append(session.Messages, message)

	// Append assistant response to episode in background
	if session.EpisodeID != "" {
		go func(episodeID, resp string) {
			assistantTurn := fmt.Sprintf("Assistant: %s\n", resp)
			_, err := clients.UserPredicato.AddToEpisode(ctx, episodeID, assistantTurn, nil)
			if err != nil {
				fmt.Printf("Warning: Failed to append assistant response: %v\n", err)
			}
		}(session.EpisodeID, response)
	}
}

func buildPrompt(query string, history []Message, contextNodes []*types.Node) string {
	var prompt strings.Builder

	prompt.WriteString("You are a helpful AI assistant. ")

	// Add context from knowledge base
	if len(contextNodes) > 0 {
		prompt.WriteString("Use the following context to help answer the question:\n\n")
		for i, node := range contextNodes {
			if i >= 3 {
				break
			}
			prompt.WriteString(fmt.Sprintf("Context %d: %s\n%s\n\n", i+1, node.Name, node.Summary))
		}
	}

	// Add conversation history (last 3 messages for model's limited context)
	if len(history) > 0 {
		prompt.WriteString("Previous conversation:\n")
		start := 0
		if len(history) > 3 {
			start = len(history) - 3
		}
		for _, msg := range history[start:] {
			prompt.WriteString(fmt.Sprintf("User: %s\nAssistant: %s\n", msg.UserQuery, truncate(msg.LLMResponse, 100)))
		}
		prompt.WriteString("\n")
	}

	prompt.WriteString(fmt.Sprintf("User: %s\nAssistant:", query))
	return prompt.String()
}

func showHistory(session *ChatSession) {
	if len(session.Messages) == 0 {
		fmt.Println("No conversation history yet")
		return
	}

	fmt.Println()
	fmt.Println("Conversation History:")
	fmt.Println(strings.Repeat("-", 70))
	for i, msg := range session.Messages {
		fmt.Printf("%d. You: %s\n", i+1, msg.UserQuery)
		fmt.Printf("   Assistant: %s\n", truncate(msg.LLMResponse, 100))
		fmt.Println(strings.Repeat("-", 70))
	}
	fmt.Println()
}

func searchGlobalKnowledge(clients *ChatClients, query string) {
	if clients.GlobalPredicato == nil {
		fmt.Println("Global knowledge base not available")
		return
	}

	fmt.Printf("Searching for: %s\n", query)

	searchConfig := &types.SearchConfig{
		Limit:              10,
		CenterNodeDistance: 2,
		MinScore:           0.0,
		IncludeEdges:       true,
	}

	results, err := clients.GlobalPredicato.Search(clients.Context, query, searchConfig)
	if err != nil {
		fmt.Printf("Search failed: %v\n", err)
		return
	}

	if results == nil || len(results.Nodes) == 0 {
		fmt.Println("No results found")
		return
	}

	// Rerank results
	passages := make([]string, len(results.Nodes))
	for i, node := range results.Nodes {
		passages[i] = node.Summary
	}

	rankedResults, err := clients.Reranker.Rank(clients.Context, query, passages)
	if err != nil {
		fmt.Printf("Reranking failed, showing original order: %v\n", err)
		fmt.Printf("Found %d nodes:\n", len(results.Nodes))
		for i, node := range results.Nodes {
			fmt.Printf("%d. %s (%s)\n", i+1, node.Name, node.Type)
			if node.Summary != "" {
				fmt.Printf("   %s\n", truncate(node.Summary, 100))
			}
		}
		return
	}

	fmt.Printf("Found %d nodes (reranked):\n", len(rankedResults))
	nodeMap := make(map[string]*types.Node)
	for _, node := range results.Nodes {
		nodeMap[node.Summary] = node
	}

	for i, ranked := range rankedResults {
		if node, ok := nodeMap[ranked.Passage]; ok {
			fmt.Printf("%d. (score: %.3f) %s (%s)\n", i+1, ranked.Score, node.Name, node.Type)
			fmt.Printf("   %s\n", truncate(ranked.Passage, 100))
		}
	}
	fmt.Println()
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
