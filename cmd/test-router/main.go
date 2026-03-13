//go:build cgo

package main

import (
	"context"
	"encoding/csv"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/soundprediction/predicato/pkg/embedder"
	"github.com/soundprediction/predicato/pkg/router"
	"github.com/soundprediction/predicato/pkg/types"
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))

	// Create embedder (qwen3-embedding-0.6b via go-embedeverything)
	fmt.Println("Loading qwen3-embedding-0.6b embedder...")
	t0 := time.Now()
	emb, err := embedder.NewEmbedEverythingClient(&embedder.EmbedEverythingConfig{
		Config: &embedder.Config{
			Model:      "qwen/qwen3-embedding-0.6b",
			Dimensions: 1024,
		},
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create embedder: %v\n", err)
		os.Exit(1)
	}
	defer emb.Close()
	fmt.Printf("Embedder loaded in %v\n", time.Since(t0))

	// Simulated chat queries — diverse medical topics to test routing
	queries := []struct {
		query       string
		expectTopic string
	}{
		{"What medications treat type 2 diabetes?", "diabetes-type2"},
		{"How does metformin affect insulin resistance?", "diabetes-type2"},
		{"What are the side effects of chemotherapy for breast cancer?", "breast-cancer"},
		{"Tell me about BRCA1 gene mutations", "breast-cancer"},
		{"What causes rheumatoid arthritis?", "autoimmune"},
		{"How is lupus diagnosed?", "autoimmune"},
		{"What vaccines are recommended for respiratory infections?", "respiratory-infections"},
		{"Explain how statins lower cholesterol", "cardiovascular"},
		{"What is the treatment for major depressive disorder?", "mental-health"},
		{"How does ADHD medication work?", "mental-health"},
		{"What are symptoms of irritable bowel syndrome?", "digestive-system"},
		{"How is kidney failure treated?", "kidney-disease"},
	}

	backends := []struct {
		name string
		dir  string
	}{
		{"duckpgq", "/data/topic-graphs/duckpgq"},
		{"ladybug", "/data/topic-graphs/ladybug"},
	}

	// Open CSV file
	csvFile, err := os.Create(os.ExpandEnv("$HOME/router-test-results.csv"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create CSV: %v\n", err)
		os.Exit(1)
	}
	defer csvFile.Close()
	w := csv.NewWriter(csvFile)
	defer w.Flush()

	// CSV header
	w.Write([]string{
		"backend", "query_num", "query", "expected_topic", "routed_to",
		"route_ms", "search_ms", "num_nodes", "num_edges",
		"node_name", "node_entity_type", "node_summary",
		"edge_name", "edge_fact", "edge_source", "edge_target",
		"errors",
	})

	for _, backend := range backends {
		fmt.Printf("\n%s\n", strings.Repeat("=", 80))
		fmt.Printf("BACKEND: %s (%s)\n", backend.name, backend.dir)
		fmt.Printf("%s\n\n", strings.Repeat("=", 80))

		// Discovery
		discoverStart := time.Now()
		configs, routingTexts, err := router.DiscoverTopicGraphs(backend.dir)
		discoverElapsed := time.Since(discoverStart)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Discovery failed for %s: %v\n", backend.name, err)
			continue
		}
		fmt.Printf("Discovered %d topic graphs in %v\n", len(configs), discoverElapsed)

		// Create embedding-based classifier
		routerConfig := router.RouterConfig{
			Strategy:      "embedding",
			MinConfidence: 0.1,
			MaxGraphs:     3,
			MergeStrategy: "rrf",
		}

		classifierStart := time.Now()
		classifier, err := router.NewTopicClassifier(configs, routerConfig, emb, nil, routingTexts)
		classifierElapsed := time.Since(classifierStart)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Classifier creation failed: %v\n", err)
			continue
		}
		fmt.Printf("Classifier initialized in %v (embedded %d topic vectors)\n", classifierElapsed, len(configs))

		// Use LRU manager with max 5 open clients (ladybug can only handle ~1 at a time safely)
		maxOpen := 38 // duckpgq can handle many
		if backend.name == "ladybug" {
			maxOpen = 5
		}
		mgr := router.NewManagerWithOptions(configs, nil, emb, logger, maxOpen)

		rtr := router.NewRouter(mgr, classifier, routerConfig, logger)

		searchConfig := &types.SearchConfig{
			Limit:              10,
			CenterNodeDistance: 2,
			IncludeEdges:       true,
			Rerank:             false,
			ExcludeEntityTypes: []string{"CLINICAL_STUDY"},
			NodeConfig: &types.NodeSearchConfig{
				SearchMethods: []string{"cosine_similarity", "bm25"},
				Reranker:      "rrf",
			},
			EdgeConfig: &types.EdgeSearchConfig{
				SearchMethods: []string{"bm25"},
				Reranker:      "rrf",
			},
		}

		ctx := context.Background()

		for i, q := range queries {
			qNum := i + 1
			fmt.Printf("--- Query %d: %q\n", qNum, q.query)
			fmt.Printf("    Expected: %s\n", q.expectTopic)

			// Route
			routeStart := time.Now()
			info, err := rtr.GetRouteInfo(ctx, q.query)
			routeTime := time.Since(routeStart)

			if err != nil {
				fmt.Printf("    ROUTE ERROR: %v\n\n", err)
				w.Write([]string{
					backend.name, fmt.Sprint(qNum), q.query, q.expectTopic, "",
					fmt.Sprint(routeTime.Milliseconds()), "", "", "",
					"", "", "",
					"", "", "", "",
					err.Error(),
				})
				continue
			}

			routedTo := strings.Join(info.ClientNames, ",")
			topN := 3
			if len(info.TopicMatches) < topN {
				topN = len(info.TopicMatches)
			}
			fmt.Printf("    Routed to: %v (route: %v)\n", info.ClientNames, routeTime)
			for j := 0; j < topN; j++ {
				m := info.TopicMatches[j]
				fmt.Printf("      #%d %s (confidence=%.4f)\n", j+1, m.ClientName, m.Confidence)
			}

			// Search
			searchStart := time.Now()
			merged, err := rtr.SearchWithClients(ctx, q.query, info.ClientNames, searchConfig)
			searchTime := time.Since(searchStart)

			if err != nil {
				fmt.Printf("    SEARCH ERROR: %v\n\n", err)
				w.Write([]string{
					backend.name, fmt.Sprint(qNum), q.query, q.expectTopic, routedTo,
					fmt.Sprint(routeTime.Milliseconds()), fmt.Sprint(searchTime.Milliseconds()), "", "",
					"", "", "",
					"", "", "", "",
					err.Error(),
				})
				continue
			}

			fmt.Printf("    Results: %d nodes, %d edges (search: %v)\n",
				len(merged.Nodes), len(merged.Edges), searchTime)

			// Collect errors
			var errStrs []string
			for clientName, searchErr := range merged.Errors {
				errStrs = append(errStrs, fmt.Sprintf("%s: %v", clientName, searchErr))
				fmt.Printf("    ERROR[%s]: %v\n", clientName, searchErr)
			}
			errStr := strings.Join(errStrs, "; ")

			// Print top nodes
			limit := 5
			if len(merged.Nodes) < limit {
				limit = len(merged.Nodes)
			}
			for j := 0; j < limit; j++ {
				n := merged.Nodes[j]
				name := n.Name
				if len(name) > 60 {
					name = name[:60] + "..."
				}
				entityType := n.EntityType
				if entityType == "" {
					entityType = string(n.Type)
				}
				src := merged.Sources[n.Uuid]
				fmt.Printf("      %d. [%s] %s (from: %s)\n", j+1, entityType, name, strings.Join(src, ","))
			}

			// Print top edges
			edgeLimit := 5
			if len(merged.Edges) < edgeLimit {
				edgeLimit = len(merged.Edges)
			}
			for j := 0; j < edgeLimit; j++ {
				e := merged.Edges[j]
				fact := e.Fact
				if len(fact) > 80 {
					fact = fact[:80] + "..."
				}
				fmt.Printf("      E%d. [%s] %s\n", j+1, e.Name, fact)
			}

			// Write node rows to CSV
			for _, n := range merged.Nodes {
				entityType := n.EntityType
				if entityType == "" {
					entityType = string(n.Type)
				}
				summary := n.Summary
				if summary == "" {
					summary = n.Content
				}
				w.Write([]string{
					backend.name, fmt.Sprint(qNum), q.query, q.expectTopic, routedTo,
					fmt.Sprint(routeTime.Milliseconds()), fmt.Sprint(searchTime.Milliseconds()),
					fmt.Sprint(len(merged.Nodes)), fmt.Sprint(len(merged.Edges)),
					n.Name, entityType, summary,
					"", "", "", "",
					errStr,
				})
			}
			// Write edge rows to CSV
			for _, e := range merged.Edges {
				w.Write([]string{
					backend.name, fmt.Sprint(qNum), q.query, q.expectTopic, routedTo,
					fmt.Sprint(routeTime.Milliseconds()), fmt.Sprint(searchTime.Milliseconds()),
					fmt.Sprint(len(merged.Nodes)), fmt.Sprint(len(merged.Edges)),
					"", "", "",
					e.Name, e.Fact, e.SourceNodeID, e.TargetNodeID,
					errStr,
				})
			}

			// If no results at all, write one row for the query
			if len(merged.Nodes) == 0 && len(merged.Edges) == 0 {
				w.Write([]string{
					backend.name, fmt.Sprint(qNum), q.query, q.expectTopic, routedTo,
					fmt.Sprint(routeTime.Milliseconds()), fmt.Sprint(searchTime.Milliseconds()),
					"0", "0",
					"", "", "",
					"", "", "", "",
					errStr,
				})
			}

			fmt.Println()
		}

		if err := rtr.Close(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "Close error: %v\n", err)
		}
	}

	fmt.Printf("\nCSV written to ~/router-test-results.csv\n")
}
