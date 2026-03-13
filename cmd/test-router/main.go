//go:build cgo

package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/soundprediction/predicato/pkg/embedder"
	"github.com/soundprediction/predicato/pkg/router"
	"github.com/soundprediction/predicato/pkg/types"
)

type queryResult struct {
	query        string
	expectTopic  string
	routedTo     []string
	topMatches   []router.TopicMatch
	numNodes     int
	numEdges     int
	routeTime    time.Duration
	searchTime   time.Duration
	topNodes     []nodeInfo
	errors       map[string]error
	usedFallback bool
	usedDefault  bool
}

type nodeInfo struct {
	name       string
	entityType string
	sources    []string
}

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

	// Collect all results for comparison
	allResults := make(map[string][]queryResult)

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

		mgr := router.NewManager(configs, nil, emb, logger)

		// Eagerly initialize all clients to avoid concurrent DB opens
		fmt.Print("Initializing all clients...")
		initStart := time.Now()
		ok, fail := mgr.InitializeAllClients()
		fmt.Printf(" done in %v (%d ok, %d fail)\n", time.Since(initStart), ok, fail)

		rtr := router.NewRouter(mgr, classifier, routerConfig, logger)

		searchConfig := &types.SearchConfig{
			Limit:              10,
			CenterNodeDistance: 2,
			IncludeEdges:       true,
			Rerank:             false,
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
		var results []queryResult

		for i, q := range queries {
			qr := queryResult{
				query:       q.query,
				expectTopic: q.expectTopic,
			}

			fmt.Printf("--- Query %d: %q\n", i+1, q.query)
			fmt.Printf("    Expected: %s\n", q.expectTopic)

			// Route
			routeStart := time.Now()
			info, err := rtr.GetRouteInfo(ctx, q.query)
			qr.routeTime = time.Since(routeStart)

			if err != nil {
				fmt.Printf("    ROUTE ERROR: %v\n\n", err)
				results = append(results, qr)
				continue
			}

			qr.routedTo = info.ClientNames
			qr.topMatches = info.TopicMatches
			qr.usedFallback = info.UsedFallback
			qr.usedDefault = info.UsedDefault

			topN := 3
			if len(info.TopicMatches) < topN {
				topN = len(info.TopicMatches)
			}
			fmt.Printf("    Routed to: %v (route: %v)\n", info.ClientNames, qr.routeTime)
			for j := 0; j < topN; j++ {
				m := info.TopicMatches[j]
				fmt.Printf("      #%d %s (confidence=%.4f)\n", j+1, m.ClientName, m.Confidence)
			}

			// Search
			searchStart := time.Now()
			merged, err := rtr.SearchWithClients(ctx, q.query, info.ClientNames, searchConfig)
			qr.searchTime = time.Since(searchStart)

			if err != nil {
				fmt.Printf("    SEARCH ERROR: %v\n\n", err)
				results = append(results, qr)
				continue
			}

			qr.numNodes = len(merged.Nodes)
			qr.numEdges = len(merged.Edges)
			qr.errors = merged.Errors

			fmt.Printf("    Results: %d nodes, %d edges (search: %v)\n",
				qr.numNodes, qr.numEdges, qr.searchTime)

			// Top nodes
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
				qr.topNodes = append(qr.topNodes, nodeInfo{
					name:       n.Name,
					entityType: entityType,
					sources:    src,
				})
			}

			for clientName, searchErr := range merged.Errors {
				fmt.Printf("    ERROR[%s]: %v\n", clientName, searchErr)
			}

			fmt.Println()
			results = append(results, qr)
		}

		allResults[backend.name] = results

		if err := rtr.Close(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "Close error: %v\n", err)
		}
	}

	// Print comparison summary
	fmt.Printf("\n%s\n", strings.Repeat("=", 80))
	fmt.Println("COMPARISON SUMMARY")
	fmt.Printf("%s\n\n", strings.Repeat("=", 80))

	duckResults := allResults["duckpgq"]
	lbugResults := allResults["ladybug"]

	if len(duckResults) == 0 || len(lbugResults) == 0 {
		fmt.Println("(skipped — not all backends completed)")
		return
	}

	fmt.Printf("%-4s  %-55s  %-12s  %-12s  %-8s  %-8s  %-10s  %-10s\n",
		"#", "Query", "Duck Nodes", "Lbug Nodes", "Duck ms", "Lbug ms", "Duck Route", "Lbug Route")
	fmt.Println(strings.Repeat("-", 140))

	var totalDuckSearch, totalLbugSearch time.Duration
	var totalDuckNodes, totalLbugNodes int

	for i := range queries {
		dr := duckResults[i]
		lr := lbugResults[i]

		dRoute := strings.Join(dr.routedTo, ",")
		lRoute := strings.Join(lr.routedTo, ",")
		if len(dRoute) > 10 {
			dRoute = dRoute[:10]
		}
		if len(lRoute) > 10 {
			lRoute = lRoute[:10]
		}

		q := dr.query
		if len(q) > 55 {
			q = q[:52] + "..."
		}

		fmt.Printf("%-4d  %-55s  %-12d  %-12d  %-8d  %-8d  %-10s  %-10s\n",
			i+1, q, dr.numNodes, lr.numNodes,
			dr.searchTime.Milliseconds(), lr.searchTime.Milliseconds(),
			dRoute, lRoute)

		totalDuckSearch += dr.searchTime
		totalLbugSearch += lr.searchTime
		totalDuckNodes += dr.numNodes
		totalLbugNodes += lr.numNodes
	}

	n := len(queries)
	fmt.Println(strings.Repeat("-", 140))
	fmt.Printf("%-4s  %-55s  %-12d  %-12d  %-8d  %-8d\n",
		"AVG", "",
		totalDuckNodes/n, totalLbugNodes/n,
		totalDuckSearch.Milliseconds()/int64(n),
		totalLbugSearch.Milliseconds()/int64(n))
	fmt.Printf("%-4s  %-55s  %-12d  %-12d  %-8d  %-8d\n",
		"TOT", "",
		totalDuckNodes, totalLbugNodes,
		totalDuckSearch.Milliseconds(),
		totalLbugSearch.Milliseconds())
}
