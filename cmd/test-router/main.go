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

	"github.com/soundprediction/predicato/pkg/crossencoder"
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

	// Create embedding-based reranker (reuses the embedder — no extra model)
	reranker := crossencoder.NewEmbeddingRerankerClient(emb, crossencoder.EmbeddingConfig{})

	// 100 diverse medical queries across all topic domains
	queries := []struct {
		query       string
		expectTopic string
	}{
		// === CARDIOVASCULAR (1-8) ===
		{"What are the warning signs of a heart attack?", "cardiovascular"},
		{"How do blood thinners prevent stroke?", "cardiovascular"},
		{"What is atrial fibrillation and how is it managed?", "cardiovascular"},
		{"Why do statins lower cholesterol?", "cardiovascular"},
		{"What are the stages of heart failure?", "cardiovascular"},
		{"How does coronary artery bypass surgery work?", "cardiovascular"},
		{"What lifestyle changes help prevent heart disease?", "cardiovascular"},
		{"What is the connection between high blood pressure and heart attack?", "hypertension"},

		// === HYPERTENSION (9-12) ===
		{"What medications are used to treat hypertension?", "hypertension"},
		{"How does sodium intake affect blood pressure?", "hypertension"},
		{"What is resistant hypertension?", "hypertension"},
		{"Can hypertension cause kidney damage?", "hypertension"},

		// === ONCOLOGY - GENERAL (13-16) ===
		{"What is immunotherapy and how does it fight cancer?", "cancer"},
		{"How do checkpoint inhibitors work against tumors?", "cancer"},
		{"What is the difference between chemotherapy and targeted therapy?", "cancer"},
		{"How does cancer metastasize to other organs?", "cancer"},

		// === ONCOLOGY - SPECIFIC CANCERS (17-22) ===
		{"What screening tests detect colorectal cancer early?", "colorectal-cancer"},
		{"What are the risk factors for breast cancer?", "breast-cancer"},
		{"How is non-small cell lung cancer treated?", "lung-cancer"},
		{"What is the role of PSA testing in prostate cancer?", "prostate-health"},
		{"What genes increase the risk of colorectal cancer?", "colorectal-cancer"},
		{"How effective is mammography for early detection?", "breast-cancer"},

		// === INFECTIOUS DISEASE (23-28) ===
		{"How do antibiotics become resistant to bacteria?", "infectious-disease"},
		{"What is the treatment for tuberculosis?", "infectious-disease"},
		{"How does HIV attack the immune system?", "infectious-disease"},
		{"What are the symptoms of sepsis?", "infectious-disease"},
		{"How do antiviral medications work against influenza?", "respiratory-infections"},
		{"What causes urinary tract infections?", "urinary-tract"},

		// === NEUROLOGY (29-34) ===
		{"What are the early signs of Alzheimer's disease?", "dementia"},
		{"How does Parkinson's disease affect movement?", "neurological"},
		{"What causes epileptic seizures?", "neurological"},
		{"How is multiple sclerosis diagnosed?", "neurological"},
		{"What are the types of dementia besides Alzheimer's?", "dementia"},
		{"What is the blood-brain barrier and why does it matter?", "neurological"},

		// === MENTAL HEALTH (35-42) ===
		{"How do SSRIs work for anxiety disorders?", "depression-anxiety"},
		{"What is the difference between bipolar I and bipolar II?", "serious-mental-illness"},
		{"How is schizophrenia treated with antipsychotics?", "serious-mental-illness"},
		{"What are the symptoms of postpartum depression?", "postpartum"},
		{"How does cognitive behavioral therapy treat depression?", "depression-anxiety"},
		{"What medications help with insomnia?", "sleep-disorders"},
		{"How does alcohol withdrawal affect the brain?", "substance-use"},
		{"What is the mechanism of opioid addiction?", "substance-use"},

		// === ENDOCRINE & METABOLIC (43-50) ===
		{"What happens during diabetic ketoacidosis?", "diabetes-type2"},
		{"How is hypothyroidism diagnosed and treated?", "thyroid"},
		{"What is insulin resistance and how does it develop?", "diabetes-type2"},
		{"How does Graves' disease cause hyperthyroidism?", "thyroid"},
		{"What are the complications of uncontrolled diabetes?", "diabetes-type2"},
		{"How does polycystic ovary syndrome affect hormones?", "pcos"},
		{"What is metabolic syndrome?", "obesity-metabolic"},
		{"How does obesity increase the risk of type 2 diabetes?", "obesity-metabolic"},

		// === RESPIRATORY (51-56) ===
		{"What triggers an asthma attack?", "respiratory-infections"},
		{"How does COPD progress over time?", "chronic-respiratory"},
		{"What is the treatment for pneumonia?", "respiratory-infections"},
		{"How does cystic fibrosis affect the lungs?", "chronic-respiratory"},
		{"What is pulmonary fibrosis?", "chronic-respiratory"},
		{"How does bronchitis differ from pneumonia?", "respiratory-infections"},

		// === MUSCULOSKELETAL (57-60) ===
		{"What is the difference between osteoarthritis and rheumatoid arthritis?", "musculoskeletal"},
		{"How does osteoporosis develop and how is it prevented?", "musculoskeletal"},
		{"What causes lower back pain?", "back-spine"},
		{"How are herniated discs treated?", "back-spine"},

		// === PAIN (61-64) ===
		{"How do opioids relieve pain and why are they addictive?", "pain"},
		{"What is neuropathic pain and how is it treated?", "pain"},
		{"How do NSAIDs reduce inflammation?", "pain"},
		{"What is fibromyalgia and what causes it?", "pain"},

		// === DERMATOLOGY & ALLERGY (65-70) ===
		{"What causes eczema flare-ups?", "dermatology"},
		{"How do antihistamines work for allergic reactions?", "allergy-immunology"},
		{"What is psoriasis and how is it treated?", "dermatology"},
		{"What causes anaphylaxis?", "allergy-immunology"},
		{"How does immunotherapy work for allergies?", "allergy-immunology"},
		{"What is the treatment for severe acne?", "dermatology"},

		// === GASTROENTEROLOGY (71-76) ===
		{"What are the complications of Crohn's disease?", "gastrointestinal"},
		{"How does hepatitis C damage the liver?", "liver-disease"},
		{"What causes irritable bowel syndrome?", "gastrointestinal"},
		{"How is cirrhosis of the liver diagnosed?", "liver-disease"},
		{"What is celiac disease and what triggers it?", "gastrointestinal"},
		{"How does gastroesophageal reflux disease damage the esophagus?", "gastrointestinal"},

		// === RENAL & UROLOGICAL (77-80) ===
		{"What causes chronic kidney disease?", "kidney-urinary"},
		{"How are kidney stones removed?", "urinary-tract"},
		{"What is dialysis and when is it needed?", "kidney-urinary"},
		{"How does glomerulonephritis damage the kidneys?", "kidney-urinary"},

		// === REPRODUCTIVE HEALTH (81-86) ===
		{"What causes endometriosis?", "female-reproductive-health"},
		{"How does preeclampsia develop during pregnancy?", "pregnancy"},
		{"What are the risk factors for ectopic pregnancy?", "pregnancy"},
		{"What causes male infertility?", "male-reproductive-health"},
		{"How does testosterone replacement therapy work?", "male-reproductive-health"},
		{"What are the symptoms of menopause?", "female-reproductive-health"},

		// === HEMATOLOGY (87-90) ===
		{"What causes iron deficiency anemia?", "hematology"},
		{"How does sickle cell disease affect red blood cells?", "hematology"},
		{"What is deep vein thrombosis and how is it treated?", "hematology"},
		{"How does hemophilia affect blood clotting?", "hematology"},

		// === AUTOIMMUNE (91-94) ===
		{"How does lupus affect the body?", "autoimmune"},
		{"What triggers autoimmune diseases?", "autoimmune"},
		{"How is rheumatoid arthritis different from osteoarthritis?", "autoimmune"},
		{"What is the treatment for Hashimoto's thyroiditis?", "autoimmune"},

		// === EAR/NOSE/THROAT & OPHTHALMOLOGY (95-97) ===
		{"What causes hearing loss in older adults?", "ear-nose-throat"},
		{"How is glaucoma detected and treated?", "ophthalmology"},
		{"What causes chronic sinusitis?", "ear-nose-throat"},

		// === NUTRITION & GERIATRIC (98-100) ===
		{"How does vitamin D deficiency affect health?", "nutrition"},
		{"What are the health risks of B12 deficiency?", "nutrition"},
		{"How does aging affect the immune system?", "geriatric-health"},
	}

	backends := []struct {
		name string
		dir  string
	}{
		{"ladybug", "/data/topic-graphs/ladybug"},
	}

	// Open CSV file
	csvFile, err := os.Create(os.ExpandEnv("$HOME/router-test-100q.csv"))
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
		configs, routingTexts, err := router.DiscoverTopicGraphs(backend.dir)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Discovery failed for %s: %v\n", backend.name, err)
			continue
		}
		fmt.Printf("Discovered %d topic graphs\n", len(configs))

		// Use embedding strategy (pre-computes topic vectors, faster for 100 queries)
		routerConfig := router.RouterConfig{
			Strategy:      "embedding",
			MinConfidence: 0.1,
			MaxGraphs:     2,
			MergeStrategy: "rrf",
		}

		classifier, err := router.NewTopicClassifier(configs, routerConfig, emb, reranker, routingTexts)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Classifier creation failed: %v\n", err)
			continue
		}
		fmt.Println("Classifier ready")

		maxOpen := 5
		mgr := router.NewManagerWithOptions(configs, nil, emb, logger, maxOpen)
		mgr.SetCrossEncoder(reranker)
		rtr := router.NewRouter(mgr, classifier, routerConfig, logger)

		searchConfig := &types.SearchConfig{
			Limit:              10,
			CenterNodeDistance: 2,
			IncludeEdges:       true,
			Rerank:             true,
			ExcludeEntityTypes: []string{"CLINICAL_STUDY"},
			NodeConfig: &types.NodeSearchConfig{
				SearchMethods: []string{"cosine_similarity", "bm25"},
				Reranker:      "cross_encoder",
				MinScore:      0.0,
			},
			EdgeConfig: &types.EdgeSearchConfig{
				SearchMethods: []string{"bm25"},
				Reranker:      "cross_encoder",
				MinScore:      0.0,
			},
		}

		ctx := context.Background()
		totalStart := time.Now()

		for i, q := range queries {
			qNum := i + 1

			// Route
			routeStart := time.Now()
			info, err := rtr.GetRouteInfo(ctx, q.query)
			routeTime := time.Since(routeStart)

			if err != nil {
				fmt.Printf("Q%03d ROUTE ERROR: %v\n", qNum, err)
				w.Write([]string{
					backend.name, fmt.Sprint(qNum), q.query, q.expectTopic, "",
					fmt.Sprint(routeTime.Milliseconds()), "", "", "",
					"", "", "", "", "", "", "", err.Error(),
				})
				continue
			}

			routedTo := strings.Join(info.ClientNames, ",")

			// Search
			searchStart := time.Now()
			merged, err := rtr.SearchWithClients(ctx, q.query, info.ClientNames, searchConfig)
			searchTime := time.Since(searchStart)

			if err != nil {
				fmt.Printf("Q%03d SEARCH ERROR: %v\n", qNum, err)
				w.Write([]string{
					backend.name, fmt.Sprint(qNum), q.query, q.expectTopic, routedTo,
					fmt.Sprint(routeTime.Milliseconds()), fmt.Sprint(searchTime.Milliseconds()),
					"", "", "", "", "", "", "", "", "", err.Error(),
				})
				continue
			}

			// Collect errors
			var errStrs []string
			for clientName, searchErr := range merged.Errors {
				errStrs = append(errStrs, fmt.Sprintf("%s: %v", clientName, searchErr))
			}
			errStr := strings.Join(errStrs, "; ")

			// Primary topic match check
			primaryMatch := "MISS"
			for _, cn := range info.ClientNames {
				if cn == q.expectTopic {
					primaryMatch = "HIT"
					break
				}
			}

			// Top confidence
			topConf := 0.0
			topTopic := ""
			if len(info.TopicMatches) > 0 {
				topConf = info.TopicMatches[0].Confidence
				topTopic = info.TopicMatches[0].ClientName
			}

			// Compact log
			fmt.Printf("Q%03d [%s] %s → %s (top: %s %.3f) nodes=%d edges=%d route=%dms search=%dms",
				qNum, primaryMatch, q.expectTopic, routedTo,
				topTopic, topConf,
				len(merged.Nodes), len(merged.Edges),
				routeTime.Milliseconds(), searchTime.Milliseconds())
			if errStr != "" {
				fmt.Printf(" ERR: %s", errStr)
			}
			fmt.Println()

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
				src := merged.Sources[n.Uuid]
				w.Write([]string{
					backend.name, fmt.Sprint(qNum), q.query, q.expectTopic, routedTo,
					fmt.Sprint(routeTime.Milliseconds()), fmt.Sprint(searchTime.Milliseconds()),
					fmt.Sprint(len(merged.Nodes)), fmt.Sprint(len(merged.Edges)),
					n.Name, entityType, summary,
					"", "", "", strings.Join(src, ","),
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
					"0", "0", "", "", "", "", "", "", "", errStr,
				})
			}
		}

		elapsed := time.Since(totalStart)
		fmt.Printf("\n%d queries completed in %v (avg %.1fs/query)\n",
			len(queries), elapsed, elapsed.Seconds()/float64(len(queries)))

		if err := rtr.Close(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "Close error: %v\n", err)
		}
	}

	fmt.Printf("\nCSV written to ~/router-test-100q.csv\n")
}
