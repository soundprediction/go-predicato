package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/soundprediction/predicato/pkg/gliner2"
	"github.com/soundprediction/predicato/pkg/nlp"
)

func main() {
	// Load configuration from environment variables
	config := loadConfig()

	// Initialize GLInER2 client
	client, err := gliner2.NewClient(config)
	if err != nil {
		log.Fatalf("Failed to create GLInER2 client: %v", err)
	}

	// Setup graceful shutdown
	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan
		log.Println("Shutting down gracefully...")
		shutdown(client)
		os.Exit(0)
	}()

	// Start HTTP server
	log.Printf("Starting GLInER2 API server on %s", config.Local.Endpoint)
	router := setupRouter(client)

	srv := &http.Server{
		Addr:     config.Local.Endpoint,
		Handler:  router,
		ErrorLog: log.New(os.Stderr, "", log.LstdFlags),
	}

	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server failed to start: %v", err)
	}
}

func loadConfig() gliner2.Config {
	// Default configuration
	endpoint := "0.0.0.0:8000"
	timeout := 30 * time.Second

	// Override from environment
	if port := os.Getenv("PORT"); port != "" {
		endpoint = "0.0.0.0:" + port
	}
	if envEndpoint := os.Getenv("GLINER2_ENDPOINT"); envEndpoint != "" {
		endpoint = envEndpoint
	}

	return gliner2.Config{
		Provider: gliner2.ProviderLocal,
		Local: &gliner2.LocalConfig{
			Endpoint: endpoint,
			Timeout:  timeout,
		},
	}
}

func setupRouter(client *gliner2.Client) *chi.Mux {
	r := chi.NewRouter()
	r.Use(middleware.Logger)
	r.Use(middleware.Recoverer)

	// Health check endpoint
	r.Get("/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"status":    "healthy",
			"models":    []string{"fastino/gliner2-base-v1", "fastino/gliner2-large-v1"},
			"timestamp": time.Now(),
		})
	})

	// Main GLInER2 endpoint (mirroring Fastino's /gliner-2)
	r.Post("/gliner-2", handleGLInER2Request(client))

	return r
}

func handleGLInER2Request(client *gliner2.Client) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var request gliner2.ExtractRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "Invalid request format"})
			return
		}

		ctx := context.Background()
		var result interface{}

		// Route to appropriate method based on task
		switch request.Task {
		case "extract_entities":
			// Convert GLInER2 entity types format to simple string array
			schema := convertToStringArray(request.Schema)
			entities, extractErr := client.ExtractEntities(ctx, request.Text, schema)
			if extractErr != nil {
				writeJSON(w, http.StatusInternalServerError, map[string]string{"error": extractErr.Error()})
				return
			}
			result = map[string]interface{}{"entities": formatEntitiesForResponse(entities)}

		case "extract_relations":
			// GLInER2 calls relation extraction for facts
			schema := convertToStringArray(request.Schema)
			facts, extractErr := client.ExtractRelations(ctx, request.Text, schema)
			if extractErr != nil {
				writeJSON(w, http.StatusInternalServerError, map[string]string{"error": extractErr.Error()})
				return
			}
			result = map[string]interface{}{"relation_extraction": formatFactsForResponse(facts)}

		case "classify_text":
			writeJSON(w, http.StatusNotImplemented, map[string]string{"error": "Text classification not yet implemented"})
			return

		case "extract_json":
			// Detect extended extraction: schema has "entities" key
			schemaMap, isMap := request.Schema.(map[string]interface{})
			if !isMap {
				writeJSON(w, http.StatusBadRequest, map[string]string{"error": "extract_json requires a map schema"})
				return
			}
			_, isExtended := schemaMap["entities"]
			if !isExtended {
				writeJSON(w, http.StatusNotImplemented, map[string]string{"error": "Simple structured extraction not yet implemented; use extended extraction with {\"entities\": true}"})
				return
			}

			extResult, extractErr := client.ExtractExtended(ctx, request.Text, nil, nil)
			if extractErr != nil {
				writeJSON(w, http.StatusInternalServerError, map[string]string{"error": extractErr.Error()})
				return
			}

			// Format response to match the Python server's to_go_client_dict() output
			result = formatExtendedForResponse(extResult)

		default:
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": fmt.Sprintf("Unsupported task: %s", request.Task)})
			return
		}

		// Return GLInER2 format response
		writeJSON(w, http.StatusOK, map[string]interface{}{"result": result})
	}
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func convertToStringArray(schema interface{}) []string {
	switch v := schema.(type) {
	case []string:
		return v
	case []interface{}:
		result := make([]string, len(v))
		for i, item := range v {
			if str, ok := item.(string); ok {
				result[i] = str
			}
		}
		return result
	default:
		return nil
	}
}

func formatEntitiesForResponse(entities []nlp.ExtractedEntity) map[string][]map[string]interface{} {
	formatted := make(map[string][]map[string]interface{})

	// Group by label
	labelToEntities := make(map[string][]nlp.ExtractedEntity)
	for _, entity := range entities {
		labelToEntities[entity.Label] = append(labelToEntities[entity.Label], entity)
	}

	// Convert to GLInER2 API format
	for label, entityList := range labelToEntities {
		formattedList := make([]map[string]interface{}, 0, len(entityList))
		for _, entity := range entityList {
			entityMap := map[string]interface{}{
				"text": entity.Text,
			}
			if entity.Confidence > 0 {
				entityMap["confidence"] = entity.Confidence
			}
			if entity.Start >= 0 {
				entityMap["start"] = entity.Start
			}
			if entity.End > 0 {
				entityMap["end"] = entity.End
			}
			formattedList = append(formattedList, entityMap)
		}
		formatted[label] = formattedList
	}
	return formatted
}

func formatFactsForResponse(facts []nlp.ExtractedRelation) map[string][]map[string]interface{} {
	formatted := make(map[string][]map[string]interface{})
	relationMap := make(map[string][]nlp.ExtractedRelation)

	// Group facts by relation type
	for _, fact := range facts {
		relationMap[fact.Type] = append(relationMap[fact.Type], fact)
	}

	// Convert to GLInER2 format
	for relationType, tuples := range relationMap {
		formattedTuples := make([]map[string]interface{}, 0, len(tuples))
		for _, tuple := range tuples {
			tupleMap := map[string]interface{}{
				"head": tuple.Source,
				"tail": tuple.Target,
			}
			formattedTuples = append(formattedTuples, tupleMap)
		}
		formatted[relationType] = formattedTuples
	}

	return formatted
}

func formatExtendedForResponse(r *nlp.ExtendedExtractionResult) map[string]interface{} {
	// Entities: map[type][]string (pass through)
	entities := r.Entities

	// Relations: group by type
	relExtraction := make(map[string][]map[string]interface{})
	for _, rel := range r.Relations {
		entry := map[string]interface{}{"head": rel.Source, "tail": rel.Target}
		if rel.Confidence > 0 {
			entry["confidence"] = rel.Confidence
		}
		relExtraction[rel.Type] = append(relExtraction[rel.Type], entry)
	}

	// Facts (triples)
	var facts []map[string]interface{}
	for _, t := range r.Triples {
		f := map[string]interface{}{
			"subject":   t.Subject,
			"predicate": t.Predicate,
			"object":    t.Object,
		}
		if t.Condition != "" {
			f["condition"] = t.Condition
		}
		if t.Temporal != "" {
			f["temporal"] = t.Temporal
		}
		if t.Location != "" {
			f["location"] = t.Location
		}
		if t.Certainty != "" {
			f["certainty"] = t.Certainty
		}
		if t.Scope != "" {
			f["scope"] = t.Scope
		}
		if t.SourceAttribution != "" {
			f["source_attribution"] = t.SourceAttribution
		}
		if t.Confidence > 0 {
			f["confidence"] = t.Confidence
		}
		facts = append(facts, f)
	}

	// Rules
	var rules []map[string]interface{}
	for _, r := range r.Rules {
		rule := map[string]interface{}{
			"antecedent": r.Antecedent,
			"consequent": r.Consequent,
		}
		if r.Exception != "" {
			rule["exception"] = r.Exception
		}
		if r.RuleType != "" {
			rule["rule_type"] = r.RuleType
		}
		if r.Scope != "" {
			rule["scope"] = r.Scope
		}
		if r.SourceAttribution != "" {
			rule["source_attribution"] = r.SourceAttribution
		}
		if r.Confidence > 0 {
			rule["confidence"] = r.Confidence
		}
		rules = append(rules, rule)
	}

	return map[string]interface{}{
		"entities":            entities,
		"relation_extraction": relExtraction,
		"fact":                facts,
		"rule":                rules,
	}
}

func shutdown(client *gliner2.Client) {
	if err := client.Close(); err != nil {
		log.Printf("Error closing client: %v", err)
	}
}
