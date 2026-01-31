package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
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

func setupRouter(client *gliner2.Client) *gin.Engine {
	router := gin.Default()

	// Health check endpoint
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":    "healthy",
			"models":    []string{"fastino/gliner2-base-v1", "fastino/gliner2-large-v1"},
			"timestamp": time.Now(),
		})
	})

	// Main GLInER2 endpoint (mirroring Fastino's /gliner-2)
	router.POST("/gliner-2", handleGLInER2Request(client))

	return router
}

func handleGLInER2Request(client *gliner2.Client) gin.HandlerFunc {
	return func(c *gin.Context) {
		var request gliner2.ExtractRequest
		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request format"})
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
				c.JSON(http.StatusInternalServerError, gin.H{"error": extractErr.Error()})
				return
			}
			result = gin.H{"entities": formatEntitiesForResponse(entities)}

		case "extract_relations":
			// GLInER2 calls relation extraction for facts
			schema := convertToStringArray(request.Schema)
			facts, extractErr := client.ExtractRelations(ctx, request.Text, schema)
			if extractErr != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": extractErr.Error()})
				return
			}
			result = gin.H{"relation_extraction": formatFactsForResponse(facts)}

		case "classify_text":
			// TODO: Implement text classification when GLInER2 client supports it
			c.JSON(http.StatusNotImplemented, gin.H{"error": "Text classification not yet implemented"})
			return

		case "extract_json":
			// TODO: Implement structured extraction when GLInER2 client supports it
			c.JSON(http.StatusNotImplemented, gin.H{"error": "Structured extraction not yet implemented"})
			return

		default:
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("Unsupported task: %s", request.Task)})
			return
		}

		// Return GLInER2 format response
		c.JSON(http.StatusOK, gin.H{"result": result})
	}
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

func shutdown(client *gliner2.Client) {
	if err := client.Close(); err != nil {
		log.Printf("Error closing client: %v", err)
	}
}
