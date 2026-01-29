package server

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/soundprediction/predicato"
	"github.com/soundprediction/predicato/pkg/config"
	"github.com/soundprediction/predicato/pkg/server/handlers"
	"github.com/soundprediction/predicato/pkg/types"
)

// Server represents the HTTP server
type Server struct {
	config    *config.Config
	router    *chi.Mux
	predicato predicato.Predicato
	server    *http.Server
}

// New creates a new server instance
func New(cfg *config.Config, predicatoClient predicato.Predicato) *Server {
	return &Server{
		config:    cfg,
		predicato: predicatoClient,
	}
}

// Setup sets up the server routes and middleware
func (s *Server) Setup() {
	// Create router
	s.router = chi.NewRouter()

	// Add middleware
	s.router.Use(middleware.RequestID)
	s.router.Use(middleware.RealIP)
	s.router.Use(middleware.Logger)
	s.router.Use(middleware.Recoverer)
	s.router.Use(middleware.Timeout(60 * time.Second))
	s.router.Use(corsMiddleware)
	s.router.Use(contextMiddleware)

	// Setup routes
	s.setupRoutes()

	// Create HTTP server
	addr := fmt.Sprintf("%s:%d", s.config.Server.Host, s.config.Server.Port)
	s.server = &http.Server{
		Addr:    addr,
		Handler: s.router,
	}
}

// setupRoutes sets up all the routes
func (s *Server) setupRoutes() {
	// Create handlers
	healthHandler := handlers.NewHealthHandler(s.predicato)
	ingestHandler := handlers.NewIngestHandler(s.predicato)
	retrieveHandler := handlers.NewRetrieveHandler(s.predicato)
	configHandler := handlers.NewConfigHandler(s.config)
	nlpHandler := handlers.NewNLPHandler(s.predicato)

	// Health endpoints
	s.router.Get("/health", healthHandler.HealthCheck)
	s.router.Get("/healthcheck", healthHandler.HealthCheck) // Legacy endpoint
	s.router.Get("/ready", healthHandler.ReadinessCheck)
	s.router.Get("/live", healthHandler.LivenessCheck) // Kubernetes liveness probe
	s.router.Get("/health/detailed", healthHandler.DetailedHealthCheck)

	// API v1 routes
	s.router.Route("/api/v1", func(r chi.Router) {
		// Config routes - expose configured models (without API keys)
		r.Route("/config", func(r chi.Router) {
			r.Get("/models", configHandler.GetModels)
			r.Get("/nlp", configHandler.GetNLPModels)
			r.Get("/embedding", configHandler.GetEmbeddingModels)
		})

		// NLP utility routes - use server's configured LLM for analysis
		r.Route("/nlp", func(r chi.Router) {
			r.Post("/analyze-relevance", nlpHandler.AnalyzeRelevance)
			r.Post("/extract-source", nlpHandler.ExtractSource)
		})

		// Ingest routes
		r.Route("/ingest", func(r chi.Router) {
			r.Post("/messages", ingestHandler.AddMessages)
			r.Post("/entity", ingestHandler.AddEntityNode)
			r.Delete("/clear", ingestHandler.ClearData)
			// Two-stage ingestion
			r.Post("/extract", ingestHandler.ExtractEpisode)
			r.Post("/promote", ingestHandler.PromoteToGraph)
		})

		// Retrieve routes
		r.Post("/search", retrieveHandler.Search)
		r.Post("/search/facts", retrieveHandler.SearchFacts) // Fact store search with cosine similarity
		r.Get("/entity-edge/{uuid}", retrieveHandler.GetEntityEdge)
		r.Get("/episodes/{group_id}", retrieveHandler.GetEpisodes)
		r.Post("/get-memory", retrieveHandler.GetMemory)
	})

	// Legacy routes for compatibility with Python server
	s.router.Post("/search", retrieveHandler.Search)
	s.router.Get("/entity-edge/{uuid}", retrieveHandler.GetEntityEdge)
	s.router.Get("/episodes/{group_id}", retrieveHandler.GetEpisodes)
	s.router.Post("/get-memory", retrieveHandler.GetMemory)
}

// Start starts the server
func (s *Server) Start() error {
	log.Printf("Starting server on %s\n", s.server.Addr)
	return s.server.ListenAndServe()
}

// Stop stops the server gracefully
func (s *Server) Stop(ctx context.Context) error {
	log.Println("Stopping server...")
	return s.server.Shutdown(ctx)
}

// corsMiddleware adds CORS headers
func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Credentials", "true")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, accept, origin, Cache-Control, X-Requested-With")
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS, GET, PUT, DELETE")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// contextMiddleware extracts context information from headers
func contextMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()

		userID := r.Header.Get("X-User-ID")
		if userID != "" {
			ctx = context.WithValue(ctx, types.ContextKeyUserID, userID)
		}

		sessionID := r.Header.Get("X-Session-ID")
		if sessionID != "" {
			ctx = context.WithValue(ctx, types.ContextKeySessionID, sessionID)
		}

		ctx = context.WithValue(ctx, types.ContextKeyRequestSource, "server")

		next.ServeHTTP(w, r.WithContext(ctx))
	})
}
