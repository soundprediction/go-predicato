package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/soundprediction/predicato/pkg/config"
)

func TestNew(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Host: "localhost",
			Port: 8080,
		},
	}

	// Test with nil predicato (server should still be created)
	server := New(cfg, nil, nil)
	if server == nil {
		t.Fatal("expected non-nil server")
	}

	if server.config != cfg {
		t.Error("expected config to be set")
	}
}

func TestSetup(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Host: "localhost",
			Port: 8080,
		},
	}

	server := New(cfg, nil, nil)
	server.Setup()

	if server.router == nil {
		t.Error("expected router to be initialized")
	}

	if server.server == nil {
		t.Error("expected http.Server to be initialized")
	}

	expectedAddr := "localhost:8080"
	if server.server.Addr != expectedAddr {
		t.Errorf("expected addr %s, got %s", expectedAddr, server.server.Addr)
	}
}

func TestHealthEndpoint(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Host: "localhost",
			Port: 8080,
		},
	}

	server := New(cfg, nil, nil)
	server.Setup()

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()

	server.router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}
}

func TestHealthcheckEndpoint(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Host: "localhost",
			Port: 8080,
		},
	}

	server := New(cfg, nil, nil)
	server.Setup()

	req := httptest.NewRequest(http.MethodGet, "/healthcheck", nil)
	w := httptest.NewRecorder()

	server.router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}
}

func TestLiveEndpoint(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Host: "localhost",
			Port: 8080,
		},
	}

	server := New(cfg, nil, nil)
	server.Setup()

	req := httptest.NewRequest(http.MethodGet, "/live", nil)
	w := httptest.NewRecorder()

	server.router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}
}

func TestReadyEndpoint(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Host: "localhost",
			Port: 8080,
		},
	}

	server := New(cfg, nil, nil)
	server.Setup()

	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	w := httptest.NewRecorder()

	server.router.ServeHTTP(w, req)

	// Without predicato, readiness check returns 503 Service Unavailable
	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("expected status 503 (no predicato), got %d", w.Code)
	}
}

func TestDetailedHealthEndpoint(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Host: "localhost",
			Port: 8080,
		},
	}

	server := New(cfg, nil, nil)
	server.Setup()

	req := httptest.NewRequest(http.MethodGet, "/health/detailed", nil)
	w := httptest.NewRecorder()

	server.router.ServeHTTP(w, req)

	// Without predicato, detailed health check returns 503 Service Unavailable
	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("expected status 503 (no predicato), got %d", w.Code)
	}
}

func TestCORSMiddleware(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Host: "localhost",
			Port: 8080,
		},
	}

	server := New(cfg, nil, nil)
	server.Setup()

	// Test OPTIONS request (CORS preflight)
	req := httptest.NewRequest(http.MethodOptions, "/health", nil)
	w := httptest.NewRecorder()

	server.router.ServeHTTP(w, req)

	// OPTIONS should return 204 No Content
	if w.Code != http.StatusNoContent {
		t.Errorf("expected status 204 for OPTIONS, got %d", w.Code)
	}

	// Check CORS headers
	if w.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Error("expected Access-Control-Allow-Origin header")
	}

	if w.Header().Get("Access-Control-Allow-Methods") == "" {
		t.Error("expected Access-Control-Allow-Methods header")
	}
}

func TestContextMiddleware(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Host: "localhost",
			Port: 8080,
		},
	}

	server := New(cfg, nil, nil)
	server.Setup()

	// Test with custom headers
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	req.Header.Set("X-User-ID", "test-user")
	req.Header.Set("X-Session-ID", "test-session")
	w := httptest.NewRecorder()

	server.router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}
}

func TestRouteExists(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Host: "localhost",
			Port: 8080,
		},
	}

	server := New(cfg, nil, nil)
	server.Setup()

	// Test that routes are registered (not 404)
	routes := []struct {
		method string
		path   string
	}{
		{http.MethodGet, "/health"},
		{http.MethodGet, "/healthcheck"},
		{http.MethodGet, "/ready"},
		{http.MethodGet, "/live"},
		{http.MethodGet, "/health/detailed"},
		// API routes (will fail without predicato but shouldn't be 404)
		{http.MethodPost, "/api/v1/ingest/messages"},
		{http.MethodPost, "/api/v1/ingest/entity"},
		{http.MethodDelete, "/api/v1/ingest/clear"},
		{http.MethodPost, "/api/v1/search"},
		// Legacy routes
		{http.MethodPost, "/search"},
		{http.MethodPost, "/get-memory"},
	}

	for _, route := range routes {
		t.Run(route.method+" "+route.path, func(t *testing.T) {
			req := httptest.NewRequest(route.method, route.path, nil)
			w := httptest.NewRecorder()

			server.router.ServeHTTP(w, req)

			if w.Code == http.StatusNotFound {
				t.Errorf("route %s %s returned 404, route not registered", route.method, route.path)
			}
		})
	}
}

func TestServerConfig(t *testing.T) {
	tests := []struct {
		name         string
		host         string
		port         int
		expectedAddr string
	}{
		{"localhost:8080", "localhost", 8080, "localhost:8080"},
		{"0.0.0.0:3000", "0.0.0.0", 3000, "0.0.0.0:3000"},
		{"127.0.0.1:9090", "127.0.0.1", 9090, "127.0.0.1:9090"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.Config{
				Server: config.ServerConfig{
					Host: tt.host,
					Port: tt.port,
				},
			}

			server := New(cfg, nil, nil)
			server.Setup()

			if server.server.Addr != tt.expectedAddr {
				t.Errorf("expected addr %s, got %s", tt.expectedAddr, server.server.Addr)
			}
		})
	}
}

func TestCORSHeaders(t *testing.T) {
	cfg := &config.Config{
		Server: config.ServerConfig{
			Host: "localhost",
			Port: 8080,
		},
	}

	server := New(cfg, nil, nil)
	server.Setup()

	// Regular GET request should also have CORS headers
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()

	server.router.ServeHTTP(w, req)

	expectedHeaders := []string{
		"Access-Control-Allow-Origin",
		"Access-Control-Allow-Credentials",
		"Access-Control-Allow-Headers",
		"Access-Control-Allow-Methods",
	}

	for _, header := range expectedHeaders {
		if w.Header().Get(header) == "" {
			t.Errorf("expected %s header to be set", header)
		}
	}
}
