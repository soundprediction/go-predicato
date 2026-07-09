package crossencoder

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRerankerClientAcceptsEmbedEverythingResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/rerank" {
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "zhiqing/Qwen3-Reranker-0.6B-ONNX",
			"results": []map[string]any{
				{"index": 1, "text": "beta", "relevance_score": 0.9},
				{"index": 0, "text": "alpha", "relevance_score": 0.2},
			},
		})
	}))
	defer srv.Close()

	client := NewRerankerClient(RerankerConfig{
		BaseURL: srv.URL + "/v1",
		Config:  Config{Model: "zhiqing/Qwen3-Reranker-0.6B-ONNX"},
	})
	results, err := client.Rank(context.Background(), "query", []string{"alpha", "beta"})
	if err != nil {
		t.Fatalf("Rank returned error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("len(results) = %d, want 2", len(results))
	}
	if results[0].Passage != "beta" || results[0].Score != 0.9 {
		t.Fatalf("top result = %+v, want beta/0.9", results[0])
	}
}

// vLLM's /v1/rerank returns "document" as an object ({"text": ...}) rather than
// a bare string. The client must tolerate that and still rank correctly.
func TestRerankerClientAcceptsVLLMObjectDocument(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/rerank" {
			t.Fatalf("unexpected path %s", r.URL.Path)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "Qwen3-Reranker-0.6B",
			"results": []map[string]any{
				{"index": 1, "document": map[string]any{"text": "beta", "multi_modal": nil}, "relevance_score": 0.88},
				{"index": 0, "document": map[string]any{"text": "alpha", "multi_modal": nil}, "relevance_score": 0.11},
			},
		})
	}))
	defer srv.Close()

	client := NewVLLMRerankerClient(srv.URL+"/v1", "Qwen3-Reranker-0.6B")
	results, err := client.Rank(context.Background(), "query", []string{"alpha", "beta"})
	if err != nil {
		t.Fatalf("Rank returned error: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("len(results) = %d, want 2", len(results))
	}
	if results[0].Passage != "beta" || results[0].Score != 0.88 {
		t.Fatalf("top result = %+v, want beta/0.88", results[0])
	}
}
