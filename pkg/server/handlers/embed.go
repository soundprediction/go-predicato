package handlers

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"

	"github.com/soundprediction/predicato/pkg/embedder"
)

// EmbedHandler handles embedding endpoint requests
type EmbedHandler struct {
	embedder embedder.Client
	model    string
}

// NewEmbedHandler creates a new embed handler.
// The model parameter is the name of the configured embedding model for inclusion in responses.
func NewEmbedHandler(e embedder.Client, model string) *EmbedHandler {
	return &EmbedHandler{
		embedder: e,
		model:    model,
	}
}

// EmbedRequest is the request body for the embed endpoint
type EmbedRequest struct {
	Texts []string `json:"texts"`
}

// EmbedResponse is the response body for the embed endpoint
type EmbedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
	Model      string      `json:"model"`
	Dimensions int         `json:"dimensions"`
}

// Embed handles POST /api/v1/embed
// Generates embeddings for the provided texts using the configured embedder
func (h *EmbedHandler) Embed(w http.ResponseWriter, r *http.Request) {
	if h.embedder == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error": "embedding model not configured",
		})
		return
	}

	var req EmbedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "invalid request body",
		})
		return
	}

	if len(req.Texts) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "texts field is required and must not be empty",
		})
		return
	}

	if len(req.Texts) > 100 {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": fmt.Sprintf("too many texts: %d (maximum 100)", len(req.Texts)),
		})
		return
	}

	for i, text := range req.Texts {
		if len(text) > 8192 {
			writeJSON(w, http.StatusBadRequest, map[string]string{
				"error": fmt.Sprintf("text at index %d exceeds maximum length of 8192 characters", i),
			})
			return
		}
	}

	ctx := r.Context()
	embeddings, err := h.embedder.Embed(ctx, req.Texts)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{
			"error": fmt.Sprintf("embedding failed: %v", err),
		})
		return
	}

	writeJSON(w, http.StatusOK, EmbedResponse{
		Embeddings: embeddings,
		Model:      h.model,
		Dimensions: h.embedder.Dimensions(),
	})
}

// OpenAI-compatible embeddings (POST /api/v1/embeddings) so any OpenAI-embeddings
// client can use predicato as a drop-in embedding provider. Because it runs the
// SAME embedder this process is configured with, embeddings are byte-for-byte
// identical to the in-process path — no re-embedding or consistency drift — which
// is exactly what lets the embedder run as a separate, poolable service in front
// of stores embedded by the same model.

type openAIEmbedRequest struct {
	Input json.RawMessage `json:"input"` // a single string or an array of strings
	Model string          `json:"model"`
	// EncodingFormat is "float" (array of numbers, OpenAI default) or "base64"
	// (little-endian float32 bytes, base64-encoded). The official openai clients
	// default to base64 on the wire, so we must honor it for drop-in compatibility.
	EncodingFormat string `json:"encoding_format"`
}

type openAIEmbedData struct {
	Object    string `json:"object"`
	Embedding any    `json:"embedding"` // []float32 (float) or string (base64)
	Index     int    `json:"index"`
}

// base64Float32 encodes a vector as the base64 of its little-endian float32 bytes,
// matching what OpenAI-compatible clients decode when encoding_format=base64.
func base64Float32(vec []float32) string {
	buf := make([]byte, 4*len(vec))
	for i, f := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return base64.StdEncoding.EncodeToString(buf)
}

type openAIEmbedResponse struct {
	Object string            `json:"object"`
	Data   []openAIEmbedData `json:"data"`
	Model  string            `json:"model"`
	Usage  map[string]int    `json:"usage"`
}

// OpenAIEmbeddings handles POST /api/v1/embeddings (OpenAI shape).
func (h *EmbedHandler) OpenAIEmbeddings(w http.ResponseWriter, r *http.Request) {
	if h.embedder == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"error": map[string]string{"message": "embedding model not configured"}})
		return
	}
	var req openAIEmbedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": map[string]string{"message": "invalid request body"}})
		return
	}
	// input may be a single string or an array of strings.
	var texts []string
	if err := json.Unmarshal(req.Input, &texts); err != nil {
		var one string
		if err2 := json.Unmarshal(req.Input, &one); err2 != nil {
			writeJSON(w, http.StatusBadRequest, map[string]any{"error": map[string]string{"message": "input must be a string or array of strings"}})
			return
		}
		texts = []string{one}
	}
	if len(texts) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]any{"error": map[string]string{"message": "input is required"}})
		return
	}
	embeddings, err := h.embedder.Embed(r.Context(), texts)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": map[string]string{"message": fmt.Sprintf("embedding failed: %v", err)}})
		return
	}
	base64Out := req.EncodingFormat == "base64"
	data := make([]openAIEmbedData, len(embeddings))
	for i, e := range embeddings {
		d := openAIEmbedData{Object: "embedding", Index: i}
		if base64Out {
			d.Embedding = base64Float32(e)
		} else {
			d.Embedding = e
		}
		data[i] = d
	}
	model := req.Model
	if model == "" {
		model = h.model
	}
	writeJSON(w, http.StatusOK, openAIEmbedResponse{
		Object: "list",
		Data:   data,
		Model:  model,
		Usage:  map[string]int{"prompt_tokens": 0, "total_tokens": 0},
	})
}
