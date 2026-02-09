package handlers

import (
	"encoding/json"
	"fmt"
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
