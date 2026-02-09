package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/soundprediction/predicato/pkg/nlp"
)

// ExtractExtendedHandler handles extended extraction endpoint requests.
type ExtractExtendedHandler struct {
	nlpClient nlp.Client
}

// NewExtractExtendedHandler creates a new extract extended handler.
func NewExtractExtendedHandler(c nlp.Client) *ExtractExtendedHandler {
	return &ExtractExtendedHandler{
		nlpClient: c,
	}
}

// ExtractExtendedRequest is the request body for the extract endpoint.
type ExtractExtendedRequest struct {
	Text          string   `json:"text"`
	EntityTypes   []string `json:"entity_types"`
	RelationTypes []string `json:"relation_types"`
}

// ExtractExtended handles POST /api/v1/extract
// Runs extended extraction (entities, relations, triples, rules) using the configured NLP client.
func (h *ExtractExtendedHandler) ExtractExtended(w http.ResponseWriter, r *http.Request) {
	if h.nlpClient == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error": "NLP client not configured",
		})
		return
	}

	var req ExtractExtendedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "invalid request body",
		})
		return
	}

	if req.Text == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "text field is required and must not be empty",
		})
		return
	}

	if len(req.Text) > 8192 {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": fmt.Sprintf("text exceeds maximum length of 8192 characters (got %d)", len(req.Text)),
		})
		return
	}

	ctx := r.Context()
	result, err := h.nlpClient.ExtractExtended(ctx, req.Text, req.EntityTypes, req.RelationTypes)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{
			"error": fmt.Sprintf("extraction failed: %v", err),
		})
		return
	}

	if result == nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{
			"error": "extraction returned nil result",
		})
		return
	}

	writeJSON(w, http.StatusOK, result)
}
