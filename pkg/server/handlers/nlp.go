package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/soundprediction/predicato"
)

// NLPHandler handles NLP-related requests
type NLPHandler struct {
	predicato predicato.Predicato
}

// NewNLPHandler creates a new NLP handler
func NewNLPHandler(p predicato.Predicato) *NLPHandler {
	return &NLPHandler{
		predicato: p,
	}
}

// AnalyzeRelevanceRequest is the request body for relevance analysis
type AnalyzeRelevanceRequest struct {
	Content  string   `json:"content"`
	Topics   []string `json:"topics,omitempty"`
	MaxChars int      `json:"max_chars,omitempty"`
}

// AnalyzeRelevanceResponse is the response for relevance analysis
type AnalyzeRelevanceResponse struct {
	IsRelevant bool   `json:"is_relevant"`
	Reason     string `json:"reason,omitempty"`
}

// ExtractSourceRequest is the request body for source extraction
type ExtractSourceRequest struct {
	Content string `json:"content"`
	URL     string `json:"url"`
}

// ExtractSourceResponse is the response for source extraction
type ExtractSourceResponse struct {
	SourceName string `json:"source_name"`
	SourceType string `json:"source_type"`
}

// AnalyzeRelevance handles POST /api/v1/nlp/analyze-relevance
// Uses the server's configured LLM to analyze content relevance
func (h *NLPHandler) AnalyzeRelevance(w http.ResponseWriter, r *http.Request) {
	var req AnalyzeRelevanceRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "Invalid request body",
		})
		return
	}

	if req.Content == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "Content is required",
		})
		return
	}

	// Default topics if not provided
	topics := req.Topics
	if len(topics) == 0 {
		topics = []string{
			"Pregnancy (prenatal care, childbirth, fertility, newborn health, maternal health)",
			"Nutrition (diet, healthy eating, vitamins, minerals, food, wellness)",
			"Diabetes (blood sugar, insulin, glucose, type 1, type 2, gestational diabetes)",
		}
	}

	// Truncate content if needed
	maxChars := req.MaxChars
	if maxChars <= 0 {
		maxChars = 15000
	}
	content := req.Content
	if len(content) > maxChars {
		content = content[:maxChars]
	}

	// Use predicato's NLP client to analyze relevance
	ctx := r.Context()
	isRelevant, err := h.predicato.AnalyzeContentRelevance(ctx, content, topics)
	if err != nil {
		// Log error but return a response (fallback to false)
		writeJSON(w, http.StatusOK, AnalyzeRelevanceResponse{
			IsRelevant: false,
			Reason:     "Analysis failed: " + err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, AnalyzeRelevanceResponse{
		IsRelevant: isRelevant,
	})
}

// ExtractSource handles POST /api/v1/nlp/extract-source
// Uses the server's configured LLM to extract source metadata
func (h *NLPHandler) ExtractSource(w http.ResponseWriter, r *http.Request) {
	var req ExtractSourceRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "Invalid request body",
		})
		return
	}

	if req.Content == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "Content is required",
		})
		return
	}

	// Truncate content for source extraction
	content := req.Content
	if len(content) > 5000 {
		content = content[:5000]
	}

	// Use predicato's NLP client to extract source info
	ctx := r.Context()
	sourceName, sourceType, err := h.predicato.ExtractSourceInfo(ctx, content, req.URL)
	if err != nil {
		// Return defaults on error
		writeJSON(w, http.StatusOK, ExtractSourceResponse{
			SourceName: "Unknown",
			SourceType: "other",
		})
		return
	}

	writeJSON(w, http.StatusOK, ExtractSourceResponse{
		SourceName: sourceName,
		SourceType: sourceType,
	})
}
