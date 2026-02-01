package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	"github.com/go-chi/chi/v5"
	"github.com/soundprediction/predicato"
	"github.com/soundprediction/predicato/pkg/server/dto"
	"github.com/soundprediction/predicato/pkg/types"
)

// RetrieveHandler handles data retrieval requests
type RetrieveHandler struct {
	predicato predicato.Predicato
}

// NewRetrieveHandler creates a new retrieve handler
func NewRetrieveHandler(g predicato.Predicato) *RetrieveHandler {
	return &RetrieveHandler{
		predicato: g,
	}
}

// Search handles POST /search
func (h *RetrieveHandler) Search(w http.ResponseWriter, r *http.Request) {
	var req dto.SearchQuery
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	// Validate required fields
	if strings.TrimSpace(req.Query) == "" {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", "query field is required and cannot be empty")
		return
	}

	ctx := context.Background()

	// Set default max facts if not provided
	if req.MaxFacts <= 0 {
		req.MaxFacts = 10
	}

	// Create search configuration
	searchConfig := &types.SearchConfig{
		Limit:        req.MaxFacts,
		MinScore:     0.0,
		IncludeEdges: true,
		Rerank:       true,
	}

	// Perform the search using predicato
	// Note: Group ID filtering would be implemented in the search configuration
	// For now, we use the global search
	searchResults, err := h.predicato.Search(ctx, req.Query, searchConfig)
	if err != nil {
		writeErrorJSON(w, http.StatusInternalServerError, "search_failed", err.Error())
		return
	}

	// Convert predicato search results to DTO format
	var facts []dto.FactResult

	// Process nodes as facts
	for _, node := range searchResults.Nodes {
		fact := dto.FactResult{
			UUID:         node.Uuid,
			Fact:         h.nodeToFactDescription(node),
			SourceName:   node.Name,
			TargetName:   "", // Nodes don't have targets
			RelationType: string(node.Type),
			CreatedAt:    node.CreatedAt,
			ValidAt:      &node.ValidFrom,
		}

		if node.ValidTo != nil {
			fact.InvalidAt = node.ValidTo
		}

		facts = append(facts, fact)
	}

	// Process edges as facts
	for _, edge := range searchResults.Edges {
		fact := dto.FactResult{
			UUID:         edge.Uuid,
			Fact:         h.edgeToFactDescription(edge),
			SourceName:   edge.SourceID, // Could be enhanced to resolve actual names
			TargetName:   edge.TargetID,
			RelationType: string(edge.Type),
			CreatedAt:    edge.CreatedAt,
			ValidAt:      &edge.ValidFrom,
		}

		if edge.ValidTo != nil {
			fact.InvalidAt = edge.ValidTo
		}

		facts = append(facts, fact)
	}

	// Create response
	results := dto.SearchResults{
		Facts: facts,
		Total: len(facts),
	}

	writeJSON(w, http.StatusOK, results)
}

// GetEntityEdge handles GET /entity-edge/{uuid}
func (h *RetrieveHandler) GetEntityEdge(w http.ResponseWriter, r *http.Request) {
	uuid := chi.URLParam(r, "uuid")
	if uuid == "" {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", "UUID parameter is required")
		return
	}

	ctx := context.Background()

	// Try to retrieve the edge from predicato
	edge, err := h.predicato.GetEdge(ctx, uuid)
	if err != nil {
		// If edge not found, try as a node
		node, nodeErr := h.predicato.GetNode(ctx, uuid)
		if nodeErr != nil {
			writeErrorJSON(w, http.StatusNotFound, "entity_not_found", "Entity with the specified UUID was not found")
			return
		}

		// Convert node to fact format
		fact := dto.FactResult{
			UUID:         node.Uuid,
			Fact:         h.nodeToFactDescription(node),
			SourceName:   node.Name,
			TargetName:   "",
			RelationType: string(node.Type),
			CreatedAt:    node.CreatedAt,
			ValidAt:      &node.ValidFrom,
		}

		if node.ValidTo != nil {
			fact.InvalidAt = node.ValidTo
		}

		writeJSON(w, http.StatusOK, fact)
		return
	}

	// Convert edge to fact format
	fact := dto.FactResult{
		UUID:         edge.Uuid,
		Fact:         h.edgeToFactDescription(edge),
		SourceName:   edge.SourceID, // Could be enhanced to resolve actual names
		TargetName:   edge.TargetID,
		RelationType: string(edge.Type),
		CreatedAt:    edge.CreatedAt,
		ValidAt:      &edge.ValidFrom,
	}

	if edge.ValidTo != nil {
		fact.InvalidAt = edge.ValidTo
	}

	writeJSON(w, http.StatusOK, fact)
}

// GetEpisodes handles GET /episodes/{group_id}
func (h *RetrieveHandler) GetEpisodes(w http.ResponseWriter, r *http.Request) {
	groupID := chi.URLParam(r, "group_id")
	if groupID == "" {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", "Group ID parameter is required")
		return
	}

	// Parse query parameters
	lastNStr := r.URL.Query().Get("last_n")
	if lastNStr == "" {
		lastNStr = "10"
	}
	lastN, err := strconv.Atoi(lastNStr)
	if err != nil {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", "last_n must be a valid integer")
		return
	}

	// Ensure reasonable limits
	if lastN <= 0 {
		lastN = 10
	}
	if lastN > 100 {
		lastN = 100 // Cap at 100 for performance
	}

	ctx := context.Background()

	// Retrieve episodes from predicato
	episodeNodes, err := h.predicato.GetEpisodes(ctx, groupID, lastN)
	if err != nil {
		writeErrorJSON(w, http.StatusInternalServerError, "retrieval_failed", err.Error())
		return
	}

	// Convert nodes to episode DTOs
	var episodes []dto.Episode
	for _, node := range episodeNodes {
		episode := dto.Episode{
			UUID:      node.Uuid,
			GroupID:   node.GroupID,
			Content:   node.Content,
			CreatedAt: node.CreatedAt,
		}

		// Add source information if available in metadata
		if node.Metadata != nil {
			if source, ok := node.Metadata["source"].(string); ok {
				episode.Source = source
			}
		}

		episodes = append(episodes, episode)
	}

	response := dto.GetEpisodesResponse{
		Episodes: episodes,
		Total:    len(episodes),
	}

	writeJSON(w, http.StatusOK, response)
}

// GetMemory handles POST /get-memory
func (h *RetrieveHandler) GetMemory(w http.ResponseWriter, r *http.Request) {
	var req dto.GetMemoryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	// Validate required fields
	if len(req.Messages) == 0 {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", "messages array is required and cannot be empty")
		return
	}

	ctx := context.Background()

	// Set default max facts if not provided
	if req.MaxFacts <= 0 {
		req.MaxFacts = 10
	}

	// Compose query from messages
	var queryParts []string
	for _, msg := range req.Messages {
		if strings.TrimSpace(msg.Content) != "" {
			queryParts = append(queryParts, msg.Content)
		}
	}

	if len(queryParts) == 0 {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", "at least one message must have non-empty content")
		return
	}

	combinedQuery := strings.Join(queryParts, " ")

	// Create search configuration for memory retrieval
	searchConfig := &types.SearchConfig{
		Limit:        req.MaxFacts,
		MinScore:     0.1, // Slightly higher threshold for memory relevance
		IncludeEdges: true,
		Rerank:       true,
	}

	// Perform search using the combined query
	searchResults, err := h.predicato.Search(ctx, combinedQuery, searchConfig)
	if err != nil {
		writeErrorJSON(w, http.StatusInternalServerError, "memory_retrieval_failed", err.Error())
		return
	}

	// Convert search results to memory facts
	var facts []dto.FactResult

	// Process nodes as memory facts
	for _, node := range searchResults.Nodes {
		// Prioritize episodic nodes for memory retrieval
		fact := dto.FactResult{
			UUID:         node.Uuid,
			Fact:         h.nodeToFactDescription(node),
			SourceName:   node.Name,
			TargetName:   "",
			RelationType: string(node.Type),
			CreatedAt:    node.CreatedAt,
			ValidAt:      &node.ValidFrom,
		}

		if node.ValidTo != nil {
			fact.InvalidAt = node.ValidTo
		}

		facts = append(facts, fact)
	}

	// Process edges as relationship facts
	for _, edge := range searchResults.Edges {
		fact := dto.FactResult{
			UUID:         edge.Uuid,
			Fact:         h.edgeToFactDescription(edge),
			SourceName:   edge.SourceID,
			TargetName:   edge.TargetID,
			RelationType: string(edge.Type),
			CreatedAt:    edge.CreatedAt,
			ValidAt:      &edge.ValidFrom,
		}

		if edge.ValidTo != nil {
			fact.InvalidAt = edge.ValidTo
		}

		facts = append(facts, fact)
	}

	// Create response
	results := dto.GetMemoryResponse{
		Facts: facts,
		Total: len(facts),
	}

	writeJSON(w, http.StatusOK, results)
}

// Helper methods for converting graph entities to fact descriptions

// nodeToFactDescription converts a node to a human-readable fact description
func (h *RetrieveHandler) nodeToFactDescription(node *types.Node) string {
	if node.Summary != "" {
		return node.Summary
	}
	if node.Content != "" {
		return node.Content
	}
	return node.Name + " is a " + string(node.Type)
}

// edgeToFactDescription converts an edge to a human-readable fact description
func (h *RetrieveHandler) edgeToFactDescription(edge *types.Edge) string {
	if edge.Summary != "" {
		return edge.Summary
	}
	if edge.Name != "" {
		return edge.Name
	}
	return edge.SourceID + " " + string(edge.Type) + " " + edge.TargetID
}

// SearchFacts handles POST /search/facts
// This searches the structured fact store using vector similarity (cosine) and/or keyword search
func (h *RetrieveHandler) SearchFacts(w http.ResponseWriter, r *http.Request) {
	var req dto.SearchFactsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	// Validate required fields
	if strings.TrimSpace(req.Query) == "" {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", "query field is required and cannot be empty")
		return
	}

	ctx := context.Background()

	// Set default limit if not provided
	if req.Limit <= 0 {
		req.Limit = 10
	}

	// Create search configuration with filters
	searchConfig := &types.SearchConfig{
		Limit:    req.Limit,
		MinScore: req.MinScore,
	}

	// Add group ID filter if provided
	if req.GroupID != "" {
		searchConfig.Filters = &types.SearchFilters{
			GroupIDs: []string{req.GroupID},
		}
	}

	// Perform the search using predicato's SearchFacts method
	searchResults, err := h.predicato.SearchFacts(ctx, req.Query, searchConfig)
	if err != nil {
		writeErrorJSON(w, http.StatusInternalServerError, "search_failed", err.Error())
		return
	}

	// Convert factstore results to DTO format
	var nodes []dto.ExtractedNodeDTO
	for _, node := range searchResults.Nodes {
		nodeDTO := dto.ExtractedNodeDTO{
			ID:          node.ID,
			SourceID:    node.SourceID,
			GroupID:     node.GroupID,
			Name:        node.Name,
			Type:        node.Type,
			Description: node.Description,
			ChunkIndex:  node.ChunkIndex,
			CreatedAt:   node.CreatedAt,
		}
		nodes = append(nodes, nodeDTO)
	}

	var edges []dto.ExtractedEdgeDTO
	for _, edge := range searchResults.Edges {
		edgeDTO := dto.ExtractedEdgeDTO{
			ID:             edge.ID,
			SourceID:       edge.SourceID,
			GroupID:        edge.GroupID,
			SourceNodeName: edge.SourceNodeName,
			SourceNodeType: edge.SourceNodeType,
			TargetNodeName: edge.TargetNodeName,
			TargetNodeType: edge.TargetNodeType,
			Relation:       edge.Relation,
			Description:    edge.Description,
			Weight:         edge.Weight,
			ChunkIndex:     edge.ChunkIndex,
			CreatedAt:      edge.CreatedAt,
		}
		edges = append(edges, edgeDTO)
	}

	response := dto.SearchFactsResponse{
		Success:    true,
		Nodes:      nodes,
		Edges:      edges,
		NodeScores: searchResults.NodeScores,
		EdgeScores: searchResults.EdgeScores,
		Query:      searchResults.Query,
		Total:      searchResults.Total,
	}

	writeJSON(w, http.StatusOK, response)
}
