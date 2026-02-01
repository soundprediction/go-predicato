package handlers

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/soundprediction/predicato"
	"github.com/soundprediction/predicato/pkg/server/dto"
	"github.com/soundprediction/predicato/pkg/types"
)

// IngestHandler handles data ingestion requests
type IngestHandler struct {
	predicato predicato.Predicato
}

// NewIngestHandler creates a new ingest handler
func NewIngestHandler(g predicato.Predicato) *IngestHandler {
	return &IngestHandler{
		predicato: g,
	}
}

// generateProcessID generates a unique process ID for tracking async operations
func generateProcessID() string {
	bytes := make([]byte, 8)
	if _, err := rand.Read(bytes); err != nil {
		// Fallback to timestamp if random generation fails
		return fmt.Sprintf("proc_%d", time.Now().UnixNano())
	}
	return fmt.Sprintf("proc_%s", hex.EncodeToString(bytes))
}

// writeErrorJSON writes an error response as JSON
func writeErrorJSON(w http.ResponseWriter, status int, errCode, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(dto.ErrorResponse{
		Error:   errCode,
		Message: message,
	})
}

// AddMessages handles POST /ingest/messages
func (h *IngestHandler) AddMessages(w http.ResponseWriter, r *http.Request) {
	var req dto.AddMessagesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	// Validate required fields
	if req.GroupID == "" {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", "group_id is required")
		return
	}

	if len(req.Messages) == 0 {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", "messages array cannot be empty")
		return
	}

	// Generate a process ID for tracking this async operation
	processID := generateProcessID()

	// Process messages asynchronously in the background with panic recovery
	go func() {
		// Recover from any panics in the background goroutine
		defer func() {
			if r := recover(); r != nil {
				log.Printf("[%s] PANIC recovered in message processing for group %s: %v\n", processID, req.GroupID, r)
			}
		}()

		ctx := context.Background()
		referenceTime := time.Now()
		if req.Reference != nil {
			referenceTime = *req.Reference
		}

		log.Printf("[%s] Starting processing of %d messages for group %s\n", processID, len(req.Messages), req.GroupID)

		// Convert messages to episodes and add them to predicato
		var episodes []types.Episode
		for i, msg := range req.Messages {
			// Generate a unique ID for each episode
			episodeID := fmt.Sprintf("%s-msg-%d-%d", req.GroupID, referenceTime.Unix(), i)

			// Create episode name from role and timestamp
			episodeName := fmt.Sprintf("%s message at %s", msg.Role, referenceTime.Format("2006-01-02 15:04:05"))

			// Create episode content
			episodeContent := fmt.Sprintf("%s: %s", msg.Role, msg.Content)

			// Use message timestamp if provided, otherwise use reference time
			episodeTime := referenceTime
			if msg.Timestamp != nil {
				episodeTime = *msg.Timestamp
			}

			episode := types.Episode{
				ID:        episodeID,
				Name:      episodeName,
				Content:   episodeContent,
				Reference: episodeTime,
				CreatedAt: time.Now(),
				GroupID:   req.GroupID,
				Metadata: map[string]interface{}{
					"role":             msg.Role,
					"original_content": msg.Content,
					"source":           "api_ingest",
					"process_id":       processID,
				},
			}

			episodes = append(episodes, episode)
		}

		// Add episodes to predicato
		if _, err := h.predicato.Add(ctx, episodes, nil); err != nil {
			// Log error but don't fail the entire request since it's async
			log.Printf("[%s] Error adding episodes to predicato for group %s: %v\n", processID, req.GroupID, err)
		} else {
			log.Printf("[%s] Successfully processed %d episodes for group %s\n", processID, len(episodes), req.GroupID)
		}
	}()

	writeJSON(w, http.StatusAccepted, dto.IngestResponse{
		Success:   true,
		Message:   fmt.Sprintf("Queued %d messages for processing", len(req.Messages)),
		ProcessID: processID,
	})
}

// AddEntityNode handles POST /ingest/entity
func (h *IngestHandler) AddEntityNode(w http.ResponseWriter, r *http.Request) {
	var req dto.AddEntityNodeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	// Validate required fields
	if req.GroupID == "" {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", "group_id is required")
		return
	}

	if req.Name == "" {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", "name is required")
		return
	}

	ctx := context.Background()

	// Create an episode that mentions this entity to add it to the knowledge graph
	// This leverages predicato's entity extraction capabilities
	now := time.Now()
	episodeID := fmt.Sprintf("%s-entity-%d", req.GroupID, now.Unix())

	// Create episode content that mentions the entity
	entityType := req.EntityType
	if entityType == "" {
		entityType = "entity"
	}

	episodeContent := fmt.Sprintf("New %s entity: %s", entityType, req.Name)
	if len(req.Attributes) > 0 {
		episodeContent += fmt.Sprintf(" with attributes: %v", req.Attributes)
	}

	// Create metadata that includes the entity information
	metadata := map[string]interface{}{
		"source":      "api_entity_ingest",
		"entity_name": req.Name,
		"entity_type": entityType,
	}

	// Add request attributes to metadata
	if req.Attributes != nil {
		for key, value := range req.Attributes {
			metadata["attr_"+key] = value
		}
	}

	episode := types.Episode{
		ID:        episodeID,
		Name:      fmt.Sprintf("Entity creation: %s", req.Name),
		Content:   episodeContent,
		Reference: now,
		CreatedAt: now,
		GroupID:   req.GroupID,
		Metadata:  metadata,
	}

	// Add the episode to predicato which will extract and create the entity
	if _, err := h.predicato.Add(ctx, []types.Episode{episode}, nil); err != nil {
		writeErrorJSON(w, http.StatusInternalServerError, "creation_failed", fmt.Sprintf("Failed to create entity node: %v", err))
		return
	}

	writeJSON(w, http.StatusCreated, dto.IngestResponse{
		Success: true,
		Message: fmt.Sprintf("Entity node '%s' created via episode processing", req.Name),
	})
}

// ClearData handles DELETE /ingest/clear
func (h *IngestHandler) ClearData(w http.ResponseWriter, r *http.Request) {
	var req dto.ClearDataRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	ctx := context.Background()

	// If no specific group IDs provided, this is a dangerous operation
	if len(req.GroupIDs) == 0 {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", "group_ids must be specified for data clearing. Clearing all data is not supported via API for safety.")
		return
	}

	// Process clearing for each specified group
	var successGroups []string
	var failedGroups []string

	for _, groupID := range req.GroupIDs {
		if groupID == "" {
			failedGroups = append(failedGroups, "(empty group ID)")
			continue
		}

		// Use predicato's ClearGraph method to clear data for this group
		if err := h.predicato.ClearGraph(ctx, groupID); err != nil {
			log.Printf("Error clearing data for group %s: %v\n", groupID, err)
			failedGroups = append(failedGroups, groupID)
		} else {
			log.Printf("Successfully cleared data for group %s\n", groupID)
			successGroups = append(successGroups, groupID)
		}
	}

	// Prepare response message
	var message string
	var success bool
	var statusCode int

	if len(failedGroups) == 0 {
		// All groups cleared successfully
		message = fmt.Sprintf("Successfully cleared data for groups: %v", successGroups)
		success = true
		statusCode = http.StatusOK
	} else if len(successGroups) == 0 {
		// All groups failed to clear
		message = fmt.Sprintf("Failed to clear data for all groups: %v", failedGroups)
		success = false
		statusCode = http.StatusInternalServerError
	} else {
		// Partial success
		message = fmt.Sprintf("Partially cleared data. Success: %v, Failed: %v", successGroups, failedGroups)
		success = true // Consider partial success as success
		statusCode = http.StatusOK
	}

	if !success {
		writeErrorJSON(w, statusCode, "clear_failed", message)
		return
	}

	writeJSON(w, statusCode, dto.IngestResponse{
		Success: success,
		Message: message,
	})
}

// ExtractEpisode handles POST /ingest/extract
// This is the first stage of two-stage ingestion - extracts entities and relationships
// without promoting to the graph.
func (h *IngestHandler) ExtractEpisode(w http.ResponseWriter, r *http.Request) {
	var req dto.ExtractEpisodeRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	// Validate the request
	if err := req.Validate(); err != nil {
		writeErrorJSON(w, http.StatusBadRequest, "validation_error", err.Error())
		return
	}

	ctx := context.Background()

	// Build episode from request
	now := time.Now()
	episodeID := req.ID
	if episodeID == "" {
		episodeID = fmt.Sprintf("%s-extract-%d", req.GroupID, now.UnixNano())
	}

	episode := types.Episode{
		ID:        episodeID,
		Name:      req.Name,
		Content:   req.Content,
		Source:    req.Source,
		GroupID:   req.GroupID,
		Metadata:  req.Metadata,
		CreatedAt: now,
	}
	if req.Reference != nil {
		episode.Reference = *req.Reference
	} else {
		episode.Reference = now
	}

	// Build options
	options := &predicato.AddEpisodeOptions{
		EntityTypes:          req.EntityTypes,
		ExcludedEntityTypes:  req.ExcludedEntityTypes,
		EdgeTypes:            req.EdgeTypes,
		PreviousEpisodeUUIDs: req.PreviousEpisodeUUIDs,
		MaxCharacters:        req.MaxCharacters,
		UseYAML:              req.UseYAML,
	}

	// Convert EdgeTypeMap if provided
	if req.EdgeTypeMap != nil {
		options.EdgeTypeMap = make(map[string]map[string][]interface{})
		for outerKey, innerMap := range req.EdgeTypeMap {
			options.EdgeTypeMap[outerKey] = make(map[string][]interface{})
			for innerKey, val := range innerMap {
				// Convert to []interface{}
				if arr, ok := val.([]interface{}); ok {
					options.EdgeTypeMap[outerKey][innerKey] = arr
				}
			}
		}
	}

	// Perform extraction
	results, err := h.predicato.ExtractToFacts(ctx, episode, options)
	if err != nil {
		writeErrorJSON(w, http.StatusInternalServerError, "extraction_failed", err.Error())
		return
	}

	// Convert to DTO
	response := dto.ExtractEpisodeResponse{
		Success:        true,
		SourceID:       results.SourceID,
		ChunkCount:     results.ChunkCount,
		ExtractionTime: results.ExtractionTime.String(),
		ExtractedNodes: make([]dto.ExtractedNodeDTO, 0, len(results.ExtractedNodes)),
		ExtractedEdges: make([]dto.ExtractedEdgeDTO, 0, len(results.ExtractedEdges)),
	}

	for _, n := range results.ExtractedNodes {
		response.ExtractedNodes = append(response.ExtractedNodes, dto.ExtractedNodeDTO{
			ID:          n.ID,
			SourceID:    n.SourceID,
			GroupID:     req.GroupID,
			Name:        n.Name,
			Type:        n.Type,
			Description: n.Description,
			ChunkIndex:  n.ChunkIndex,
			CreatedAt:   n.CreatedAt,
		})
	}

	for _, e := range results.ExtractedEdges {
		response.ExtractedEdges = append(response.ExtractedEdges, dto.ExtractedEdgeDTO{
			ID:             e.ID,
			SourceID:       e.SourceID,
			GroupID:        req.GroupID,
			SourceNodeName: e.SourceNodeName,
			SourceNodeType: e.SourceNodeType,
			TargetNodeName: e.TargetNodeName,
			TargetNodeType: e.TargetNodeType,
			Relation:       e.Relation,
			Description:    e.Description,
			Weight:         e.Weight,
			ChunkIndex:     e.ChunkIndex,
			CreatedAt:      e.CreatedAt,
		})
	}

	if results.Metadata != nil {
		response.Metadata = &dto.ExtractionMetadataDTO{
			ModelUsed:           results.Metadata.ModelUsed,
			EmbeddingModel:      results.Metadata.EmbeddingModel,
			EmbeddingDimensions: results.Metadata.EmbeddingDimensions,
			TotalTokens:         results.Metadata.TotalTokens,
		}
	}

	log.Printf("Extracted %d nodes and %d edges from episode %s in group %s\n",
		len(response.ExtractedNodes), len(response.ExtractedEdges), episodeID, req.GroupID)

	writeJSON(w, http.StatusOK, response)
}

// PromoteToGraph handles POST /ingest/promote
// This is the second stage of two-stage ingestion - promotes extracted knowledge to the graph.
func (h *IngestHandler) PromoteToGraph(w http.ResponseWriter, r *http.Request) {
	var req dto.PromoteToGraphRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeErrorJSON(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	// Validate the request
	if err := req.Validate(); err != nil {
		writeErrorJSON(w, http.StatusBadRequest, "validation_error", err.Error())
		return
	}

	ctx := context.Background()

	// Build options
	options := &predicato.AddEpisodeOptions{
		EntityTypes:          req.EntityTypes,
		EdgeTypes:            req.EdgeTypes,
		PreviousEpisodeUUIDs: req.PreviousEpisodeUUIDs,
		SkipReflexion:        req.SkipReflexion,
		SkipResolution:       req.SkipResolution,
		SkipAttributes:       req.SkipAttributes,
		SkipEdgeResolution:   req.SkipEdgeResolution,
		UseYAML:              req.UseYAML,
	}

	// Convert EdgeTypeMap if provided
	if req.EdgeTypeMap != nil {
		options.EdgeTypeMap = make(map[string]map[string][]interface{})
		for outerKey, innerMap := range req.EdgeTypeMap {
			options.EdgeTypeMap[outerKey] = make(map[string][]interface{})
			for innerKey, val := range innerMap {
				if arr, ok := val.([]interface{}); ok {
					options.EdgeTypeMap[outerKey][innerKey] = arr
				}
			}
		}
	}

	// Perform promotion
	results, err := h.predicato.PromoteToGraph(ctx, req.SourceID, options)
	if err != nil {
		writeErrorJSON(w, http.StatusInternalServerError, "promotion_failed", err.Error())
		return
	}

	// Convert to DTO
	response := dto.PromoteToGraphResponse{
		Success:        true,
		Nodes:          make([]dto.NodeDTO, 0, len(results.Nodes)),
		Edges:          make([]dto.EdgeDTO, 0, len(results.Edges)),
		EpisodicEdges:  make([]dto.EdgeDTO, 0, len(results.EpisodicEdges)),
		Communities:    make([]dto.NodeDTO, 0, len(results.Communities)),
		CommunityEdges: make([]dto.EdgeDTO, 0, len(results.CommunityEdges)),
	}

	// Convert episode
	if results.Episode != nil {
		response.Episode = nodeToDTO(results.Episode)
	}

	// Convert nodes
	for _, n := range results.Nodes {
		response.Nodes = append(response.Nodes, *nodeToDTO(n))
	}

	// Convert edges
	for _, e := range results.Edges {
		response.Edges = append(response.Edges, *edgeToDTO(e))
	}

	// Convert episodic edges
	for _, e := range results.EpisodicEdges {
		response.EpisodicEdges = append(response.EpisodicEdges, *edgeToDTO(e))
	}

	// Convert communities
	for _, n := range results.Communities {
		response.Communities = append(response.Communities, *nodeToDTO(n))
	}

	// Convert community edges
	for _, e := range results.CommunityEdges {
		response.CommunityEdges = append(response.CommunityEdges, *edgeToDTO(e))
	}

	log.Printf("Promoted source %s to graph: %d nodes, %d edges, %d communities\n",
		req.SourceID, len(response.Nodes), len(response.Edges), len(response.Communities))

	writeJSON(w, http.StatusOK, response)
}

// nodeToDTO converts a types.Node to dto.NodeDTO
func nodeToDTO(n *types.Node) *dto.NodeDTO {
	d := &dto.NodeDTO{
		UUID:       n.Uuid,
		GroupID:    n.GroupID,
		Name:       n.Name,
		Type:       string(n.Type),
		EntityType: n.EntityType,
		Summary:    n.Summary,
		CreatedAt:  n.CreatedAt,
		SourceIDs:  n.SourceIDs,
		Metadata:   n.Metadata,
	}
	if !n.UpdatedAt.IsZero() {
		d.UpdatedAt = &n.UpdatedAt
	}
	if !n.ValidFrom.IsZero() {
		d.ValidFrom = &n.ValidFrom
	}
	if n.ValidTo != nil {
		d.ValidTo = n.ValidTo
	}
	return d
}

// edgeToDTO converts a types.Edge to dto.EdgeDTO
func edgeToDTO(e *types.Edge) *dto.EdgeDTO {
	d := &dto.EdgeDTO{
		UUID:           e.Uuid,
		GroupID:        e.GroupID,
		SourceNodeUUID: e.SourceNodeID,
		TargetNodeUUID: e.TargetNodeID,
		Type:           string(e.Type),
		Name:           e.Name,
		Fact:           e.Fact,
		CreatedAt:      e.CreatedAt,
		Episodes:       e.Episodes,
		Metadata:       e.Metadata,
	}
	if e.ExpiredAt != nil {
		d.ExpiredAt = e.ExpiredAt
	}
	if !e.ValidFrom.IsZero() {
		d.ValidAt = &e.ValidFrom
	}
	if e.ValidTo != nil {
		d.InvalidAt = e.ValidTo
	}
	return d
}
