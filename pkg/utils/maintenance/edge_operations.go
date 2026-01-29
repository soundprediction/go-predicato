package maintenance

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"reflect"
	"strings"
	"time"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/embedder"
	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/prompts"
	"github.com/soundprediction/predicato/pkg/types"
	"github.com/soundprediction/predicato/pkg/utils"
)

// EdgeOperations provides edge-related maintenance operations
type EdgeOperations struct {
	driver      driver.GraphDriver
	nlProcessor nlp.Client
	embedder    embedder.Client
	prompts     prompts.Library
	logger      *slog.Logger

	// Specialized NLP clients
	ExtractionNLP nlp.Client
	ResolutionNLP nlp.Client

	// Skip flags
	SkipResolution bool

	// Format flags
	UseYAML bool
}

// NewEdgeOperations creates a new EdgeOperations instance
func NewEdgeOperations(driver driver.GraphDriver, nlProcessor nlp.Client, embedder embedder.Client, prompts prompts.Library) *EdgeOperations {
	return &EdgeOperations{
		driver:      driver,
		nlProcessor: nlProcessor,
		embedder:    embedder,
		prompts:     prompts,
		logger:      slog.Default(), // Use default logger, can be overridden
	}
}

// SetLogger sets a custom logger for the EdgeOperations
func (eo *EdgeOperations) SetLogger(logger *slog.Logger) {
	eo.logger = logger
}

// Helper methods to get the appropriate LLM client with fallback to default
func (eo *EdgeOperations) getExtractionNLP() nlp.Client {
	if eo.ExtractionNLP != nil {
		return eo.ExtractionNLP
	}
	return eo.nlProcessor
}

func (eo *EdgeOperations) getResolutionNLP() nlp.Client {
	if eo.ResolutionNLP != nil {
		return eo.ResolutionNLP
	}
	return eo.nlProcessor
}

// BuildEpisodicEdges creates episodic edges from entity nodes to an episode
func (eo *EdgeOperations) BuildEpisodicEdges(ctx context.Context, entityNodes []*types.Node, episodeUUID string, createdAt time.Time) ([]*types.Edge, error) {
	if len(entityNodes) == 0 {
		return []*types.Edge{}, nil
	}

	episodicEdges := make([]*types.Edge, 0, len(entityNodes))

	for _, node := range entityNodes {
		edge := types.NewEntityEdge(
			utils.GenerateUUID(),
			episodeUUID,
			node.Uuid,
			node.GroupID,
			"MENTIONED_IN",
			types.EpisodicEdgeType,
		)
		edge.UpdatedAt = createdAt
		edge.ValidFrom = createdAt
		episodicEdges = append(episodicEdges, edge)
	}

	log.Printf("Built %d episodic edges", len(episodicEdges))
	return episodicEdges, nil
}

// BuildDuplicateOfEdges creates IS_DUPLICATE_OF edges between duplicate node pairs
func (eo *EdgeOperations) BuildDuplicateOfEdges(ctx context.Context, episode *types.Node, createdAt time.Time, duplicateNodes []NodePair) ([]*types.Edge, error) {
	duplicateEdges := make([]*types.Edge, 0, len(duplicateNodes))

	for _, pair := range duplicateNodes {
		if pair.Source.Uuid == pair.Target.Uuid {
			continue
		}

		fact := fmt.Sprintf("%s is a duplicate of %s", pair.Source.Name, pair.Target.Name)

		edge := types.NewEntityEdge(
			utils.GenerateUUID(),
			pair.Source.Uuid,
			pair.Target.Uuid,
			episode.GroupID,
			"IS_DUPLICATE_OF",
			types.EntityEdgeType,
		)
		edge.Summary = fact
		edge.Fact = fact
		edge.UpdatedAt = createdAt
		edge.ValidFrom = createdAt
		edge.SourceIDs = []string{episode.Uuid}

		duplicateEdges = append(duplicateEdges, edge)
	}

	return duplicateEdges, nil
}

// ExtractEdges extracts relationship edges from episode content using LLM
func (eo *EdgeOperations) ExtractEdges(ctx context.Context, episode *types.Node, nodes []*types.Node, previousEpisodes []*types.Node, edgeTypeMap map[string][][]string, edgeTypes map[string]interface{}, groupID string) ([]*types.Edge, error) {
	start := time.Now()

	if len(nodes) == 0 {
		return []*types.Edge{}, nil
	}

	// Batch processing for large node sets to avoid overwhelming the LLM
	const batchSize = 15
	if len(nodes) > batchSize {
		eo.logger.Info("Batching edge extraction",
			"total_nodes", len(nodes),
			"batch_size", batchSize,
			"num_batches", (len(nodes)+batchSize-1)/batchSize)

		var allEdges []*types.Edge
		for i := 0; i < len(nodes); i += batchSize {
			end := i + batchSize
			if end > len(nodes) {
				end = len(nodes)
			}
			batch := nodes[i:end]

			eo.logger.Debug("Processing edge extraction batch",
				"batch_start", i,
				"batch_end", end,
				"batch_nodes", len(batch))

			batchEdges, err := eo.extractEdgesBatch(ctx, episode, batch, nodes, previousEpisodes, edgeTypeMap, edgeTypes, groupID, i)
			if err != nil {
				return nil, fmt.Errorf("failed to extract edges from batch %d-%d: %w", i, end, err)
			}
			allEdges = append(allEdges, batchEdges...)
		}

		eo.logger.Info("Completed batched edge extraction",
			"total_nodes", len(nodes),
			"total_edges", len(allEdges),
			"duration", time.Since(start))

		return allEdges, nil
	}

	// For small node sets, process directly
	return eo.extractEdgesBatch(ctx, episode, nodes, nodes, previousEpisodes, edgeTypeMap, edgeTypes, groupID, 0)
}

// extractEdgesBatch extracts edges for a batch of nodes
func (eo *EdgeOperations) extractEdgesBatch(ctx context.Context, episode *types.Node, batchNodes []*types.Node, allNodes []*types.Node, previousEpisodes []*types.Node, edgeTypeMap map[string][][]string, edgeTypes map[string]interface{}, groupID string, batchOffset int) ([]*types.Edge, error) {
	start := time.Now()

	if len(batchNodes) == 0 {
		return []*types.Edge{}, nil
	}

	// Prepare edge types context as a slice for TSV formatting
	edgeTypesContext := []map[string]interface{}{}
	if edgeTypeMap != nil {
		for typeName := range edgeTypes {
			edgeTypesContext = append(edgeTypesContext, map[string]interface{}{
				"fact_type_name":        typeName,
				"fact_type_description": fmt.Sprintf("custom type: %s", typeName),
				"fact_type_signature":   edgeTypeMap[typeName], // Include the signature for source/target entity types
			})
		}
	}

	// Prepare context for LLM using batch nodes
	// Note: Data is passed as slices for TSV formatting in prompts
	nodeContexts := make([]map[string]interface{}, len(batchNodes))
	for i, node := range batchNodes {
		nodeContexts[i] = map[string]interface{}{
			"id":           i,
			"name":         node.Name,
			"entity_types": []string{string(node.EntityType)}, // Simplified for now
		}
	}

	previousEpisodeContents := make([]string, len(previousEpisodes))
	for i, ep := range previousEpisodes {
		previousEpisodeContents[i] = ep.Summary
	}

	promptContext := map[string]interface{}{
		"episode_content":   episode.Content,
		"nodes":             nodeContexts,
		"previous_episodes": previousEpisodeContents,
		"reference_time":    episode.ValidFrom,
		"edge_types":        edgeTypesContext,
		"custom_prompt":     "",
		"ensure_ascii":      true,
		"logger":            eo.logger,
	}

	// Extract edges using LLM
	messages, err := eo.prompts.ExtractEdges().Edge().Call(promptContext)
	if err != nil {
		return nil, fmt.Errorf("failed to create prompt: %w", err)
	}

	// Create CSV parser function for ExtractedEdge
	csvParser := func(csvContent string) ([]*prompts.ExtractedEdge, error) {
		return utils.UnmarshalCSV[prompts.ExtractedEdge](csvContent, '\t')
	}

	// Use GenerateCSVResponse for robust CSV parsing with retries
	extractedEdgeSlice, badResp, err := nlp.GenerateCSVResponse[prompts.ExtractedEdge](
		ctx,
		eo.getExtractionNLP(),
		eo.logger,
		messages,
		csvParser,
		0, // maxRetries (use default of 8)
	)

	if err != nil {
		// Log detailed error information
		if badResp != nil {
			eo.logger.Error("Failed to extract edges from CSV",
				"error", badResp.Error,
				"response_length", len(badResp.Response),
				"num_messages", len(badResp.Messages))
			if badResp.Response != "" {
				fmt.Printf("\nFailed LLM edge extraction response:\n%v\n\n", badResp.Response)
			}
		}
		return []*types.Edge{}, fmt.Errorf("failed to unmarshal extracted edges: %w", err)
	}

	// Convert to ExtractedEdges struct
	var extractedEdges prompts.ExtractedEdges
	extractedEdges.Edges = extractedEdgeSlice

	log.Printf("Extracted %d edges in %v", len(extractedEdges.Edges), time.Since(start))

	if len(extractedEdges.Edges) == 0 {
		return []*types.Edge{}, nil
	}

	// Convert to Edge objects
	edges := make([]*types.Edge, 0, len(extractedEdges.Edges))
	for _, edgeData := range extractedEdges.Edges {
		// Validate node indices (relative to batch)
		if edgeData.SourceID < 0 || edgeData.SourceID >= len(batchNodes) ||
			edgeData.TargetID < 0 || edgeData.TargetID >= len(batchNodes) {
			log.Printf("Warning: invalid node indices for edge %s (batch has %d nodes)", edgeData.Name, len(batchNodes))
			continue
		}

		sourceNode := batchNodes[edgeData.SourceID]
		targetNode := batchNodes[edgeData.TargetID]

		// Parse temporal information
		var validAt time.Time
		var validTo *time.Time

		if edgeData.ValidAt != "" {
			// Strip any surrounding quotes (can happen with double JSON encoding)
			cleanValidAt := strings.Trim(edgeData.ValidAt, "\"")
			if parsed, err := time.Parse(time.RFC3339, strings.ReplaceAll(cleanValidAt, "Z", "+00:00")); err == nil {
				validAt = parsed.UTC()
			} else if cleanValidAt == "null" {
				validAt = episode.ValidFrom
			} else {
				log.Printf("Warning: failed to parse valid_at date '%s': %v", cleanValidAt, err)
				validAt = episode.ValidFrom
			}
		} else {
			validAt = episode.ValidFrom
		}

		if edgeData.InvalidAt != "" {
			// Strip any surrounding quotes (can happen with double JSON encoding)
			cleanInvalidAt := strings.Trim(edgeData.InvalidAt, "\"")
			if parsed, err := time.Parse(time.RFC3339, strings.ReplaceAll(cleanInvalidAt, "Z", "+00:00")); err == nil {
				parsedUTC := parsed.UTC()
				validTo = &parsedUTC
			} else if cleanInvalidAt == "null" {
				validTo = nil
			} else {
				log.Printf("Warning: failed to parse invalid_at date '%s': %v", cleanInvalidAt, err)
			}
		}

		edge := types.NewEntityEdge(
			utils.GenerateUUID(),
			sourceNode.Uuid,
			targetNode.Uuid,
			groupID,
			edgeData.Name,
			types.EntityEdgeType,
		)
		edge.Summary = edgeData.Summary
		edge.Fact = edgeData.Fact
		edge.UpdatedAt = time.Now().UTC()
		edge.ValidFrom = validAt
		edge.ValidTo = validTo
		edge.SourceIDs = []string{episode.Uuid}

		edges = append(edges, edge)
		log.Printf("Created edge: %s from %s to %s", edge.Name, sourceNode.Name, targetNode.Name)
	}

	return edges, nil
}

// GetBetweenNodes retrieves edges between two specific nodes using the proper Ladybug query pattern
func (eo *EdgeOperations) GetBetweenNodes(ctx context.Context, sourceNodeID, targetNodeID string) ([]*types.Edge, error) {
	query := `
		MATCH (a:Entity {uuid: $source_uuid})-[:RELATES_TO]->(rel:RelatesToNode_)-[:RELATES_TO]->(b:Entity {uuid: $target_uuid})
		RETURN rel.uuid AS uuid, rel.name AS name, rel.fact AS fact, rel.group_id AS group_id,
		       rel.created_at AS created_at, rel.valid_at AS valid_at, rel.invalid_at AS invalid_at,
		       rel.expired_at AS expired_at, rel.episodes AS episodes, rel.attributes AS attributes,
		       a.uuid AS source_id, b.uuid AS target_id
		UNION
		MATCH (a:Entity {uuid: $target_uuid})-[:RELATES_TO]->(rel:RelatesToNode_)-[:RELATES_TO]->(b:Entity {uuid: $source_uuid})
		RETURN rel.uuid AS uuid, rel.name AS name, rel.fact AS fact, rel.group_id AS group_id,
		       rel.created_at AS created_at, rel.valid_at AS valid_at, rel.invalid_at AS invalid_at,
		       rel.expired_at AS expired_at, rel.episodes AS episodes, rel.attributes AS attributes,
		       a.uuid AS source_id, b.uuid AS target_id
	`

	params := map[string]interface{}{
		"source_uuid": sourceNodeID,
		"target_uuid": targetNodeID,
	}

	result, _, _, err := eo.driver.ExecuteQuery(ctx, query, params)
	if err != nil {
		return nil, fmt.Errorf("failed to execute GetBetweenNodes query: %w", err)
	}

	// Convert result to Edge objects
	var edges []*types.Edge
	if result != nil {
		// Handle different result types based on driver implementation
		switch records := result.(type) {
		case []map[string]interface{}:
			for _, record := range records {
				edge, err := eo.convertRecordToEdge(record)
				if err != nil {
					log.Printf("Warning: failed to convert record to edge: %v", err)
					continue
				}
				edges = append(edges, edge)
			}
		default:
			// Try to handle Neo4j/Memgraph []*db.Record using reflection
			edges = eo.parseNeo4jEdgeRecords(result)
		}
	}

	return edges, nil
}

// convertRecordToEdge converts a database record to an Edge object
func (eo *EdgeOperations) convertRecordToEdge(record map[string]interface{}) (*types.Edge, error) {
	edge := &types.Edge{}

	// Extract basic fields
	if uuid, ok := record["uuid"].(string); ok {
		edge.Uuid = uuid
	} else {
		return nil, fmt.Errorf("missing or invalid uuid field")
	}

	if name, ok := record["name"].(string); ok {
		edge.Name = name
	}

	if fact, ok := record["fact"].(string); ok {
		edge.Summary = fact
	}

	if groupID, ok := record["group_id"].(string); ok {
		edge.GroupID = groupID
	}

	// Extract source and target IDs
	if sourceID, ok := record["source_id"].(string); ok {
		edge.SourceID = sourceID
	}
	if targetID, ok := record["target_id"].(string); ok {
		edge.TargetID = targetID
	}

	// Extract timestamps
	if createdAt, ok := record["created_at"].(time.Time); ok {
		edge.CreatedAt = createdAt
	}
	if updatedAt, ok := record["updated_at"].(time.Time); ok {
		edge.UpdatedAt = updatedAt
	}
	if validFrom, ok := record["valid_from"].(time.Time); ok {
		edge.ValidFrom = validFrom
	}
	if validTo, ok := record["valid_to"].(time.Time); ok {
		edge.ValidTo = &validTo
	}

	// Set edge type - assume EntityEdge for relationships from RelatesToNode_
	edge.Type = types.EntityEdgeType

	// Extract source IDs if present
	if sourceIDs, ok := record["source_ids"].([]interface{}); ok {
		strSourceIDs := make([]string, len(sourceIDs))
		for i, id := range sourceIDs {
			if strID, ok := id.(string); ok {
				strSourceIDs[i] = strID
			}
		}
		edge.SourceIDs = strSourceIDs
	}

	return edge, nil
}

// ResolveExtractedEdges resolves newly extracted edges with existing ones in the graph
func (eo *EdgeOperations) ResolveExtractedEdges(ctx context.Context, extractedEdges []*types.Edge, episode *types.Node, entities []*types.Node, createEmbeddings bool, edgeTypes map[string]interface{}) ([]*types.Edge, []*types.Edge, error) {
	if len(extractedEdges) == 0 {
		return []*types.Edge{}, []*types.Edge{}, nil
	}

	// Create entity UUID to node mapping for quick lookup
	entityMap := make(map[string]*types.Node)
	for _, entity := range entities {
		entityMap[entity.Uuid] = entity
	}

	resolvedEdges := make([]*types.Edge, 0, len(extractedEdges))
	invalidatedEdges := make([]*types.Edge, 0)

	// Check if resolution should be skipped
	if eo.SkipResolution {
		eo.logger.Info("Skipping edge resolution as requested")
		return bypassResolveEdges(ctx, extractedEdges)
	}

	// Process each extracted edge
	for _, extractedEdge := range extractedEdges {
		// Create embeddings for the edge
		if err := eo.createEdgeEmbedding(ctx, extractedEdge); err != nil {
			log.Printf("Warning: failed to create embedding for edge: %v", err)
		}

		// Get existing edges between the same nodes
		existingEdges, err := eo.GetBetweenNodes(ctx, extractedEdge.SourceID, extractedEdge.TargetID)
		if err != nil {
			log.Printf("Warning: failed to get existing edges: %v", err)
			existingEdges = []*types.Edge{}
		}

		// Search for related edges using semantic search
		relatedEdges, err := eo.searchRelatedEdges(ctx, extractedEdge, existingEdges)
		if err != nil {
			log.Printf("Warning: failed to search related edges: %v", err)
			relatedEdges = []*types.Edge{}
		}

		// Resolve the edge against existing ones
		resolvedEdge, newlyInvalidated, err := eo.resolveExtractedEdge(ctx, extractedEdge, relatedEdges, existingEdges, episode, edgeTypes)
		if err != nil {
			log.Printf("Warning: failed to resolve edge: %v", err)
			// Use the original edge if resolution fails
			resolvedEdge = extractedEdge
		}

		// If the edge is a duplicate, add episode to existing edge
		if resolvedEdge != extractedEdge && episode != nil {
			// Add episode to source IDs if not already present
			found := false
			for _, sourceID := range resolvedEdge.SourceIDs {
				if sourceID == episode.Uuid {
					found = true
					break
				}
			}
			if !found {
				resolvedEdge.SourceIDs = append(resolvedEdge.SourceIDs, episode.Uuid)
				resolvedEdge.UpdatedAt = time.Now().UTC()
			}
		}

		resolvedEdges = append(resolvedEdges, resolvedEdge)
		invalidatedEdges = append(invalidatedEdges, newlyInvalidated...)
	}

	if createEmbeddings {
		// Create embeddings for all resolved and invalidated edges
		allEdges := append(resolvedEdges, invalidatedEdges...)
		for _, edge := range allEdges {
			if err := eo.createEdgeEmbedding(ctx, edge); err != nil {
				log.Printf("Warning: failed to create embedding for edge: %v", err)
			}
		}
	}

	log.Printf("Resolved %d edges, invalidated %d edges", len(resolvedEdges), len(invalidatedEdges))
	return resolvedEdges, invalidatedEdges, nil
}

// createEdgeEmbedding creates an embedding for an edge based on its summary
func (eo *EdgeOperations) createEdgeEmbedding(ctx context.Context, edge *types.Edge) error {
	if edge.Summary == "" {
		return nil
	}
	if eo.embedder == nil {
		return nil
	}
	embedding, err := eo.embedder.EmbedSingle(ctx, edge.Summary)
	if err != nil {
		return fmt.Errorf("failed to create embedding: %w", err)
	}

	edge.Embedding = embedding
	return nil
}

// searchRelatedEdges searches for semantically related edges using hybrid search with UUID filtering
func (eo *EdgeOperations) searchRelatedEdges(ctx context.Context, extractedEdge *types.Edge, existingEdges []*types.Edge) ([]*types.Edge, error) {
	if extractedEdge.Summary == "" {
		return []*types.Edge{}, nil
	}

	// Create UUID filter for existing edges (equivalent to Python's SearchFilters(edge_uuids=...))
	edgeUUIDs := make([]string, len(existingEdges))
	for i, edge := range existingEdges {
		edgeUUIDs[i] = edge.Uuid
	}

	// Create a map for quick UUID lookup
	validUUIDs := make(map[string]bool)
	for _, uuid := range edgeUUIDs {
		validUUIDs[uuid] = true
	}

	// Use hybrid search with proper filtering
	// This is equivalent to Python's EDGE_HYBRID_SEARCH_RRF config
	searchOptions := &driver.SearchOptions{
		Limit:     50,
		EdgeTypes: []types.EdgeType{types.EntityEdgeType},
		// Note: GroupIDs filtering would need to be added to SearchOptions
	}

	// Search for edges using semantic similarity (fact content)
	edges, err := eo.driver.SearchEdges(ctx, extractedEdge.Summary, extractedEdge.GroupID, searchOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to search related edges: %w", err)
	}

	// Filter results to only include edges in the UUID filter
	var relatedEdges []*types.Edge
	for _, edge := range edges {
		// Only include edges that are in the valid UUID set
		if len(edgeUUIDs) == 0 || validUUIDs[edge.Uuid] {
			// Exclude the extracted edge itself
			if edge.Uuid != extractedEdge.Uuid {
				relatedEdges = append(relatedEdges, edge)
			}
		}
	}

	log.Printf("Found %d related edges for edge fact: %s", len(relatedEdges), extractedEdge.Summary)
	return relatedEdges, nil
}

// resolveExtractedEdge resolves a single extracted edge against existing edges
func (eo *EdgeOperations) resolveExtractedEdge(ctx context.Context, extractedEdge *types.Edge, relatedEdges []*types.Edge, existingEdges []*types.Edge, episode *types.Node, edgeTypes map[string]interface{}) (*types.Edge, []*types.Edge, error) {
	if len(relatedEdges) == 0 && len(existingEdges) == 0 {
		return extractedEdge, []*types.Edge{}, nil
	}

	start := time.Now()

	// Prepare context for LLM deduplication
	relatedEdgesContext := make([]map[string]interface{}, len(relatedEdges))
	for i, edge := range relatedEdges {
		relatedEdgesContext[i] = map[string]interface{}{
			"id":   edge.Uuid,
			"fact": edge.Summary,
		}
	}

	invalidationCandidatesContext := make([]map[string]interface{}, len(existingEdges))
	for i, edge := range existingEdges {
		invalidationCandidatesContext[i] = map[string]interface{}{
			"id":   i,
			"fact": edge.Summary,
		}
	}

	// Build edge_types_context for deduplication prompt
	// Note: This context is simpler than the extraction context - it only includes name and description
	// Equivalent to Python (lines 497-507):
	// edge_types_context = (
	//     [
	//         {
	//             'fact_type_name': type_name,
	//             'fact_type_description': type_model.__doc__,
	//         }
	//         for type_name, type_model in edge_type_candidates.items()
	//     ]
	//     if edge_type_candidates is not None
	//     else []
	// )

	// Convert edge types map to slice format for TSV formatting in prompts
	edgeTypesContext := []map[string]interface{}{}
	for typeName := range edgeTypes {
		edgeTypesContext = append(edgeTypesContext, map[string]interface{}{
			"fact_type_name":        typeName,
			"fact_type_description": fmt.Sprintf("custom type: %s", typeName),
		})
	}

	// Note: Data is passed as slices for TSV formatting in prompts
	promptContext := map[string]interface{}{
		"existing_edges":               relatedEdgesContext,
		"new_edge":                     extractedEdge.Summary,
		"episode_content":              episode.Content,
		"invalidation_candidate_edges": invalidationCandidatesContext,
		"edge_types":                   edgeTypesContext, // Python passes this to prompt
		"ensure_ascii":                 true,
		"logger":                       eo.logger,
		"use_yaml":                     eo.UseYAML,
	}

	messages, err := eo.prompts.DedupeEdges().ResolveEdge().Call(promptContext)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create deduplication prompt: %w", err)
	}

	// Create CSV parser function for EdgeDuplicateTSV
	csvParser := func(csvContent string) ([]*prompts.EdgeDuplicateTSV, error) {
		return utils.UnmarshalCSV[prompts.EdgeDuplicateTSV](csvContent, '\t')
	}

	var edgeDuplicateTSVSlice []prompts.EdgeDuplicateTSV
	var badResp *types.BadLlmCsvResponse

	if eo.UseYAML {
		// Create YAML parser function for EdgeDuplicateTSV
		yamlParser := func(yamlContent string) ([]*prompts.EdgeDuplicateTSV, error) {
			return utils.UnmarshalYAML[prompts.EdgeDuplicateTSV](yamlContent)
		}

		// Use GenerateYAMLResponse
		edgeDuplicateTSVSlice, badResp, err = nlp.GenerateYAMLResponse[prompts.EdgeDuplicateTSV](
			ctx,
			eo.getResolutionNLP(),
			eo.logger,
			messages,
			yamlParser,
			3, // maxRetries
		)
	} else {
		// Use GenerateCSVResponse
		edgeDuplicateTSVSlice, badResp, err = nlp.GenerateCSVResponse[prompts.EdgeDuplicateTSV](
			ctx,
			eo.getResolutionNLP(),
			eo.logger,
			messages,
			csvParser,
			3, // maxRetries
		)
	}

	if err != nil {
		// Log detailed error information
		if badResp != nil {
			eo.logger.Warn("Failed to resolve edge duplicates from CSV",
				"error", badResp.Error,
				"response_length", len(badResp.Response),
				"num_messages", len(badResp.Messages))
			if badResp.Response != "" {
				fmt.Printf("\nFailed LLM edge resolution response:\n%v\n\n", badResp.Response)
			}
		}
		log.Printf("Warning: failed to parse edge deduplication TSV: %v", err)
		return extractedEdge, []*types.Edge{}, nil
	}

	if len(edgeDuplicateTSVSlice) == 0 {
		log.Printf("Warning: empty edge deduplication response")
		return extractedEdge, []*types.Edge{}, nil
	}

	// Convert TSV result to EdgeDuplicate
	edgeDuplicateTSV := &edgeDuplicateTSVSlice[0]
	var edgeDuplicate prompts.EdgeDuplicate
	edgeDuplicate.FactType = edgeDuplicateTSV.FactType
	edgeDuplicate.DuplicateFacts = edgeDuplicateTSV.DuplicateFacts
	edgeDuplicate.ContradictedFacts = edgeDuplicateTSV.ContradictedFacts

	// Process duplicate facts - find edges by UUID
	resolvedEdge := extractedEdge
	for _, duplicateFactUUID := range edgeDuplicate.DuplicateFacts {
		// Find the edge with matching UUID in relatedEdges
		for _, edge := range relatedEdges {
			if edge.Uuid == duplicateFactUUID {
				resolvedEdge = edge
				break
			}
		}
		if resolvedEdge != extractedEdge {
			break // Found a duplicate, stop searching
		}
	}

	// Process contradicted facts (invalidation candidates) - find edges by UUID
	var invalidatedEdges []*types.Edge
	for _, contradictedFactUUID := range edgeDuplicate.ContradictedFacts {
		// Find the edge with matching UUID in existingEdges
		for _, edge := range existingEdges {
			if edge.Uuid == contradictedFactUUID {
				// Apply temporal logic for invalidation
				invalidatedEdge := eo.resolveEdgeContradictions(resolvedEdge, []*types.Edge{edge})
				invalidatedEdges = append(invalidatedEdges, invalidatedEdge...)
				break
			}
		}
	}

	// Update fact type if specified
	if edgeDuplicate.FactType != "" && strings.ToUpper(edgeDuplicate.FactType) != "DEFAULT" {
		resolvedEdge.Name = edgeDuplicate.FactType
	}

	// Note: Temporal invalidation logic is handled elsewhere.
	// If resolvedEdge.ValidTo is set and in the past, the edge is already expired.
	// We don't modify expiration here as it's handled by the graph driver.

	log.Printf("Resolved edge %s in %v", extractedEdge.Name, time.Since(start))
	return resolvedEdge, invalidatedEdges, nil
}

// resolveEdgeContradictions handles temporal contradictions between edges
func (eo *EdgeOperations) resolveEdgeContradictions(resolvedEdge *types.Edge, invalidationCandidates []*types.Edge) []*types.Edge {
	if len(invalidationCandidates) == 0 {
		return []*types.Edge{}
	}

	now := time.Now().UTC()
	var invalidatedEdges []*types.Edge

	for _, edge := range invalidationCandidates {
		// Skip edges that are already invalid before the new edge becomes valid
		if edge.ValidTo != nil && resolvedEdge.ValidFrom.After(*edge.ValidTo) {
			continue
		}

		// Skip if new edge is invalid before the candidate becomes valid
		if resolvedEdge.ValidTo != nil && edge.ValidFrom.After(*resolvedEdge.ValidTo) {
			continue
		}

		// Invalidate edge if the new edge becomes valid after this one
		if edge.ValidFrom.Before(resolvedEdge.ValidFrom) {
			edgeCopy := *edge
			validTo := resolvedEdge.ValidFrom
			edgeCopy.ValidTo = &validTo
			edgeCopy.UpdatedAt = now
			invalidatedEdges = append(invalidatedEdges, &edgeCopy)
		}
	}

	return invalidatedEdges
}

// FilterExistingDuplicateOfEdges filters out duplicate node pairs that already have IS_DUPLICATE_OF edges using proper Ladybug query
func (eo *EdgeOperations) FilterExistingDuplicateOfEdges(ctx context.Context, duplicateNodePairs []NodePair) ([]NodePair, error) {
	if len(duplicateNodePairs) == 0 {
		return []NodePair{}, nil
	}

	// Prepare parameters exactly like Python implementation
	duplicateNodeUUIDs := make([]map[string]interface{}, len(duplicateNodePairs))
	for i, pair := range duplicateNodePairs {
		duplicateNodeUUIDs[i] = map[string]interface{}{
			"src": pair.Source.Uuid,
			"dst": pair.Target.Uuid,
		}
	}

	query := `
		UNWIND $duplicate_node_uuids AS duplicate
		MATCH (n:Entity {uuid: duplicate.src})-[:RELATES_TO]->(e:RelatesToNode_ {name: 'IS_DUPLICATE_OF'})-[:RELATES_TO]->(m:Entity {uuid: duplicate.dst})
		RETURN DISTINCT
			n.uuid AS source_uuid,
			m.uuid AS target_uuid
	`

	params := map[string]interface{}{
		"duplicate_node_uuids": duplicateNodeUUIDs,
	}

	result, _, _, err := eo.driver.ExecuteQuery(ctx, query, params)
	if err != nil {
		return nil, fmt.Errorf("failed to execute FilterExistingDuplicateOfEdges query: %w", err)
	}

	// Create a set of existing duplicate pairs
	existingPairs := make(map[string]bool)
	if result != nil {
		switch records := result.(type) {
		case []map[string]interface{}:
			// Ladybug format
			for _, record := range records {
				if sourceUUID, ok := record["source_uuid"].(string); ok {
					if targetUUID, ok := record["target_uuid"].(string); ok {
						key := fmt.Sprintf("%s-%s", sourceUUID, targetUUID)
						existingPairs[key] = true
					}
				}
			}
		default:
			// Try Neo4j/Memgraph format using reflection
			value := reflect.ValueOf(result)
			if value.Kind() == reflect.Slice {
				for i := 0; i < value.Len(); i++ {
					record := value.Index(i)

					// Call Get() method on the record
					getMethod := record.MethodByName("Get")
					if !getMethod.IsValid() {
						continue
					}

					// Get source_uuid
					sourceResults := getMethod.Call([]reflect.Value{reflect.ValueOf("source_uuid")})
					if len(sourceResults) == 0 {
						continue
					}

					// Get target_uuid
					targetResults := getMethod.Call([]reflect.Value{reflect.ValueOf("target_uuid")})
					if len(targetResults) == 0 {
						continue
					}

					sourceInterface := sourceResults[0].Interface()
					targetInterface := targetResults[0].Interface()

					if sourceUUID, ok := sourceInterface.(string); ok {
						if targetUUID, ok := targetInterface.(string); ok {
							key := fmt.Sprintf("%s-%s", sourceUUID, targetUUID)
							existingPairs[key] = true
						}
					}
				}
			} else {
				log.Printf("Warning: unexpected result type from FilterExistingDuplicateOfEdges query: %T", result)
			}
		}
	}

	// Filter out pairs that already exist
	var filteredPairs []NodePair
	for _, pair := range duplicateNodePairs {
		key := fmt.Sprintf("%s-%s", pair.Source.Uuid, pair.Target.Uuid)
		if !existingPairs[key] {
			filteredPairs = append(filteredPairs, pair)
		}
	}

	log.Printf("Filtered %d duplicate node pairs, %d remain after filtering existing IS_DUPLICATE_OF edges",
		len(duplicateNodePairs)-len(filteredPairs), len(filteredPairs))

	return filteredPairs, nil
}

// parseNeo4jEdgeRecords parses Neo4j/Memgraph driver records into edges using reflection.
// This handles the []*db.Record type returned by Memgraph's ExecuteQuery for edge queries.
func (eo *EdgeOperations) parseNeo4jEdgeRecords(result interface{}) []*types.Edge {
	var edges []*types.Edge

	// Use reflection to handle Neo4j driver records
	value := reflect.ValueOf(result)
	if value.Kind() != reflect.Slice {
		log.Printf("Warning: expected slice for Neo4j records, got %T", result)
		return edges
	}

	// Iterate through records
	for i := 0; i < value.Len(); i++ {
		record := value.Index(i)

		// Get Keys and Values from the record
		keysMethod := record.MethodByName("Keys")
		valuesField := record.FieldByName("Values")

		if !keysMethod.IsValid() || !valuesField.IsValid() {
			continue
		}

		// Call Keys() method
		keysResult := keysMethod.Call([]reflect.Value{})
		if len(keysResult) < 1 {
			continue
		}

		keys := keysResult[0]
		if keys.Kind() != reflect.Slice {
			continue
		}

		// Get Values field
		values := valuesField
		if values.Kind() != reflect.Slice {
			continue
		}

		// Build map from keys and values
		recordMap := make(map[string]interface{})
		for j := 0; j < keys.Len() && j < values.Len(); j++ {
			keyVal := keys.Index(j)
			valueVal := values.Index(j)

			if key, ok := keyVal.Interface().(string); ok {
				recordMap[key] = valueVal.Interface()
			}
		}

		// Convert record map to edge
		edge, err := eo.convertRecordToEdge(recordMap)
		if err != nil {
			log.Printf("Warning: failed to convert Neo4j record to edge: %v", err)
			continue
		}

		edges = append(edges, edge)
	}

	return edges
}

// bypassResolveEdges is a helper that simulates edge resolution without using LLM.
// It effectively maps each extracted edge to itself.
func bypassResolveEdges(ctx context.Context, edges []*types.Edge) ([]*types.Edge, []*types.Edge, error) {
	// Just return edges as resolved, with no invalidated edges
	return edges, []*types.Edge{}, nil
}
