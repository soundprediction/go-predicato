package maintenance

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"strings"
	"time"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/embedder"
	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/prompts"
	"github.com/soundprediction/predicato/pkg/types"
	"github.com/soundprediction/predicato/pkg/utils"
)

const (
	// MaxAttributeExtractionBatchSize is the maximum number of nodes to process in a single LLM call
	MaxAttributeExtractionBatchSize = 24
)

// NodeOperations provides node-related maintenance operations
type NodeOperations struct {
	driver      driver.GraphDriver
	nlProcessor nlp.Client
	embedder    embedder.Client
	prompts     prompts.Library
	logger      *slog.Logger

	// Specialized NLP clients for different steps
	ExtractionNLP nlp.Client
	ReflexionNLP  nlp.Client
	ResolutionNLP nlp.Client
	AttributeNLP  nlp.Client

	// Skip flags
	SkipReflexion  bool
	SkipResolution bool
	SkipAttributes bool

	// Format flags
	UseYAML bool
}

// NewNodeOperations creates a new NodeOperations instance
func NewNodeOperations(driver driver.GraphDriver, nlProcessor nlp.Client, embedder embedder.Client, prompts prompts.Library) *NodeOperations {
	return &NodeOperations{
		driver:      driver,
		nlProcessor: nlProcessor,
		embedder:    embedder,
		prompts:     prompts,
		logger:      slog.Default(), // Use default logger, can be overridden
	}
}

// SetLogger sets a custom logger for the NodeOperations
func (no *NodeOperations) SetLogger(logger *slog.Logger) {
	no.logger = logger
}

// Helper methods to get the appropriate LLM client with fallback to default
func (no *NodeOperations) getExtractionNLP() nlp.Client {
	if no.ExtractionNLP != nil {
		return no.ExtractionNLP
	}
	return no.nlProcessor
}

func (no *NodeOperations) getReflexionNLP() nlp.Client {
	if no.ReflexionNLP != nil {
		return no.ReflexionNLP
	}
	return no.nlProcessor
}

func (no *NodeOperations) getResolutionNLP() nlp.Client {
	if no.ResolutionNLP != nil {
		return no.ResolutionNLP
	}
	return no.nlProcessor
}

func (no *NodeOperations) getAttributeNLP() nlp.Client {
	if no.AttributeNLP != nil {
		return no.AttributeNLP
	}
	return no.nlProcessor
}

// ExtractNodes extracts entity nodes from episode content using LLM
func (no *NodeOperations) ExtractNodes(ctx context.Context, episode *types.Node, previousEpisodes []*types.Node, entityTypes map[string]interface{}, excludedEntityTypes []string) ([]*types.Node, error) {
	start := time.Now()

	// Prepare entity types context
	entityTypesContext := []map[string]interface{}{
		{
			"entity_type_id":          0,
			"entity_type_name":        "Entity",
			"entity_type_description": "Default classification. Use this entity type if the entity is not one of the other listed types.",
		},
	}

	if entityTypes != nil {
		id := 1
		for typeName := range entityTypes {
			entityTypesContext = append(entityTypesContext, map[string]interface{}{
				"entity_type_id":          id,
				"entity_type_name":        typeName,
				"entity_type_description": fmt.Sprintf("custom type: %s", typeName),
			})
			id++
		}
	}

	// Prepare previous episodes content
	previousEpisodeContents := make([]string, len(previousEpisodes))
	for i, ep := range previousEpisodes {
		previousEpisodeContents[i] = ep.Summary
	}

	// Prepare context for LLM
	// Note: entity_types is passed as a slice for TSV formatting in prompts
	promptContext := map[string]interface{}{
		"episode_content":    episode.Content,
		"episode_timestamp":  episode.ValidFrom.Format(time.RFC3339),
		"previous_episodes":  previousEpisodeContents,
		"custom_prompt":      "",
		"entity_types":       entityTypesContext,
		"source_description": string(episode.EpisodeType),
		"ensure_ascii":       true,
		"logger":             no.logger,
		"use_yaml":           no.UseYAML,
	}

	// Extract entities with reflexion
	entitiesMissed := true
	reflexionIterations := 0
	maxReflexionIterations := utils.GetMaxReflexionIterations()

	var extractedEntities prompts.ExtractedEntities

	for entitiesMissed && reflexionIterations <= maxReflexionIterations {
		// Choose the appropriate extraction method based on episode source
		var messages []types.Message
		var err error

		switch strings.ToLower(string(episode.EpisodeType)) {
		case "message":
			messages, err = no.prompts.ExtractNodes().ExtractMessage().Call(promptContext)
		case "text":
			messages, err = no.prompts.ExtractNodes().ExtractText().Call(promptContext)
		case "json":
			messages, err = no.prompts.ExtractNodes().ExtractJSON().Call(promptContext)
		default:
			messages, err = no.prompts.ExtractNodes().ExtractText().Call(promptContext)
		}

		if err != nil {
			return nil, fmt.Errorf("failed to create extraction prompt: %w", err)
		}

		var extractedEntitySlice []prompts.ExtractedEntity
		var badResp *types.BadLlmCsvResponse

		if no.UseYAML {
			// Create YAML parser function for ExtractedEntity
			yamlParser := func(yamlContent string) ([]*prompts.ExtractedEntity, error) {
				return utils.UnmarshalYAML[prompts.ExtractedEntity](yamlContent)
			}

			// Use GenerateYAMLResponse
			extractedEntitySlice, badResp, err = nlp.GenerateYAMLResponse[prompts.ExtractedEntity](
				ctx,
				no.getExtractionNLP(),
				no.logger,
				messages,
				yamlParser,
				3, // maxRetries
			)
		} else {
			// Create CSV parser function for ExtractedEntity
			csvParser := func(csvContent string) ([]*prompts.ExtractedEntity, error) {
				return utils.UnmarshalCSV[prompts.ExtractedEntity](csvContent, '\t')
			}

			// Use GenerateCSVResponse for robust CSV parsing with retries
			extractedEntitySlice, badResp, err = nlp.GenerateCSVResponse[prompts.ExtractedEntity](
				ctx,
				no.getExtractionNLP(),
				no.logger,
				messages,
				csvParser,
				3, // maxRetries
			)
		}

		if err != nil {
			// Log detailed error information
			if badResp != nil {
				no.logger.Error("Failed to extract entities from CSV",
					"error", badResp.Error,
					"response_length", len(badResp.Response),
					"num_messages", len(badResp.Messages))
				if badResp.Response != "" {
					fmt.Printf("\nFailed LLM response:\n%v\n\n", badResp.Response)
				}
			}
			return nil, fmt.Errorf("failed to extract entities from csv: %w", err)
		}

		// Convert to ExtractedEntities struct
		extractedEntities.ExtractedEntities = extractedEntitySlice

		reflexionIterations++
		if !no.SkipReflexion && reflexionIterations < maxReflexionIterations {
			// Run reflexion to check for missed entities
			missedEntities, err := no.extractNodesReflexion(ctx, episode, previousEpisodes, extractedEntities)
			if err != nil {
				log.Printf("Warning: reflexion failed: %v", err)
				break
			}

			entitiesMissed = len(missedEntities) > 0
			if entitiesMissed {
				customPrompt := "Make sure that the following entities are extracted:"
				for _, entity := range missedEntities {
					customPrompt += fmt.Sprintf("\n%s,", entity)
				}
				promptContext["custom_prompt"] = customPrompt
			}
		} else {
			entitiesMissed = false
		}
	}

	// Filter out empty entity names
	var filteredEntities []prompts.ExtractedEntity
	for _, entity := range extractedEntities.ExtractedEntities {
		if strings.TrimSpace(entity.Name) != "" {
			filteredEntities = append(filteredEntities, entity)
		}
	}

	log.Printf("Extracted %d entities in %v", len(filteredEntities), time.Since(start))

	// Convert to Node objects
	var extractedNodes []*types.Node
	for _, extractedEntity := range filteredEntities {
		// Determine entity type
		var entityTypeName string
		if extractedEntity.EntityTypeID >= 0 && extractedEntity.EntityTypeID < len(entityTypesContext) {
			entityTypeName = entityTypesContext[extractedEntity.EntityTypeID]["entity_type_name"].(string)
		} else {
			entityTypeName = "Entity"
		}

		// Check if this entity type should be excluded
		if len(excludedEntityTypes) > 0 {
			excluded := false
			for _, excludedType := range excludedEntityTypes {
				if entityTypeName == excludedType {
					excluded = true
					break
				}
			}
			if excluded {
				log.Printf("Excluding entity %s of type %s", extractedEntity.Name, entityTypeName)
				continue
			}
		}

		node := &types.Node{
			Uuid:       utils.GenerateUUID(),
			Type:       types.EntityNodeType,
			GroupID:    episode.GroupID,
			Name:       extractedEntity.Name,
			Summary:    extractedEntity.Name,
			CreatedAt:  time.Now().UTC(),
			UpdatedAt:  time.Now().UTC(),
			ValidFrom:  episode.ValidFrom,
			EntityType: entityTypeName,
			Metadata:   make(map[string]interface{}),
		}

		extractedNodes = append(extractedNodes, node)
		// log.Printf("Created entity node: %s of type: %s (UUID: %s)", node.Name, node.EntityType, node.ID)
	}

	return extractedNodes, nil
}

// extractNodesReflexion performs reflexion to identify missed entities
func (no *NodeOperations) extractNodesReflexion(ctx context.Context, episode *types.Node, previousEpisodes []*types.Node, extractedEntities prompts.ExtractedEntities) ([]string, error) {
	// Get entity names
	var entityNames []string
	for _, entity := range extractedEntities.ExtractedEntities {
		entityNames = append(entityNames, entity.Name)
	}

	// Prepare previous episodes content
	previousEpisodeContents := make([]string, len(previousEpisodes))
	for i, ep := range previousEpisodes {
		previousEpisodeContents[i] = ep.Summary
	}

	// Prepare context for reflexion
	promptContext := map[string]interface{}{
		"episode_content":    episode.Summary,
		"previous_episodes":  previousEpisodeContents,
		"extracted_entities": entityNames,
		"ensure_ascii":       true,
		"logger":             no.logger,
	}

	messages, err := no.prompts.ExtractNodes().Reflexion().Call(promptContext)
	if err != nil {
		return nil, fmt.Errorf("failed to create reflexion prompt: %w", err)
	}

	// Create CSV parser function for MissedEntitiesTSV
	csvParser := func(csvContent string) ([]*prompts.MissedEntitiesTSV, error) {
		return utils.UnmarshalCSV[prompts.MissedEntitiesTSV](csvContent, '\t')
	}

	// Use GenerateCSVResponse for robust CSV parsing with retries
	missedEntitiesSlice, badResp, err := nlp.GenerateCSVResponse[prompts.MissedEntitiesTSV](
		ctx, no.getReflexionNLP(), no.logger, messages, csvParser, 3,
	)
	if err != nil {
		if badResp != nil {
			no.logger.Error("Failed to parse reflexion response",
				"error", err,
				"last_response", badResp.Response,
				"conversation", badResp.Messages,
			)
		}
		return nil, fmt.Errorf("failed to run reflexion: %w", err)
	}

	// Convert slice of MissedEntitiesTSV to slice of entity names
	missedEntityNames := make([]string, 0, len(missedEntitiesSlice))
	for _, entity := range missedEntitiesSlice {
		missedEntityNames = append(missedEntityNames, entity.EntityName)
	}

	return missedEntityNames, nil
}

// ResolveExtractedNodes resolves newly extracted nodes against existing ones in the graph
func (no *NodeOperations) ResolveExtractedNodes(ctx context.Context, extractedNodes []*types.Node, episode *types.Node, previousEpisodes []*types.Node, entityTypes map[string]interface{}) ([]*types.Node, map[string]string, []NodePair, error) {
	if len(extractedNodes) == 0 {
		return []*types.Node{}, make(map[string]string), []NodePair{}, nil
	}

	// Search for existing nodes that might be duplicates
	var candidateNodes []*types.Node
	searchResults := make(map[string][]*types.Node)

	for _, node := range extractedNodes {
		// Search for nodes with similar names
		options := &driver.SearchOptions{
			Limit:     50,
			NodeTypes: []types.NodeType{types.EntityNodeType},
		}

		nodes, err := no.driver.SearchNodes(ctx, node.Name, node.GroupID, options)
		if err != nil {
			log.Printf("Warning: failed to search for similar nodes: %v", err)
			nodes = []*types.Node{}
		}

		searchResults[node.Uuid] = nodes
		candidateNodes = append(candidateNodes, nodes...)
	}

	// Remove duplicates from candidate nodes
	candidateMap := make(map[string]*types.Node)
	for _, node := range candidateNodes {
		candidateMap[node.Uuid] = node
	}

	var existingNodes []*types.Node
	for _, node := range candidateMap {
		existingNodes = append(existingNodes, node)
	}

	// Build entity type description lookup map
	entityTypeDescriptions := make(map[string]string)
	entityTypeDescriptions["Entity"] = "Default classification. Use this entity type if the entity is not one of the other listed types."

	for typeName := range entityTypes {
		entityTypeDescriptions[typeName] = fmt.Sprintf("custom type: %s", typeName)
	}

	// Prepare context for LLM deduplication
	extractedNodesContext := make([]map[string]interface{}, len(extractedNodes))
	for i, node := range extractedNodes {
		// Look up the description for this entity type
		description := entityTypeDescriptions[node.EntityType]
		if description == "" {
			description = "Entity description"
		}

		extractedNodesContext[i] = map[string]interface{}{
			"id":                      i,
			"name":                    node.Name,
			"entity_type":             []string{"Entity", node.EntityType},
			"entity_type_description": description,
		}
	}

	existingNodesContext := make([]map[string]interface{}, len(existingNodes))
	for i, node := range existingNodes {
		existingNodesContext[i] = map[string]interface{}{
			"idx":          i,
			"name":         node.Name,
			"entity_types": []string{"Entity", node.EntityType},
			"summary":      node.Summary,
		}
		// Add metadata as attributes
		for k, v := range node.Metadata {
			existingNodesContext[i][k] = v
		}
	}

	// Prepare previous episodes content
	previousEpisodeContents := make([]string, len(previousEpisodes))
	for i, ep := range previousEpisodes {
		previousEpisodeContents[i] = ep.Summary
	}

	promptContext := map[string]interface{}{
		"extracted_nodes":   extractedNodesContext,
		"existing_nodes":    existingNodesContext,
		"episode_content":   episode.Content,
		"previous_episodes": previousEpisodeContents,
		"ensure_ascii":      true,
		"logger":            no.logger,
	}

	// Use LLM to resolve duplicates
	if no.SkipResolution {
		no.logger.Info("Skipping node resolution (deduplication) as requested")
		return bypassResolveExtractedNodes(ctx, extractedNodes)
	}

	messages, err := no.prompts.DedupeNodes().Nodes().Call(promptContext)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to create dedupe prompt: %w", err)
	}

	// Create CSV parser function for NodeDuplicate
	csvParser := func(csvContent string) ([]*prompts.NodeDuplicate, error) {
		return utils.UnmarshalCSV[prompts.NodeDuplicate](csvContent, '\t')
	}

	// Use GenerateCSVResponse for robust CSV parsing with retries
	nodeDuplicateSlice, badResp, err := nlp.GenerateCSVResponse[prompts.NodeDuplicate](
		ctx,
		no.getResolutionNLP(),
		no.logger,
		messages,
		csvParser,
		0, // maxRetries (use default of 8)
	)

	if err != nil {
		// Log detailed error information
		if badResp != nil {
			no.logger.Error("Failed to deduplicate nodes from CSV",
				"error", badResp.Error,
				"response_length", len(badResp.Response),
				"num_messages", len(badResp.Messages))
			if badResp.Response != "" {
				no.logger.Debug("Failed LLM deduplication response", "response", badResp.Response)
			}
		}
		no.logger.Warn("Skipping node deduplication due to error", "error", err)
		return bypassResolveExtractedNodes(ctx, extractedNodes)
	}

	// Convert to NodeResolutions struct
	var nodeResolutions prompts.NodeResolutions
	nodeResolutions.EntityResolutions = nodeDuplicateSlice

	// Process the resolutions
	var resolvedNodes []*types.Node
	uuidMap := make(map[string]string)
	var nodeDuplicates []NodePair

	for _, resolution := range nodeResolutions.EntityResolutions {
		if resolution.ID < 0 || resolution.ID >= len(extractedNodes) {
			continue
		}

		extractedNode := extractedNodes[resolution.ID]
		var resolvedNode *types.Node

		// Check if it's a duplicate of an existing node
		if resolution.DuplicateIdx >= 0 && resolution.DuplicateIdx < len(existingNodes) {
			resolvedNode = existingNodes[resolution.DuplicateIdx]
		} else {
			resolvedNode = extractedNode
		}

		resolvedNodes = append(resolvedNodes, resolvedNode)
		uuidMap[extractedNode.Uuid] = resolvedNode.Uuid

		// Track duplicates for edge creation
		for _, duplicateIdx := range resolution.Duplicates {
			if duplicateIdx >= 0 && duplicateIdx < len(existingNodes) {
				existingNode := existingNodes[duplicateIdx]
				nodeDuplicates = append(nodeDuplicates, NodePair{
					Source: extractedNode,
					Target: existingNode,
				})
			}
		}
	}

	log.Printf("Resolved %d nodes, found %d duplicates", len(resolvedNodes), len(nodeDuplicates))

	// Filter duplicates using edge operations to remove those that already have IS_DUPLICATE_OF edges
	edgeOps := NewEdgeOperations(no.driver, no.nlProcessor, no.embedder, no.prompts)
	edgeOps.ResolutionNLP = no.getResolutionNLP()
	filteredDuplicates, err := edgeOps.FilterExistingDuplicateOfEdges(ctx, nodeDuplicates)
	if err != nil {
		log.Printf("Warning: failed to filter existing duplicate edges: %v", err)
		filteredDuplicates = nodeDuplicates
	}

	return resolvedNodes, uuidMap, filteredDuplicates, nil
}

func bypassResolveExtractedNodes(ctx context.Context, nodes []*types.Node) ([]*types.Node, map[string]string, []NodePair, error) {
	uuidMap := make(map[string]string)
	for _, node := range nodes {
		uuidMap[node.Uuid] = node.Uuid
	}
	return nodes, uuidMap, []NodePair{}, nil
}

// ExtractAttributesFromNodes extracts and updates attributes for nodes using LLM in batches
func (no *NodeOperations) ExtractAttributesFromNodes(ctx context.Context, nodes []*types.Node, episode *types.Node, previousEpisodes []*types.Node, entityTypes map[string]interface{}) ([]*types.Node, error) {
	if len(nodes) == 0 {
		return nodes, nil
	}

	log.Printf("Extracting attributes for %d entities in batches of %d", len(nodes), MaxAttributeExtractionBatchSize)

	// Prepare previous episodes content (shared across all batches)
	previousEpisodeContents := make([]string, len(previousEpisodes))
	for i, ep := range previousEpisodes {
		previousEpisodeContents[i] = ep.Summary
	}

	// Map to store all extracted attributes by original node index
	allExtractedMap := make(map[int]*prompts.ExtractedNodeAttributes)

	// Process nodes in batches
	for batchStart := 0; batchStart < len(nodes); batchStart += MaxAttributeExtractionBatchSize {
		batchEnd := batchStart + MaxAttributeExtractionBatchSize
		if batchEnd > len(nodes) {
			batchEnd = len(nodes)
		}

		batchNodes := nodes[batchStart:batchEnd]
		log.Printf("Processing batch %d-%d of %d nodes", batchStart, batchEnd, len(nodes))

		// Prepare nodes context for this batch
		nodesContext := make([]map[string]interface{}, len(batchNodes))
		for i, node := range batchNodes {
			nodesContext[i] = map[string]interface{}{
				"node_id":      i, // Local batch index
				"name":         node.Name,
				"summary":      node.Summary,
				"entity_types": []string{"Entity", node.EntityType},
				"attributes":   node.Metadata,
			}
		}

		// Prepare context for batch LLM call
		promptContext := map[string]interface{}{
			"nodes":             nodesContext,
			"episode_content":   episode.Content,
			"previous_episodes": previousEpisodeContents,
			"ensure_ascii":      true,
			"logger":            no.logger,
			"use_yaml":          no.UseYAML,
		}

		var extractedAttributesSlice []prompts.ExtractedNodeAttributes
		var badResp *types.BadLlmCsvResponse
		var err error

		if no.SkipAttributes {
			no.logger.Info("Skipping attribute extraction as requested")
			// Skip LLM call, simulated empty result
			extractedAttributesSlice = []prompts.ExtractedNodeAttributes{}
		} else {
			// Call batch extraction prompt
			messages, callErr := no.prompts.ExtractNodes().ExtractAttributesBatch().Call(promptContext)
			if callErr != nil {
				return nil, fmt.Errorf("failed to create batch extraction prompt: %w", callErr)
			}

			if no.UseYAML {
				// Create YAML parser function for ExtractedNodeAttributes
				yamlParser := func(yamlContent string) ([]*prompts.ExtractedNodeAttributes, error) {
					return utils.UnmarshalYAML[prompts.ExtractedNodeAttributes](yamlContent)
				}

				// Use GenerateYAMLResponse
				extractedAttributesSlice, badResp, err = nlp.GenerateYAMLResponse[prompts.ExtractedNodeAttributes](
					ctx,
					no.getAttributeNLP(),
					no.logger,
					messages,
					yamlParser,
					3, // maxRetries
				)
			} else {
				// Create CSV parser function for ExtractedNodeAttributes
				csvParser := func(csvContent string) ([]*prompts.ExtractedNodeAttributes, error) {
					return utils.UnmarshalCSV[prompts.ExtractedNodeAttributes](csvContent, '\t')
				}

				// Use GenerateCSVResponse for robust CSV parsing with retries
				extractedAttributesSlice, badResp, err = nlp.GenerateCSVResponse[prompts.ExtractedNodeAttributes](
					ctx,
					no.getAttributeNLP(),
					no.logger,
					messages,
					csvParser,
					3, // maxRetries
				)
			}
		}

		if err != nil {
			// Log detailed error information
			if badResp != nil {
				no.logger.Error("Failed to extract batch attributes from CSV",
					"error", badResp.Error,
					"response_length", len(badResp.Response),
					"num_messages", len(badResp.Messages))
				if badResp.Response != "" {
					fmt.Printf("\nFailed LLM response:\n%v\n\n", badResp.Response)
				}
			}
			return nil, fmt.Errorf("failed to parse batch extraction TSV: %w", err)
		}

		// Store extracted attributes with global node index
		for _, extracted := range extractedAttributesSlice {
			globalIndex := batchStart + extracted.NodeID
			allExtractedMap[globalIndex] = &extracted
		}
	}

	// Update all nodes with extracted summaries
	var updatedNodes []*types.Node
	for i, node := range nodes {
		updatedNode := *node // Copy the node
		updatedNode.UpdatedAt = time.Now().UTC()

		if extracted, ok := allExtractedMap[i]; ok {
			updatedNode.Summary = extracted.Summary
		} else {
			log.Printf("Warning: no extraction result for node %d (%s), keeping original", i, node.Name)
		}

		updatedNodes = append(updatedNodes, &updatedNode)
	}

	// Create embeddings for all updated nodes
	for _, node := range updatedNodes {
		if err := no.createNodeEmbedding(ctx, node); err != nil {
			log.Printf("Warning: failed to create embedding for node %s: %v", node.Name, err)
		}
	}

	log.Printf("Successfully extracted attributes for %d entities", len(updatedNodes))
	return updatedNodes, nil
}

// createNodeEmbedding creates an embedding for a node based on its name and summary
func (no *NodeOperations) createNodeEmbedding(ctx context.Context, node *types.Node) error {
	// Create text for embedding from name and summary
	if node == nil {
		return nil
	}
	text := node.Name
	if node.Summary != "" {
		text += " " + node.Summary
	}

	embedding, err := no.embedder.EmbedSingle(ctx, text)
	if err != nil {
		return fmt.Errorf("failed to create embedding: %w", err)
	}

	node.Embedding = embedding

	nameEmbedding, _ := no.embedder.EmbedSingle(ctx, node.Name)
	node.NameEmbedding = nameEmbedding
	return nil
}
