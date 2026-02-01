package predicato

import (
	"context"
	"fmt"
	"time"

	"github.com/soundprediction/predicato/pkg/factstore"
	"github.com/soundprediction/predicato/pkg/modeler"
	"github.com/soundprediction/predicato/pkg/prompts"
	"github.com/soundprediction/predicato/pkg/types"
	"github.com/soundprediction/predicato/pkg/utils/maintenance"
)

// ExtractToFacts extracts knowledge from an episode and saves it to the facts database.
// Returns ExtractionResults containing the raw extracted entities and relationships
// before graph modeling/resolution.
func (c *Client) ExtractToFacts(ctx context.Context, episode types.Episode, options *AddEpisodeOptions) (*types.ExtractionResults, error) {
	startTime := time.Now()

	if c.factStore == nil {
		return nil, fmt.Errorf("facts DB not configured")
	}

	if options == nil {
		options = &AddEpisodeOptions{}
	}

	// 1. Prepare and Chunk
	chunks, err := c.prepareAndValidateEpisode(&episode, options, options.MaxCharacters)
	if err != nil {
		return nil, err
	}

	// 2. Get Context
	previousEpisodes, err := c.getPreviousEpisodesForContext(ctx, episode, options)
	if err != nil {
		return nil, err
	}

	// 3. Create Structures
	chunkData, err := c.createChunkEpisodeStructures(ctx, episode, chunks, previousEpisodes, options)
	if err != nil {
		return nil, err
	}

	// 4. Extract Entities (Raw)
	nodeOps := maintenance.NewNodeOperations(c.driver, c.nlpModels.NodeExtraction, c.embedder, prompts.NewLibrary())
	nodeOps.ReflexionNLP = c.nlpModels.NodeReflexion
	nodeOps.ResolutionNLP = c.nlpModels.NodeResolution
	nodeOps.AttributeNLP = c.nlpModels.NodeAttribute
	if options.UseYAML {
		nodeOps.UseYAML = true
	}
	nodeOps.SetLogger(c.logger)

	extractedNodesByChunk, err := c.extractEntitiesFromAllChunks(ctx, episode.ID, chunkData.chunkEpisodeNodes, previousEpisodes, options, nodeOps)
	if err != nil {
		return nil, err
	}

	// 5. Prepare Facts Data (Nodes)
	var flattenedNodes []*types.Node
	var factsNodes []*factstore.ExtractedNode

	for chunkIdx, nodes := range extractedNodesByChunk {
		for _, n := range nodes {
			flattenedNodes = append(flattenedNodes, n)
			factsNodes = append(factsNodes, &factstore.ExtractedNode{
				ID:          n.Uuid,
				SourceID:    episode.ID,
				Name:        n.Name,
				Type:        string(n.Type),
				Description: n.Summary,
				Embedding:   n.Embedding,
				ChunkIndex:  chunkIdx,
			})
		}
	}

	// 6. Extract Edges (Raw) and Prepare Facts Data
	edgeOps := maintenance.NewEdgeOperations(c.driver, c.nlProcessor, c.embedder, prompts.NewLibrary())
	edgeOps.ExtractionNLP = c.nlpModels.EdgeExtraction
	edgeOps.ResolutionNLP = c.nlpModels.EdgeResolution
	edgeOps.SkipResolution = options.SkipEdgeResolution
	edgeOps.UseYAML = options.UseYAML
	edgeOps.SetLogger(c.logger)

	edgeTypeMap := make(map[string][][]string)
	if options.EdgeTypeMap != nil {
		for outerEntity, innerMap := range options.EdgeTypeMap {
			for innerEntity, relationships := range innerMap {
				for _, relation := range relationships {
					edgeTypeMap[relation.(string)] = append(edgeTypeMap[relation.(string)], []string{outerEntity, innerEntity})
				}
			}
		}
	}

	var factsEdges []*factstore.ExtractedEdge

	for chunkIdx, nodes := range extractedNodesByChunk {
		if len(nodes) > 0 {
			// Extract edges using the chunk's context
			extracted, err := edgeOps.ExtractEdges(ctx, chunkData.chunkEpisodeNodes[chunkIdx], nodes, previousEpisodes, edgeTypeMap, options.EdgeTypes, episode.GroupID)
			if err != nil {
				return nil, err
			}

			for _, e := range extracted {
				// Resolve Names for Facts DB
				// e.SourceNodeID and e.TargetNodeID are UUIDs from the extraction context (nodes + previous)

				var sourceName, sourceType, targetName, targetType string

				// Lookup in current chunk nodes (most likely)
				for _, n := range nodes {
					if n.Uuid == e.SourceNodeID {
						sourceName = n.Name
							sourceType = n.EntityType
					}
					if n.Uuid == e.TargetNodeID {
						targetName = n.Name
							targetType = n.EntityType
					}
				}

				// If not found, check flattenedNodes (all current chunks)
				if sourceName == "" || targetName == "" {
					for _, n := range flattenedNodes {
						if n.Uuid == e.SourceNodeID {
							sourceName = n.Name
							sourceType = n.EntityType
						}
						if n.Uuid == e.TargetNodeID {
							targetName = n.Name
							targetType = n.EntityType
						}
					}
				}

				// If still not found, check previousEpisodes?
				if sourceName == "" || targetName == "" {
					for _, n := range previousEpisodes {
						if n.Uuid == e.SourceNodeID {
							sourceName = n.Name
							sourceType = n.EntityType
						}
						if n.Uuid == e.TargetNodeID {
							targetName = n.Name
							targetType = n.EntityType
						}
					}
				}

				factsEdges = append(factsEdges, &factstore.ExtractedEdge{
					ID:             e.Uuid,
					SourceID:       episode.ID,
					SourceNodeName: sourceName,
					SourceNodeType: sourceType,
					TargetNodeName: targetName,
					TargetNodeType: targetType,
					Relation:       e.Name,
					Description:    e.Summary,  // Alias for Fact
					Weight:         e.Strength, // Use Strength
					ChunkIndex:     chunkIdx,
				})
			}
		}
	}

	// 7. Save to Facts
	source := &factstore.Source{
		ID:        episode.ID,
		Name:      episode.Name,
		Content:   episode.Content,
		GroupID:   episode.GroupID,
		Metadata:  episode.Metadata,
		CreatedAt: episode.CreatedAt,
	}
	if err := c.factStore.SaveSource(ctx, source); err != nil {
		return nil, err
	}

	if err := c.factStore.SaveExtractedKnowledge(ctx, episode.ID, factsNodes, factsEdges); err != nil {
		return nil, err
	}

	// 8. Return ExtractionResults
	return &types.ExtractionResults{
		SourceID:       episode.ID,
		ExtractedNodes: factsNodes,
		ExtractedEdges: factsEdges,
		ChunkCount:     len(chunks),
		ExtractionTime: time.Since(startTime),
	}, nil
}

// getOrCreateModeler returns the GraphModeler to use, checking options, config, then creating default.
func (c *Client) getOrCreateModeler(options *AddEpisodeOptions) (modeler.GraphModeler, error) {
	// 1. Check options
	if options != nil && options.GraphModeler != nil {
		return options.GraphModeler, nil
	}

	// 2. Check config
	if c.config != nil && c.config.DefaultGraphModeler != nil {
		return c.config.DefaultGraphModeler, nil
	}

	// 3. Create DefaultModeler
	nlpModels := &modeler.NlpModels{
		NodeExtraction: c.nlpModels.NodeExtraction,
		NodeReflexion:  c.nlpModels.NodeReflexion,
		NodeResolution: c.nlpModels.NodeResolution,
		NodeAttribute:  c.nlpModels.NodeAttribute,
		EdgeExtraction: c.nlpModels.EdgeExtraction,
		EdgeResolution: c.nlpModels.EdgeResolution,
		Summarization:  c.nlpModels.Summarization,
	}

	defaultModeler, err := modeler.NewDefaultModeler(&modeler.DefaultModelerOptions{
		Driver:    c.driver,
		NlpClient: c.nlProcessor,
		Embedder:  c.embedder,
		NlpModels: nlpModels,
		Logger:    c.logger,
		UseYAML:   options != nil && options.UseYAML,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create default modeler: %w", err)
	}

	return defaultModeler, nil
}

// handleModelerError handles errors from GraphModeler based on ModelerErrorHandling setting.
// Returns: (shouldContinue bool, fallbackModeler GraphModeler, error)
func (c *Client) handleModelerError(step string, err error, options *AddEpisodeOptions) (bool, modeler.GraphModeler, error) {
	handling := modeler.FailOnError
	if options != nil {
		handling = options.ModelerErrorHandling
	}

	modelerErr := modeler.NewModelerError(step, err)

	switch handling {
	case modeler.FailOnError:
		return false, nil, modelerErr

	case modeler.FallbackOnError:
		c.logger.Warn("GraphModeler failed, falling back to DefaultModeler",
			"step", step,
			"error", err)

		// Create a fresh DefaultModeler for fallback
		fallback, createErr := c.getOrCreateModeler(nil) // nil forces DefaultModeler
		if createErr != nil {
			return false, nil, fmt.Errorf("failed to create fallback modeler: %w", createErr)
		}
		return true, fallback, modelerErr.WithFallback()

	case modeler.SkipOnError:
		c.logger.Warn("GraphModeler failed, skipping step",
			"step", step,
			"error", err)
		return true, nil, modelerErr.WithSkipped()

	default:
		return false, nil, modelerErr
	}
}

// PromoteToGraph reads extracted knowledge from facts DB and ingests it into the graph.
// Uses the configured GraphModeler (from options, config, or DefaultModeler) for
// entity resolution, relationship resolution, and community detection.
func (c *Client) PromoteToGraph(ctx context.Context, sourceID string, options *AddEpisodeOptions) (*types.AddEpisodeResults, error) {
	if c.factStore == nil {
		return nil, fmt.Errorf("facts DB not configured")
	}

	if options == nil {
		options = &AddEpisodeOptions{}
	}

	// 1. Load from Facts
	source, err := c.factStore.GetSource(ctx, sourceID)
	if err != nil {
		return nil, err
	}

	extNodes, err := c.factStore.GetExtractedNodes(ctx, sourceID)
	if err != nil {
		return nil, err
	}

	extEdges, err := c.factStore.GetExtractedEdges(ctx, sourceID)
	if err != nil {
		return nil, err
	}

	// 2. Reconstruct Episode and Chunks to get Tuples/Structure
	episode := types.Episode{
		ID:        source.ID,
		Name:      source.Name,
		Content:   source.Content,
		GroupID:   source.GroupID,
		Metadata:  source.Metadata,
		CreatedAt: source.CreatedAt,
	}

	chunks, err := c.prepareAndValidateEpisode(&episode, options, options.MaxCharacters)
	if err != nil {
		return nil, err
	}

	previousEpisodes, err := c.getPreviousEpisodesForContext(ctx, episode, options)
	if err != nil {
		return nil, err
	}

	chunkData, err := c.createChunkEpisodeStructures(ctx, episode, chunks, previousEpisodes, options)
	if err != nil {
		return nil, err
	}

	// 3. Reconstruct Nodes from Facts
	uuidToNode := make(map[string]*types.Node)
	var allExtractedNodes []*types.Node

	for _, n := range extNodes {
		tn := &types.Node{
			Uuid:      n.ID,
			Name:      n.Name,
			Summary:   n.Description,
			Type:      types.EntityNodeType,
			Embedding: n.Embedding,
			GroupID:   source.GroupID,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
			ValidFrom: time.Now(),
		}
		if n.Type != "" {
			tn.Type = types.NodeType(n.Type)
		}
		uuidToNode[n.ID] = tn
		allExtractedNodes = append(allExtractedNodes, tn)
	}

	// 4. Reconstruct Edges from Facts
	var allExtractedEdges []*types.Edge
	for _, e := range extEdges {
		var sUUID, tUUID string
		for _, n := range uuidToNode {
			if n.Name == e.SourceNodeName {
				sUUID = n.Uuid
			}
			if n.Name == e.TargetNodeName {
				tUUID = n.Uuid
			}
		}

		if sUUID != "" && tUUID != "" {
			te := &types.Edge{
				BaseEdge: types.BaseEdge{
					Uuid:         e.ID,
					SourceNodeID: sUUID,
					TargetNodeID: tUUID,
					GroupID:      source.GroupID,
					CreatedAt:    time.Now(),
				},
				Name:     e.Relation,
				Summary:  e.Description,
				Fact:     e.Description,
				Strength: e.Weight,
				Type:     types.EntityEdgeType,
				SourceID: sUUID,
				TargetID: tUUID,
			}
			allExtractedEdges = append(allExtractedEdges, te)
		}
	}

	// 5. Get or create GraphModeler
	gm, err := c.getOrCreateModeler(options)
	if err != nil {
		return nil, err
	}

	c.logger.Info("Using GraphModeler for promotion",
		"source_id", sourceID,
		"modeler_type", fmt.Sprintf("%T", gm),
		"nodes", len(allExtractedNodes),
		"edges", len(allExtractedEdges))

	// 6. Resolve Entities using GraphModeler
	entityInput := &modeler.EntityResolutionInput{
		ExtractedNodes:   allExtractedNodes,
		Episode:          chunkData.mainEpisodeNode,
		PreviousEpisodes: previousEpisodes,
		EntityTypes:      options.EntityTypes,
		GroupID:          source.GroupID,
		Options: &modeler.EntityResolutionOptions{
			SkipResolution: options.SkipResolution,
			SkipReflexion:  options.SkipReflexion,
			SkipAttributes: options.SkipAttributes,
		},
	}

	entityOutput, err := gm.ResolveEntities(ctx, entityInput)
	if err != nil {
		shouldContinue, fallback, handledErr := c.handleModelerError("ResolveEntities", err, options)
		if !shouldContinue {
			return nil, handledErr
		}
		if fallback != nil {
			// Retry with fallback modeler
			entityOutput, err = fallback.ResolveEntities(ctx, entityInput)
			if err != nil {
				return nil, fmt.Errorf("fallback ResolveEntities failed: %w", err)
			}
		} else {
			// Skipped - use identity mapping
			entityOutput = &modeler.EntityResolutionOutput{
				ResolvedNodes: allExtractedNodes,
				UUIDMap:       make(map[string]string),
				NewCount:      len(allExtractedNodes),
			}
			for _, n := range allExtractedNodes {
				entityOutput.UUIDMap[n.Uuid] = n.Uuid
			}
		}
	}

	c.logger.Info("Entity resolution complete",
		"resolved", len(entityOutput.ResolvedNodes),
		"merged", entityOutput.MergedCount,
		"new", entityOutput.NewCount)

	// 7. Resolve Relationships using GraphModeler
	relInput := &modeler.RelationshipResolutionInput{
		ExtractedEdges: allExtractedEdges,
		ResolvedNodes:  entityOutput.ResolvedNodes,
		UUIDMap:        entityOutput.UUIDMap,
		Episode:        chunkData.mainEpisodeNode,
		GroupID:        source.GroupID,
		EdgeTypes:      options.EdgeTypes,
		Options: &modeler.RelationshipResolutionOptions{
			SkipEdgeResolution: options.SkipEdgeResolution,
		},
	}

	relOutput, err := gm.ResolveRelationships(ctx, relInput)
	if err != nil {
		shouldContinue, fallback, handledErr := c.handleModelerError("ResolveRelationships", err, options)
		if !shouldContinue {
			return nil, handledErr
		}
		if fallback != nil {
			relOutput, err = fallback.ResolveRelationships(ctx, relInput)
			if err != nil {
				return nil, fmt.Errorf("fallback ResolveRelationships failed: %w", err)
			}
		} else {
			// Skipped - use edges as-is with identity mapping
			relOutput = &modeler.RelationshipResolutionOutput{
				ResolvedEdges: allExtractedEdges,
				EpisodicEdges: []*types.Edge{},
				NewCount:      len(allExtractedEdges),
			}
		}
	}

	c.logger.Info("Relationship resolution complete",
		"resolved_edges", len(relOutput.ResolvedEdges),
		"episodic_edges", len(relOutput.EpisodicEdges),
		"new", relOutput.NewCount)

	// 8. Build Communities using GraphModeler
	commInput := &modeler.CommunityInput{
		Nodes:     entityOutput.ResolvedNodes,
		Edges:     relOutput.ResolvedEdges,
		GroupID:   source.GroupID,
		EpisodeID: source.ID,
	}

	var communities []*types.Node
	var communityEdges []*types.Edge

	commOutput, err := gm.BuildCommunities(ctx, commInput)
	if err != nil {
		shouldContinue, fallback, handledErr := c.handleModelerError("BuildCommunities", err, options)
		if !shouldContinue {
			return nil, handledErr
		}
		if fallback != nil {
			commOutput, err = fallback.BuildCommunities(ctx, commInput)
			if err != nil {
				c.logger.Warn("Fallback BuildCommunities failed", "error", err)
				// Community detection is optional, continue without it
			}
		}
		// If skipped or fallback failed, commOutput stays nil
		_ = handledErr // Acknowledged but not returned
	}

	if commOutput != nil {
		communities = commOutput.Communities
		communityEdges = commOutput.CommunityEdges
		c.logger.Info("Community detection complete",
			"communities", len(communities),
			"community_edges", len(communityEdges))
	}

	return &types.AddEpisodeResults{
		Episode:        chunkData.mainEpisodeNode,
		EpisodicEdges:  relOutput.EpisodicEdges,
		Nodes:          entityOutput.ResolvedNodes,
		Edges:          relOutput.ResolvedEdges,
		Communities:    communities,
		CommunityEdges: communityEdges,
	}, nil
}

// ValidateModeler tests a GraphModeler implementation with sample data to verify
// it works correctly before using it in production.
func (c *Client) ValidateModeler(ctx context.Context, gm modeler.GraphModeler) (*modeler.ModelerValidationResult, error) {
	opts := &modeler.ValidateModelerOptions{
		GroupID: c.config.GroupID,
	}
	return modeler.ValidateModeler(ctx, gm, opts)
}
