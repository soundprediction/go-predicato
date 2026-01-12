package predicato

import (
	"context"
	"fmt"
	"time"

	"github.com/soundprediction/predicato/pkg/factstore"
	"github.com/soundprediction/predicato/pkg/prompts"
	"github.com/soundprediction/predicato/pkg/types"
	"github.com/soundprediction/predicato/pkg/utils/maintenance"
)

// ExtractToFacts extracts knowledge from an episode and saves it to the facts database.
func (c *Client) ExtractToFacts(ctx context.Context, episode types.Episode, options *AddEpisodeOptions) error {
	if c.factStore == nil {
		return fmt.Errorf("facts DB not configured")
	}

	if options == nil {
		options = &AddEpisodeOptions{}
	}

	// 1. Prepare and Chunk
	chunks, err := c.prepareAndValidateEpisode(&episode, options, options.MaxCharacters)
	if err != nil {
		return err
	}

	// 2. Get Context
	previousEpisodes, err := c.getPreviousEpisodesForContext(ctx, episode, options)
	if err != nil {
		return err
	}

	// 3. Create Structures
	chunkData, err := c.createChunkEpisodeStructures(ctx, episode, chunks, previousEpisodes, options)
	if err != nil {
		return err
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
		return err
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
				EntityType:  n.EntityType,
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
				return err
			}

			for _, e := range extracted {
				// Resolve Names for Facts DB
				// e.SourceNodeID and e.TargetNodeID are UUIDs from the extraction context (nodes + previous)

				var sourceName, targetName string

				// Lookup in current chunk nodes (most likely)
				for _, n := range nodes {
					if n.Uuid == e.SourceNodeID {
						sourceName = n.Name
					}
					if n.Uuid == e.TargetNodeID {
						targetName = n.Name
					}
				}

				// If not found, check flattenedNodes (all current chunks)
				if sourceName == "" || targetName == "" {
					for _, n := range flattenedNodes {
						if n.Uuid == e.SourceNodeID {
							sourceName = n.Name
						}
						if n.Uuid == e.TargetNodeID {
							targetName = n.Name
						}
					}
				}

				// If still not found, check previousEpisodes?
				if sourceName == "" || targetName == "" {
					for _, n := range previousEpisodes {
						if n.Uuid == e.SourceNodeID {
							sourceName = n.Name
						}
						if n.Uuid == e.TargetNodeID {
							targetName = n.Name
						}
					}
				}

				factsEdges = append(factsEdges, &factstore.ExtractedEdge{
					ID:             e.Uuid,
					SourceID:       episode.ID,
					SourceNodeName: sourceName,
					TargetNodeName: targetName,
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
		return err
	}

	return c.factStore.SaveExtractedKnowledge(ctx, episode.ID, factsNodes, factsEdges)
}

// PromoteToGraph reads extracted knowledge from facts DB and ingests it into the graph.
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

	// 3. Reconstruct Nodes aligned with Chunks
	extractedNodesByChunk := make([][]*types.Node, len(chunks))
	// Validation: check max chunk index
	maxIdx := 0
	for _, n := range extNodes {
		if n.ChunkIndex > maxIdx {
			maxIdx = n.ChunkIndex
		}
	}
	if maxIdx >= len(extractedNodesByChunk) {
		// Resize if stored chunks exceed recalculated chunks
		newSlice := make([][]*types.Node, maxIdx+1)
		copy(newSlice, extractedNodesByChunk)
		extractedNodesByChunk = newSlice
	}

	uuidToNode := make(map[string]*types.Node)

	for _, n := range extNodes {
		tn := &types.Node{
			Uuid:      n.ID,
			Name:      n.Name,
			Summary:   n.Description,
			Type:      types.EntityNodeType, // Default
			Embedding: n.Embedding,
			GroupID:   source.GroupID,
			// Important defaults to prevent validation errors
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
			ValidFrom: time.Now(),
		}
		// override type if possible, or assume simple entity
		if n.Type != "" {
			tn.Type = types.NodeType(n.Type)
		} else {
			tn.Type = types.EntityNodeType
		}

		if n.ChunkIndex < len(extractedNodesByChunk) {
			extractedNodesByChunk[n.ChunkIndex] = append(extractedNodesByChunk[n.ChunkIndex], tn)
		} else {
			// Fallback
			extractedNodesByChunk[0] = append(extractedNodesByChunk[0], tn)
		}
		uuidToNode[n.ID] = tn
	}

	// 4. Ingestion Pipeline (Dedupe)
	nodeOps := maintenance.NewNodeOperations(c.driver, c.nlpModels.NodeExtraction, c.embedder, prompts.NewLibrary())
	nodeOps.ReflexionNLP = c.nlpModels.NodeReflexion
	nodeOps.ResolutionNLP = c.nlpModels.NodeResolution
	nodeOps.AttributeNLP = c.nlpModels.NodeAttribute
	if options.UseYAML {
		nodeOps.UseYAML = true
	}
	nodeOps.SetLogger(c.logger)

	// Ignore dedupeResult. Using _ to avoid lint error.
	_, allResolvedNodes, err := c.deduplicateEntitiesAcrossChunks(ctx, source.ID, extractedNodesByChunk, chunkData.episodeTuples, options, nodeOps)
	if err != nil {
		return nil, err
	}

	// 5. Reconstruct Edges
	var allExtractedEdges []*types.Edge
	for _, e := range extEdges {
		var sUUID, tUUID string
		// Resolve using Names within current set of nodes

		for _, n := range uuidToNode {
			if n.Name == e.SourceNodeName {
				sUUID = n.Uuid
			}
			if n.Name == e.TargetNodeName {
				tUUID = n.Uuid
			}
		}

		if sUUID != "" && tUUID != "" {
			// Construct types.EntityEdge (alias Edge)
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
				Fact:     e.Description, // Summary/Fact usually same
				Strength: e.Weight,
				Type:     types.EntityEdgeType,
				SourceID: sUUID,
				TargetID: tUUID,
			}
			allExtractedEdges = append(allExtractedEdges, te)
		}
	}

	// 6. Resolve Edges
	edgeOps := maintenance.NewEdgeOperations(c.driver, c.nlProcessor, c.embedder, prompts.NewLibrary())
	edgeOps.ExtractionNLP = c.nlpModels.EdgeExtraction
	edgeOps.ResolutionNLP = c.nlpModels.EdgeResolution
	edgeOps.SkipResolution = options.SkipEdgeResolution
	edgeOps.UseYAML = options.UseYAML
	edgeOps.SetLogger(c.logger)

	resolvedEdges, _, err := c.resolveAndPersistRelationships(ctx, source.ID, allExtractedEdges, chunkData.mainEpisodeNode, allResolvedNodes, options, edgeOps)
	if err != nil {
		return nil, err
	}

	// 7. Episodic Edges (Entities <-> Episode)
	episodicEdges, err := c.buildEpisodicEdgesForEntities(ctx, allResolvedNodes, chunkData.mainEpisodeNode, time.Now(), edgeOps)
	if err != nil {
		return nil, err
	}

	// 8. Update Communities
	communities, communityEdges, err := c.UpdateCommunities(ctx, source.ID, source.GroupID)
	if err != nil {
		c.logger.Warn("Failed to update communities", "error", err)
	}

	return &types.AddEpisodeResults{
		Episode:        chunkData.mainEpisodeNode,
		EpisodicEdges:  episodicEdges,
		Nodes:          allResolvedNodes,
		Edges:          resolvedEdges,
		Communities:    communities,
		CommunityEdges: communityEdges,
	}, nil
}
