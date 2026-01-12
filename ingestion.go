package predicato

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"

	jsonrepair "github.com/kaptinlin/jsonrepair"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/prompts"
	"github.com/soundprediction/predicato/pkg/search"
	"github.com/soundprediction/predicato/pkg/types"
	"github.com/soundprediction/predicato/pkg/utils"
	"github.com/soundprediction/predicato/pkg/utils/maintenance"
)

// retrieveAndValidateEpisode retrieves an existing episode and validates it.
func (c *Client) retrieveAndValidateEpisode(ctx context.Context, episodeID string, groupID string) (*types.Node, error) {
	existingEpisode, err := c.driver.GetNode(ctx, episodeID, groupID)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve episode %s: %w", episodeID, err)
	}
	if existingEpisode == nil {
		return nil, fmt.Errorf("episode %s not found", episodeID)
	}
	if existingEpisode.Type != types.EpisodicNodeType {
		return nil, fmt.Errorf("node %s is not an episode (type: %s)", episodeID, existingEpisode.Type)
	}
	return existingEpisode, nil
}

// generateID generates a unique ID for nodes and edges.
func generateID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

// chunkText splits text into chunks of approximately maxChars size,
// preserving paragraph boundaries when possible. It prioritizes keeping
// complete paragraphs together and only splits within paragraphs when necessary.
func chunkText(text string, maxChars int) []string {
	if len(text) <= maxChars {
		return []string{text}
	}

	// Split text into paragraphs first (preserve paragraph structure)
	paragraphs := strings.Split(text, "\n\n")

	var chunks []string
	var currentChunk strings.Builder
	currentLen := 0

	for i, para := range paragraphs {
		paraLen := len(para)

		// If this single paragraph is longer than maxChars, we need to split it
		if paraLen > maxChars {
			// Flush current chunk if it has content
			if currentChunk.Len() > 0 {
				chunks = append(chunks, strings.TrimSpace(currentChunk.String()))
				currentChunk.Reset()
				currentLen = 0
			}

			// Split the large paragraph into smaller chunks
			subChunks := chunkParagraph(para, maxChars)
			chunks = append(chunks, subChunks...)
			continue
		}

		// Will adding this paragraph exceed maxChars?
		separator := ""
		if currentChunk.Len() > 0 {
			separator = "\n\n"
		}
		newLen := currentLen + len(separator) + paraLen

		if newLen > maxChars && currentChunk.Len() > 0 {
			// Adding this paragraph would exceed limit, flush current chunk
			chunks = append(chunks, strings.TrimSpace(currentChunk.String()))
			currentChunk.Reset()
			currentChunk.WriteString(para)
			currentLen = paraLen
		} else {
			// Add paragraph to current chunk
			if currentChunk.Len() > 0 {
				currentChunk.WriteString("\n\n")
			}
			currentChunk.WriteString(para)
			currentLen = newLen
		}

		// If this is the last paragraph, flush the chunk
		if i == len(paragraphs)-1 && currentChunk.Len() > 0 {
			chunks = append(chunks, strings.TrimSpace(currentChunk.String()))
		}
	}

	return chunks
}

// chunkParagraph splits a single paragraph that's too large into smaller chunks,
// breaking at sentence or word boundaries.
func chunkParagraph(para string, maxChars int) []string {
	var chunks []string
	remaining := para

	for len(remaining) > 0 {
		if len(remaining) <= maxChars {
			chunks = append(chunks, strings.TrimSpace(remaining))
			break
		}

		// Try to find a good break point within maxChars
		chunkEnd := maxChars
		var breakPoint int

		// Minimum chunk size to avoid tiny fragments (at least 1/3 of maxChars)
		minChunkSize := maxChars / 3

		// Try to break at a sentence boundary first
		if idx := strings.LastIndex(remaining[:chunkEnd], ". "); idx > minChunkSize {
			breakPoint = idx + 2
		} else if idx := strings.LastIndex(remaining[:chunkEnd], "! "); idx > minChunkSize {
			breakPoint = idx + 2
		} else if idx := strings.LastIndex(remaining[:chunkEnd], "? "); idx > minChunkSize {
			breakPoint = idx + 2
		} else if idx := strings.LastIndex(remaining[:chunkEnd], "\n"); idx > minChunkSize {
			// Try to break at a newline
			breakPoint = idx + 1
		} else if idx := strings.LastIndex(remaining[:chunkEnd], " "); idx > minChunkSize {
			// Try to break at a word boundary
			breakPoint = idx + 1
		} else {
			// No good break point found, just split at maxChars
			breakPoint = maxChars
		}

		chunks = append(chunks, strings.TrimSpace(remaining[:breakPoint]))
		remaining = remaining[breakPoint:]
	}

	return chunks
}

// Add processes episodes and adds them to the knowledge graph.
func (c *Client) Add(ctx context.Context, episodes []types.Episode, options *AddEpisodeOptions) (*types.AddBulkEpisodeResults, error) {
	// Check if this batch is completely empty
	if len(episodes) == 0 {
		return &types.AddBulkEpisodeResults{}, nil
	}

	// Filter out episodes that already exist
	var newEpisodes []types.Episode
	var skippedCount int

	for _, episode := range episodes {
		existingNode, err := c.driver.GetNode(ctx, episode.ID, c.config.GroupID)
		if err == nil && existingNode != nil {
			// Episode already exists, skip it
			skippedCount++
			c.logger.Debug("Skipping existing episode", "episode_id", episode.ID)
			continue
		}
		newEpisodes = append(newEpisodes, episode)
	}

	// Use the filtered list for processing
	episodes = newEpisodes

	// Print initial database statistics
	if stats, err := c.GetStats(ctx); err == nil {
		episodesInDB := int64(0)
		if stats.NodesByType != nil {
			episodesInDB = stats.NodesByType["Episodic"]
		}
		c.logger.Info("Initial database state",
			"node_count", stats.NodeCount,
			"edge_count", stats.EdgeCount,
			"episodes_in_db", episodesInDB,
			"communities", stats.CommunityCount,
			"episodes_to_add", len(episodes),
			"episodes_skipped", skippedCount)
	} else {
		c.logger.Warn("Failed to retrieve initial database stats", "error", err)
	}

	if len(episodes) == 0 {
		c.logger.Info("No new episodes to add")
		return &types.AddBulkEpisodeResults{}, nil
	}

	result := &types.AddBulkEpisodeResults{
		Episodes:       []*types.Node{},
		EpisodicEdges:  []*types.Edge{},
		Nodes:          []*types.Node{},
		Edges:          []*types.Edge{},
		Communities:    []*types.Node{},
		CommunityEdges: []*types.Edge{},
	}

	for _, episode := range episodes {
		episodeResult, err := c.AddEpisode(ctx, episode, options)
		if err != nil {
			return nil, fmt.Errorf("failed to process episode %s: %w", episode.ID, err)
		}

		// Aggregate results
		if episodeResult.Episode != nil {
			result.Episodes = append(result.Episodes, episodeResult.Episode)
		}
		result.EpisodicEdges = append(result.EpisodicEdges, episodeResult.EpisodicEdges...)
		result.Nodes = append(result.Nodes, episodeResult.Nodes...)
		result.Edges = append(result.Edges, episodeResult.Edges...)
		result.Communities = append(result.Communities, episodeResult.Communities...)
		result.CommunityEdges = append(result.CommunityEdges, episodeResult.CommunityEdges...)
	}

	return result, nil
}

// AddEpisode processes and adds a single episode to the knowledge graph.
// This implementation uses bulk processing with sophisticated deduplication.
// Content is automatically chunked if it exceeds MaxCharacters, but the same
// efficient bulk processing path is used for both single and multi-chunk episodes.
func (c *Client) AddEpisode(ctx context.Context, episode types.Episode, options *AddEpisodeOptions) (*types.AddEpisodeResults, error) {
	if options == nil {
		options = &AddEpisodeOptions{}
	}

	// Inject ingestion source into context for token tracking
	ingestionSource := episode.Source
	if ingestionSource == "" {
		ingestionSource = fmt.Sprintf("episode:%s", episode.ID)
	}
	ctx = context.WithValue(ctx, types.ContextKeyIngestionSource, ingestionSource)

	maxCharacters := 2048
	if options.MaxCharacters > 0 {
		maxCharacters = options.MaxCharacters
	}

	// Always use the bulk processing path for consistent, sophisticated deduplication
	// If content is small, it will be processed as a single chunk
	return c.addEpisodeChunked(ctx, episode, options, maxCharacters)
}

// addEpisodeChunked chunks long episode content and uses bulk deduplication
// processing across all chunks to efficiently handle large episodes.
func (c *Client) addEpisodeChunked(ctx context.Context, episode types.Episode, options *AddEpisodeOptions, maxCharacters int) (*types.AddEpisodeResults, error) {
	now := time.Now()

	// STEP 1: Prepare and validate episode
	chunks, err := c.prepareAndValidateEpisode(&episode, options, maxCharacters)
	if err != nil {
		return nil, err
	}

	// STEP 2: Get previous episodes for context
	previousEpisodes, err := c.getPreviousEpisodesForContext(ctx, episode, options)
	if err != nil {
		return nil, err
	}

	// STEP 3: Create chunk episode structures
	chunkData, err := c.createChunkEpisodeStructures(ctx, episode, chunks, previousEpisodes, options)
	if err != nil {
		return nil, err
	}

	// STEP 4: Initialize maintenance operations
	nodeOps := maintenance.NewNodeOperations(c.driver, c.nlProcessor, c.embedder, prompts.NewLibrary())
	nodeOps.ExtractionNLP = c.nlpModels.NodeExtraction
	nodeOps.ReflexionNLP = c.nlpModels.NodeReflexion
	nodeOps.ResolutionNLP = c.nlpModels.NodeResolution
	nodeOps.AttributeNLP = c.nlpModels.NodeAttribute
	nodeOps.SkipReflexion = options.SkipReflexion
	nodeOps.SkipResolution = options.SkipResolution
	nodeOps.SkipAttributes = options.SkipAttributes
	nodeOps.UseYAML = options.UseYAML
	nodeOps.SetLogger(c.logger)

	edgeOps := maintenance.NewEdgeOperations(c.driver, c.nlProcessor, c.embedder, prompts.NewLibrary())
	edgeOps.ExtractionNLP = c.nlpModels.EdgeExtraction
	edgeOps.ResolutionNLP = c.nlpModels.EdgeResolution
	edgeOps.SkipResolution = options.SkipEdgeResolution
	edgeOps.UseYAML = options.UseYAML
	edgeOps.SetLogger(c.logger)

	// STEP 5: Extract entities from all chunks
	extractedNodesByChunk, err := c.extractEntitiesFromAllChunks(ctx, episode.ID, chunkData.chunkEpisodeNodes, chunkData.previousEpisodes, options, nodeOps)
	if err != nil {
		return nil, err
	}

	// OPTIMIZATION: Filter out chunks with no extracted entities
	var filteredNodesByChunk [][]*types.Node
	var filteredEpisodeTuples []utils.EpisodeTuple
	chunksWithEntities := 0
	chunksWithoutEntities := 0

	for i, nodes := range extractedNodesByChunk {
		if len(nodes) > 0 {
			filteredNodesByChunk = append(filteredNodesByChunk, nodes)
			filteredEpisodeTuples = append(filteredEpisodeTuples, chunkData.episodeTuples[i])
			chunksWithEntities++
		} else {
			chunksWithoutEntities++
		}
	}

	c.logger.Info("Filtered chunks for processing",
		"episode_id", episode.ID,
		"total_chunks", len(extractedNodesByChunk),
		"chunks_with_entities", chunksWithEntities,
		"chunks_skipped", chunksWithoutEntities)

	var hydratedNodes []*types.Node
	var resolvedEdges []*types.Edge
	var invalidatedEdges []*types.Edge
	var episodicEdges []*types.Edge

	// Only process entities and relationships if we have chunks with entities
	if chunksWithEntities > 0 {
		// STEP 6: Deduplicate entities across chunks (only chunks with entities)
		dedupeResult, allResolvedNodes, err := c.deduplicateEntitiesAcrossChunks(ctx, episode.ID, filteredNodesByChunk, filteredEpisodeTuples, options, nodeOps)
		if err != nil {
			return nil, err
		}

		// STEP 7: Extract relationships
		allExtractedEdges, err := c.extractRelationshipsFromChunks(ctx, episode.ID, chunkData.mainEpisodeNode, dedupeResult, chunkData.previousEpisodes, options, edgeOps)
		if err != nil {
			return nil, err
		}

		// STEP 8: Resolve and persist relationships
		resolvedEdges, invalidatedEdges, err = c.resolveAndPersistRelationships(ctx, episode.ID, allExtractedEdges, chunkData.mainEpisodeNode, allResolvedNodes, options, edgeOps)
		if err != nil {
			return nil, err
		}

		// STEP 9: Extract attributes
		hydratedNodes, err = c.extractEntityAttributes(ctx, episode.ID, allResolvedNodes, chunkData.mainEpisodeNode, chunkData.previousEpisodes, options, nodeOps)
		if err != nil {
			return nil, err
		}

		// STEP 10: Build episodic edges
		episodicEdges, err = c.buildEpisodicEdgesForEntities(ctx, hydratedNodes, chunkData.mainEpisodeNode, now, edgeOps)
		if err != nil {
			return nil, err
		}

		// STEP 11: Perform final graph updates
		if err := c.performFinalGraphUpdates(ctx, episode.ID, chunkData.mainEpisodeNode, hydratedNodes, resolvedEdges, invalidatedEdges, episodicEdges); err != nil {
			return nil, err
		}
	} else {
		c.logger.Info("No entities extracted from any chunks, skipping entity and relationship processing",
			"episode_id", episode.ID)

		// Still need to persist the episode node with its content
		if err := c.driver.UpsertNode(ctx, chunkData.mainEpisodeNode); err != nil {
			return nil, fmt.Errorf("failed to persist episode node: %w", err)
		}
	}

	// STEP 12: Prepare result
	result := &types.AddEpisodeResults{
		Episode:        chunkData.mainEpisodeNode,
		EpisodicEdges:  episodicEdges,
		Nodes:          hydratedNodes,
		Edges:          append(resolvedEdges, invalidatedEdges...),
		Communities:    []*types.Node{},
		CommunityEdges: []*types.Edge{},
	}

	// STEP 13: Update communities
	communities, communityEdges, err := c.UpdateCommunities(ctx, episode.ID, episode.GroupID)
	if err != nil {
		return nil, err
	}
	result.Communities = communities
	result.CommunityEdges = communityEdges

	// STEP 14: Persist community nodes and edges using bulk operation
	if len(communities) > 0 || len(communityEdges) > 0 {
		_, err = utils.AddNodesAndEdgesBulk(ctx, c.driver, communities, communityEdges, []*types.Node{}, []*types.Edge{}, c.embedder)
		if err != nil {
			c.logger.Warn("Failed to persist community nodes and edges in bulk",
				"episode_id", episode.ID,
				"community_count", len(communities),
				"community_edge_count", len(communityEdges),
				"error", err)
		} else {
			c.logger.Info("Persisted community nodes and edges",
				"episode_id", episode.ID,
				"community_count", len(communities),
				"community_edge_count", len(communityEdges))
		}
	}

	// STEP 15: Log final results
	c.logger.Info("Chunked episode processing completed with bulk deduplication",
		"episode_id", episode.ID,
		"total_chunks", len(chunks),
		"total_entities", len(result.Nodes),
		"total_relationships", len(result.Edges),
		"total_episodic_edges", len(result.EpisodicEdges),
		"total_communities", len(result.Communities))

	// STEP 16: Report overall graph database statistics
	stats, err := c.driver.GetStats(ctx, episode.GroupID)
	if err == nil {
		c.logger.Info("Graph database statistics after episode processing",
			"episode_id", episode.ID,
			"group_id", episode.GroupID,
			"total_nodes", stats.NodeCount,
			"total_edges", stats.EdgeCount,
			"total_communities", stats.CommunityCount,
			"entity_nodes", stats.NodesByType["Entity"],
			"episodic_nodes", stats.NodesByType["Episodic"],
			"community_nodes", stats.NodesByType["Community"])
	} else {
		c.logger.Warn("Failed to retrieve graph database statistics",
			"episode_id", episode.ID,
			"error", err)
	}

	return result, nil
}

// createTempEpisodeForAdditionalContent creates a temporary episode structure with the additional content for processing.
func (c *Client) createTempEpisodeForAdditionalContent(existingEpisode *types.Node, episodeID string, additionalContent string, groupID string) types.Episode {
	return types.Episode{
		ID:        episodeID, // Use the same ID to link entities/edges to this episode
		Name:      existingEpisode.Name,
		Content:   additionalContent,
		GroupID:   groupID,
		Reference: existingEpisode.Reference,
		Metadata:  existingEpisode.Metadata,
	}
}

// updateEpisodeContent updates the existing episode's content by appending the additional content and saves it.
func (c *Client) updateEpisodeContent(ctx context.Context, existingEpisode *types.Node, additionalContent string) error {
	// Append with a newline separator if original content isn't empty
	updatedContent := existingEpisode.Content
	if updatedContent != "" {
		updatedContent += "\n"
	}
	updatedContent += additionalContent

	existingEpisode.Content = updatedContent
	existingEpisode.UpdatedAt = time.Now()

	// Save the updated episode node
	if err := c.driver.UpsertNode(ctx, existingEpisode); err != nil {
		return fmt.Errorf("failed to update episode content: %w", err)
	}

	return nil
}

// AddToEpisode expands an existing episode with additional content.
// It retrieves the existing episode, processes the additional content through entity
// and edge extraction, and appends the additional content to the episode's content field.
//
// Parameters:
//   - ctx: Context for cancellation and timeouts
//   - episodeID: The UUID of the existing episode to expand
//   - additionalContent: The new content to add and process
//   - options: Options for processing (entity types, embeddings, etc.)
//
// Returns:
//   - AddEpisodeResults containing the newly extracted entities and edges
//   - Error if the episode doesn't exist or processing fails
//
// AddToEpisode adds additional content to an existing episode, extracting new entities and relationships.
func (c *Client) AddToEpisode(ctx context.Context, episodeID string, additionalContent string, options *AddEpisodeOptions) (*types.AddEpisodeResults, error) {
	if options == nil {
		options = &AddEpisodeOptions{}
	}

	// Inject ingestion source into context for token tracking
	// For AddToEpisode, we use the episode ID as primary source ref
	ctx = context.WithValue(ctx, types.ContextKeyIngestionSource, fmt.Sprintf("episode_update:%s", episodeID))

	// Use the client's configured group ID
	groupID := c.config.GroupID

	// 1. Retrieve and validate the existing episode
	existingEpisode, err := c.retrieveAndValidateEpisode(ctx, episodeID, groupID)
	if err != nil {
		return nil, err
	}

	// 2. Create a temporary episode structure with the additional content for processing
	tempEpisode := c.createTempEpisodeForAdditionalContent(existingEpisode, episodeID, additionalContent, groupID)

	// 3. Process the additional content through entity and edge extraction
	maxCharacters := 4096
	if options.MaxCharacters > 0 {
		maxCharacters = options.MaxCharacters
	}

	results, err := c.addEpisodeChunked(ctx, tempEpisode, options, maxCharacters)
	if err != nil {
		return nil, fmt.Errorf("failed to process additional content: %w", err)
	}

	// 4. Update the existing episode's content
	if err := c.updateEpisodeContent(ctx, existingEpisode, additionalContent); err != nil {
		return nil, err
	}

	// 5. Log results
	c.logger.Info("Successfully expanded episode",
		"episode_id", episodeID,
		"additional_content_length", len(additionalContent),
		"new_total_length", len(existingEpisode.Content),
		"new_entities", len(results.Nodes),
		"new_edges", len(results.Edges))

	return results, nil
}

// chunkEpisodeData holds the prepared data structures for chunked episode processing.
type chunkEpisodeData struct {
	chunks            []string
	mainEpisodeNode   *types.Node
	chunkEpisodeNodes []*types.Node
	episodeTuples     []utils.EpisodeTuple
	previousEpisodes  []*types.Node
	prevEps           []*types.Episode
}

// prepareAndValidateEpisode chunks the episode content and validates entity types and group ID.
func (c *Client) prepareAndValidateEpisode(episode *types.Episode, options *AddEpisodeOptions, maxCharacters int) ([]string, error) {
	// Chunk the content
	chunks := chunkText(episode.Content, maxCharacters)

	c.logger.Info("Chunking episode content",
		"episode_id", episode.ID,
		"original_length", len(episode.Content),
		"num_chunks", len(chunks),
		"max_characters", maxCharacters)

	// Validate entity types
	if err := utils.ValidateEntityTypes(options.EntityTypes); err != nil {
		return nil, fmt.Errorf("invalid entity types: %w", err)
	}

	// Validate and set group ID
	if err := utils.ValidateGroupID(episode.GroupID); err != nil {
		return nil, fmt.Errorf("invalid group ID: %w", err)
	}
	if episode.GroupID == "" {
		episode.GroupID = utils.GetDefaultGroupID(c.driver.Provider())
	}

	return chunks, nil
}

// getPreviousEpisodesForContext retrieves previous episodes for providing context during entity extraction.
func (c *Client) getPreviousEpisodesForContext(ctx context.Context, episode types.Episode, options *AddEpisodeOptions) ([]*types.Node, error) {
	var previousEpisodes []*types.Node
	var err error

	if len(options.PreviousEpisodeUUIDs) > 0 {
		for _, uuid := range options.PreviousEpisodeUUIDs {
			episodeNode, err := c.driver.GetNode(ctx, uuid, episode.GroupID)
			if err == nil && episodeNode != nil {
				previousEpisodes = append(previousEpisodes, episodeNode)
			}
		}
	} else {
		previousEpisodes, err = c.RetrieveEpisodes(
			ctx,
			episode.Reference,
			[]string{episode.GroupID},
			search.RelevantSchemaLimit,
			nil,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to retrieve previous episodes: %w", err)
		}
	}

	return previousEpisodes, nil
}

// createChunkEpisodeStructures creates the episode nodes and tuples needed for processing each chunk.
func (c *Client) createChunkEpisodeStructures(ctx context.Context, episode types.Episode, chunks []string, previousEpisodes []*types.Node, options *AddEpisodeOptions) (*chunkEpisodeData, error) {
	data := &chunkEpisodeData{
		chunks:            chunks,
		chunkEpisodeNodes: make([]*types.Node, len(chunks)),
		episodeTuples:     make([]utils.EpisodeTuple, len(chunks)),
		previousEpisodes:  previousEpisodes,
	}

	// Convert previous episodes to Episode type for EpisodeTuple (reused for all chunks)
	data.prevEps = make([]*types.Episode, len(previousEpisodes))
	for j, prevNode := range previousEpisodes {
		data.prevEps[j] = &types.Episode{
			ID:        prevNode.Uuid,
			Name:      prevNode.Name,
			Content:   prevNode.Content,
			Reference: prevNode.ValidFrom,
			CreatedAt: prevNode.CreatedAt,
			GroupID:   prevNode.GroupID,
			Metadata:  prevNode.Metadata,
		}
	}

	// Create temporary episode nodes for entity extraction (one per chunk)
	for i, chunk := range chunks {
		chunkEpisode := types.Episode{
			ID:        episode.ID,
			Name:      episode.Name,
			Content:   chunk, // Individual chunk content for extraction
			Reference: episode.Reference,
			CreatedAt: episode.CreatedAt,
			GroupID:   episode.GroupID,
			Metadata:  episode.Metadata,
		}

		// Create temporary episode node for this chunk's extraction
		chunkNode := &types.Node{
			Uuid:      episode.ID,
			Name:      episode.Name,
			Type:      types.EpisodicNodeType,
			Content:   chunk,
			GroupID:   episode.GroupID,
			Metadata:  episode.Metadata,
			ValidFrom: episode.Reference,
			CreatedAt: episode.CreatedAt,
		}
		data.chunkEpisodeNodes[i] = chunkNode

		data.episodeTuples[i] = utils.EpisodeTuple{
			Episode:          &chunkEpisode,
			PreviousEpisodes: data.prevEps,
		}

		if i == 0 {
			// Create the actual persisted episode node with first chunk
			var err error
			data.mainEpisodeNode, err = c.createEpisodeNode(ctx, chunkEpisode, options)
			if err != nil {
				return nil, fmt.Errorf("failed to create episode node: %w", err)
			}
		}
	}

	// Update the main episode with full content
	fullContent := strings.Join(chunks, "\n")
	data.mainEpisodeNode.Content = fullContent
	data.mainEpisodeNode.UpdatedAt = time.Now()

	// STEP: Create source node and edge if episode has a source
	if episode.Source != "" {
		sourceNode, isNew, err := c.getOrCreateSourceNode(ctx, episode.Source, episode.GroupID)
		if err != nil {
			c.logger.Warn("Failed to create source node", "source", episode.Source, "error", err)
		} else if sourceNode != nil {
			if isNew {
				c.logger.Info("Created new source node for episode", "source", episode.Source, "episode_id", episode.ID)
			} else {
				c.logger.Debug("Using existing source node for episode", "source", episode.Source, "episode_id", episode.ID)
			}

			// Create edge from source to episode
			sourceEdge, err := c.createSourceEdge(ctx, sourceNode, data.mainEpisodeNode)
			if err != nil {
				c.logger.Warn("Failed to create source edge", "source", episode.Source, "episode_id", episode.ID, "error", err)
			} else if sourceEdge != nil {
				c.logger.Debug("Created source edge", "source", episode.Source, "episode_id", episode.ID, "edge_id", sourceEdge.Uuid)
			}
		}
	}

	return data, nil
}

// extractEntitiesFromAllChunks extracts entities from each chunk using the LLM.
func (c *Client) extractEntitiesFromAllChunks(ctx context.Context, episodeID string, chunkEpisodeNodes []*types.Node, previousEpisodes []*types.Node, options *AddEpisodeOptions, nodeOps *maintenance.NodeOperations) ([][]*types.Node, error) {
	c.logger.Info("Starting bulk entity extraction",
		"episode_id", episodeID,
		"num_chunks", len(chunkEpisodeNodes))

	extractedNodesByChunk := make([][]*types.Node, len(chunkEpisodeNodes))
	for i, chunkNode := range chunkEpisodeNodes {
		extractedNodes, err := nodeOps.ExtractNodes(ctx, chunkNode, previousEpisodes,
			options.EntityTypes, options.ExcludedEntityTypes)
		if err != nil {
			return nil, fmt.Errorf("failed to extract nodes from chunk %d: %w", i, err)
		}
		extractedNodesByChunk[i] = extractedNodes
	}

	totalExtracted := 0
	for _, nodes := range extractedNodesByChunk {
		totalExtracted += len(nodes)
	}

	c.logger.Info("Bulk entity extraction completed",
		"episode_id", episodeID,
		"total_entities_extracted", totalExtracted)

	return extractedNodesByChunk, nil
}

// deduplicateEntitiesAcrossChunks performs bulk entity deduplication across all chunks and persists them.
func (c *Client) deduplicateEntitiesAcrossChunks(ctx context.Context, episodeID string, extractedNodesByChunk [][]*types.Node, episodeTuples []utils.EpisodeTuple, options *AddEpisodeOptions, nodeOps *maintenance.NodeOperations) (*utils.DedupeNodesResult, []*types.Node, error) {
	c.logger.Info("Starting bulk entity deduplication",
		"episode_id", episodeID,
		"num_chunks", len(extractedNodesByChunk))

	clients := &utils.Clients{
		Driver:   c.driver,
		NLP:      c.nlProcessor,
		Embedder: c.embedder,
		Prompts:  prompts.NewLibrary(),
	}

	dedupeResult, err := utils.DedupeNodesBulk(
		ctx,
		clients,
		extractedNodesByChunk,
		episodeTuples,
		options.EntityTypes,
		&nodeOpsWrapper{nodeOps},
	)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to deduplicate nodes in bulk: %w", err)
	}

	c.logger.Info("Bulk entity deduplication completed",
		"episode_id", episodeID,
		"uuid_mappings", len(dedupeResult.UUIDMap))

	// Collect all resolved nodes across chunks
	var allResolvedNodes []*types.Node
	seenNodeIDs := make(map[string]bool)
	for _, nodes := range dedupeResult.NodesByEpisode {
		for _, node := range nodes {
			// Guard against nil nodes
			if node == nil {
				c.logger.Warn("Encountered nil node in deduplication result", "episode_id", episodeID)
				continue
			}
			if !seenNodeIDs[node.Uuid] {
				allResolvedNodes = append(allResolvedNodes, node)
				seenNodeIDs[node.Uuid] = true
			}
		}
	}

	// EARLY WRITE: Persist deduplicated nodes
	c.logger.Info("Persisting deduplicated nodes early",
		"episode_id", episodeID,
		"num_nodes", len(allResolvedNodes))

	validNodes := 0
	for i, node := range allResolvedNodes {
		// Comprehensive validation before persistence
		if node == nil {
			c.logger.Warn("Skipping nil node during persistence",
				"episode_id", episodeID,
				"node_index", i)
			continue
		}
		if !validateNodeForPersistence(node, episodeID, i, c.logger) {
			continue
		}

		if err := c.driver.UpsertNode(ctx, node); err != nil {
			c.logger.Warn("Failed to persist deduplicated node",
				"episode_id", episodeID,
				"node_index", i,
				"node_id", node.Uuid,
				"node_name", node.Name,
				"error", err)
		} else {
			validNodes++
		}
	}

	c.logger.Info("Deduplicated nodes persisted",
		"episode_id", episodeID,
		"total_nodes", len(allResolvedNodes),
		"valid_nodes", validNodes,
		"skipped_nodes", len(allResolvedNodes)-validNodes)

	return dedupeResult, allResolvedNodes, nil
}

// validateNodeForPersistence performs comprehensive validation on a node before database persistence
// to prevent segmentation faults caused by invalid or incomplete data.
func validateNodeForPersistence(node *types.Node, episodeID string, index int, logger *slog.Logger) bool {
	// Check 1: Nil node
	if node == nil {
		logger.Warn("Skipping nil node during persistence",
			"episode_id", episodeID,
			"node_index", index)
		return false
	}

	// Check 2: Required field - Node ID
	if node.Uuid == "" {
		logger.Warn("Skipping node with empty ID",
			"episode_id", episodeID,
			"node_index", index,
			"node_name", node.Name)
		return false
	}

	// Check 3: Required field - Node Name
	if node.Name == "" {
		logger.Warn("Skipping node with empty Name",
			"episode_id", episodeID,
			"node_index", index,
			"node_id", node.Uuid)
		return false
	}

	// Check 4: Required field - GroupID
	if node.GroupID == "" {
		logger.Warn("Skipping node with empty GroupID",
			"episode_id", episodeID,
			"node_index", index,
			"node_id", node.Uuid,
			"node_name", node.Name)
		return false
	}

	// Check 5: Required field - Node Type
	if node.Type == "" {
		logger.Warn("Skipping node with empty Type",
			"episode_id", episodeID,
			"node_index", index,
			"node_id", node.Uuid,
			"node_name", node.Name)
		return false
	}

	// Check 6: Validate timestamps are not zero
	if node.CreatedAt.IsZero() {
		logger.Warn("Node has zero CreatedAt timestamp, setting to now",
			"episode_id", episodeID,
			"node_index", index,
			"node_id", node.Uuid,
			"node_name", node.Name)
		node.CreatedAt = time.Now()
	}

	if node.UpdatedAt.IsZero() {
		node.UpdatedAt = time.Now()
	}

	if node.ValidFrom.IsZero() {
		node.ValidFrom = time.Now()
	}

	// Check 7: Initialize Metadata if nil
	if node.Metadata == nil {
		node.Metadata = make(map[string]interface{})
	}

	// All validation checks passed
	return true
}

// extractRelationshipsFromChunks extracts relationships between entities using the LLM.
func (c *Client) extractRelationshipsFromChunks(ctx context.Context, episodeID string, mainEpisodeNode *types.Node, dedupeResult *utils.DedupeNodesResult, previousEpisodes []*types.Node, options *AddEpisodeOptions, edgeOps *maintenance.EdgeOperations) ([]*types.Edge, error) {
	c.logger.Info("Starting bulk relationship extraction",
		"episode_id", episodeID,
		"num_chunks", len(dedupeResult.NodesByEpisode))

	var allExtractedEdges []*types.Edge
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

	// Get nodes for the episode and extract edges
	episodeNodes := dedupeResult.NodesByEpisode[mainEpisodeNode.Uuid]
	if len(episodeNodes) > 0 {
		extractedEdges, err := edgeOps.ExtractEdges(ctx, mainEpisodeNode, episodeNodes,
			previousEpisodes, edgeTypeMap, options.EdgeTypes, mainEpisodeNode.GroupID)
		if err != nil {
			return nil, fmt.Errorf("failed to extract edges: %w", err)
		}

		// Apply UUID mapping to edge pointers
		utils.ResolveEdgePointers(extractedEdges, dedupeResult.UUIDMap)
		allExtractedEdges = extractedEdges
	}

	c.logger.Info("Bulk relationship extraction completed",
		"episode_id", episodeID,
		"total_relationships_extracted", len(allExtractedEdges))

	return allExtractedEdges, nil
}

// resolveAndPersistRelationships resolves extracted relationships and persists them to the graph.
func (c *Client) resolveAndPersistRelationships(ctx context.Context, episodeID string, allExtractedEdges []*types.Edge, mainEpisodeNode *types.Node, allResolvedNodes []*types.Node, options *AddEpisodeOptions, edgeOps *maintenance.EdgeOperations) ([]*types.Edge, []*types.Edge, error) {
	c.logger.Info("Starting bulk relationship resolution",
		"episode_id", episodeID,
		"relationships_to_resolve", len(allExtractedEdges))

	var resolvedEdges []*types.Edge
	var invalidatedEdges []*types.Edge
	var err error

	if len(allExtractedEdges) > 0 {
		resolvedEdges, invalidatedEdges, err = edgeOps.ResolveExtractedEdges(ctx,
			allExtractedEdges, mainEpisodeNode, allResolvedNodes, options.GenerateEmbeddings, options.EdgeTypes)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to resolve edges: %w", err)
		}
	}

	c.logger.Info("Bulk relationship resolution completed",
		"episode_id", episodeID,
		"resolved_relationships", len(resolvedEdges),
		"invalidated_relationships", len(invalidatedEdges))

	// EARLY WRITE: Persist resolved edges
	c.logger.Info("Persisting resolved edges early",
		"episode_id", episodeID,
		"num_edges", len(resolvedEdges)+len(invalidatedEdges))

	allResolvedEdges := append(resolvedEdges, invalidatedEdges...)
	for _, edge := range allResolvedEdges {
		if err := c.driver.UpsertEdge(ctx, edge); err != nil {
			c.logger.Warn("Failed to persist resolved edge",
				"episode_id", episodeID,
				"edge_id", edge.Uuid,
				"error", err)
		}
	}

	c.logger.Info("Resolved edges persisted",
		"episode_id", episodeID,
		"num_edges", len(allResolvedEdges))

	return resolvedEdges, invalidatedEdges, nil
}

// extractEntityAttributes extracts attributes for all resolved entities.
func (c *Client) extractEntityAttributes(ctx context.Context, episodeID string, allResolvedNodes []*types.Node, mainEpisodeNode *types.Node, previousEpisodes []*types.Node, options *AddEpisodeOptions, nodeOps *maintenance.NodeOperations) ([]*types.Node, error) {
	c.logger.Info("Starting bulk attribute extraction",
		"episode_id", episodeID,
		"entities_to_hydrate", len(allResolvedNodes))

	hydratedNodes, err := nodeOps.ExtractAttributesFromNodes(ctx,
		allResolvedNodes, mainEpisodeNode, previousEpisodes, options.EntityTypes)
	if err != nil {
		return nil, fmt.Errorf("failed to extract attributes: %w", err)
	}

	c.logger.Info("Bulk attribute extraction completed",
		"episode_id", episodeID,
		"hydrated_entities", len(hydratedNodes))

	return hydratedNodes, nil
}

// buildEpisodicEdgesForEntities creates edges linking entities to the episode.
func (c *Client) buildEpisodicEdgesForEntities(ctx context.Context, hydratedNodes []*types.Node, mainEpisodeNode *types.Node, now time.Time, edgeOps *maintenance.EdgeOperations) ([]*types.Edge, error) {
	episodicEdges, err := edgeOps.BuildEpisodicEdges(ctx, hydratedNodes, mainEpisodeNode.Uuid, now)
	if err != nil {
		return nil, fmt.Errorf("failed to build episodic edges: %w", err)
	}
	return episodicEdges, nil
}

// performFinalGraphUpdates performs the final bulk update of nodes and edges to the graph.
func (c *Client) performFinalGraphUpdates(ctx context.Context, episodeID string, mainEpisodeNode *types.Node, hydratedNodes []*types.Node, resolvedEdges []*types.Edge, invalidatedEdges []*types.Edge, episodicEdges []*types.Edge) error {
	allEdges := append(resolvedEdges, invalidatedEdges...)

	c.logger.Info("Starting final updates",
		"episode_id", episodeID,
		"episodic_nodes", 1,
		"entity_nodes_to_update", len(hydratedNodes),
		"entity_edges_to_update", len(allEdges),
		"episodic_edges_to_add", len(episodicEdges))

	_, err := utils.AddNodesAndEdgesBulk(ctx, c.driver,
		[]*types.Node{mainEpisodeNode},
		episodicEdges,
		hydratedNodes,
		allEdges,
		c.embedder)
	if err != nil {
		return fmt.Errorf("failed to perform final updates: %w", err)
	}

	// Report final database statistics after bulk operations
	if stats, err := c.GetStats(ctx); err == nil {
		episodesInDB := int64(0)
		if stats.NodesByType != nil {
			episodesInDB = stats.NodesByType["Episodic"]
		}
		c.logger.Info("Final database state after bulk operations",
			"node_count", stats.NodeCount,
			"edge_count", stats.EdgeCount,
			"episodes_in_db", episodesInDB,
			"communities", stats.CommunityCount)
	} else {
		c.logger.Warn("Failed to retrieve final database stats", "error", err)
	}

	return nil
}

// UpdateCommunities updates graph communities if requested in options.
func (c *Client) UpdateCommunities(ctx context.Context, episodeID string, groupID string) ([]*types.Node, []*types.Edge, error) {

	c.logger.Info("Starting community update",
		"episode_id", episodeID,
		"group_id", groupID)

	communityResult, err := c.community.BuildCommunities(ctx, []string{groupID}, c.logger)
	if err != nil && len(communityResult.CommunityNodes) == 0 {
		return nil, nil, fmt.Errorf("failed to build communities: %w", err)
	}

	c.logger.Info("Community update completed",
		"episode_id", episodeID,
		"communities", len(communityResult.CommunityNodes),
		"community_edges", len(communityResult.CommunityEdges))

	return communityResult.CommunityNodes, communityResult.CommunityEdges, nil
}

// createEpisodeNode creates an episode node in the graph.
func (c *Client) createEpisodeNode(ctx context.Context, episode types.Episode, options *AddEpisodeOptions) (*types.Node, error) {
	now := time.Now()

	// Use existing embedding or create new one if embedder is available
	var embedding []float32
	if len(episode.ContentEmbedding) > 0 {
		// Use pre-computed embedding if available
		embedding = episode.ContentEmbedding
	} else if c.embedder != nil {
		// Generate embedding if not provided and embedder is available
		var err error
		embedding, err = c.embedder.EmbedSingle(ctx, episode.Content)
		if err != nil {
			return nil, fmt.Errorf("failed to create episode embedding: %w", err)
		}
	}

	episodeNode := &types.Node{
		Uuid:        episode.ID,
		Name:        episode.Name,
		Type:        types.EpisodicNodeType,
		GroupID:     episode.GroupID,
		CreatedAt:   now,
		UpdatedAt:   now,
		EpisodeType: types.ConversationEpisodeType, // Default to conversation type
		Content:     episode.Content,
		Reference:   episode.Reference,
		ValidFrom:   episode.Reference,
		Embedding:   embedding,
		Metadata:    episode.Metadata,
	}

	if err := c.driver.UpsertNode(ctx, episodeNode); err != nil {
		return nil, fmt.Errorf("failed to create episode node: %w", err)
	}

	return episodeNode, nil
}

// ExtractedEntity represents an entity extracted by the LLM
// Supports multiple field names for compatibility with different LLM response formats
type ExtractedEntity struct {
	Name         string `json:"name"`        // Expected format
	Entity       string `json:"entity"`      // Common LLM format
	EntityName   string `json:"entity_name"` // Alternative LLM format
	EntityTypeID int    `json:"entity_type_id"`
}

// GetEntityName returns the entity name, checking all possible field names
func (e *ExtractedEntity) GetEntityName() string {
	if e.Name != "" {
		return e.Name
	}
	if e.Entity != "" {
		return e.Entity
	}
	return e.EntityName
}

// ExtractedEntities represents the response from entity extraction (multiple formats)
type ExtractedEntities struct {
	ExtractedEntities []ExtractedEntity `json:"extracted_entities"` // Expected format
	Entities          []ExtractedEntity `json:"entities"`           // Alternative LLM format
}

// GetEntitiesList returns the entities list, checking all possible field names
func (e *ExtractedEntities) GetEntitiesList() []ExtractedEntity {
	if len(e.ExtractedEntities) > 0 {
		return e.ExtractedEntities
	}
	return e.Entities
}

// ParseEntitiesFromResponse parses the LLM response and converts it to Node structures
func (c *Client) ParseEntitiesFromJsonResponse(responseContent, groupID string) ([]*types.Node, error) {
	// 1. Parse the structured JSON response from the LLM
	responseContent, _ = jsonrepair.JSONRepair(responseContent)

	var entitiesList []ExtractedEntity

	// Try multiple parsing strategies to handle different LLM response formats

	// Strategy 1: Try to parse as wrapped format {"extracted_entities": [...]} or {"entities": [...]}
	var extractedEntities ExtractedEntities
	if err := json.Unmarshal([]byte(responseContent), &extractedEntities); err == nil {
		entitiesList = extractedEntities.GetEntitiesList()
	}

	// Strategy 2: If wrapped format didn't work or was empty, try direct array
	if len(entitiesList) == 0 {
		if err := json.Unmarshal([]byte(responseContent), &entitiesList); err != nil {
			// Strategy 3: Try to extract JSON from response text
			jsonStart := strings.Index(responseContent, "[")
			if jsonStart == -1 {
				jsonStart = strings.Index(responseContent, "{")
			}
			jsonEnd := strings.LastIndex(responseContent, "]")
			if jsonEnd == -1 {
				jsonEnd = strings.LastIndex(responseContent, "}")
			}

			if jsonStart != -1 && jsonEnd != -1 && jsonEnd > jsonStart {
				jsonContent := responseContent[jsonStart : jsonEnd+1]

				// Try direct array first
				if err := json.Unmarshal([]byte(jsonContent), &entitiesList); err != nil {
					// Try wrapped format
					var wrappedEntities ExtractedEntities
					if err := json.Unmarshal([]byte(jsonContent), &wrappedEntities); err != nil {
						// If all JSON parsing fails, fall back to simple text parsing
						return c.parseEntitiesFromText(responseContent, groupID)
					} else {
						entitiesList = wrappedEntities.GetEntitiesList()
					}
				}
			} else {
				// Fall back to simple text parsing
				return c.parseEntitiesFromText(responseContent, groupID)
			}
		}
	}

	// 2. Process the extracted entities list
	entities := make([]*types.Node, 0, len(entitiesList))
	now := time.Now()

	// Default entity types (matching Python implementation)
	entityTypes := map[int]string{
		0: "Entity", // Default entity type
	}

	// 3. Create proper EntityNode objects with all attributes
	for _, extractedEntity := range entitiesList {
		// Get entity name using flexible field mapping
		entityName := strings.TrimSpace(extractedEntity.GetEntityName())

		// Skip empty names
		if entityName == "" {
			continue
		}

		// Determine entity type from ID
		entityType := "Entity" // Default
		if entityTypeName, exists := entityTypes[extractedEntity.EntityTypeID]; exists {
			entityType = entityTypeName
		}

		entity := &types.Node{
			Uuid:       generateID(),
			Name:       entityName,
			Type:       types.EntityNodeType,
			GroupID:    groupID,
			CreatedAt:  now,
			UpdatedAt:  now,
			ValidFrom:  now,
			EntityType: entityType,
			Summary:    "", // Will be populated later if needed
			Metadata:   make(map[string]interface{}),
		}

		// Add entity type information to metadata
		entity.Metadata["entity_type_id"] = extractedEntity.EntityTypeID
		entity.Metadata["labels"] = []string{"Entity", entityType}

		entities = append(entities, entity)
	}

	return entities, nil
}

// parseEntitiesFromText provides fallback text-based parsing when JSON parsing fails
func (c *Client) parseEntitiesFromText(responseContent, groupID string) ([]*types.Node, error) {
	entities := []*types.Node{}
	now := time.Now()

	// Simple text-based extraction as fallback
	lines := strings.Split(responseContent, "\n")

	for _, line := range lines {
		line = strings.TrimSpace(line)

		// Look for entity patterns in various formats
		patterns := []string{
			"entity:", "Entity:", "name:", "Name:",
			"- entity:", "- Entity:", "* entity:", "* Entity:",
		}

		for _, pattern := range patterns {
			if strings.Contains(strings.ToLower(line), strings.ToLower(pattern)) {
				// Extract entity name from the line
				if strings.Contains(line, ":") {
					parts := strings.Split(line, ":")
					if len(parts) > 1 {
						entityName := strings.TrimSpace(parts[1])
						entityName = strings.Trim(entityName, `"'.,`)

						if entityName != "" && len(entityName) > 2 {
							entity := &types.Node{
								Uuid:       generateID(),
								Name:       entityName,
								Type:       types.EntityNodeType,
								GroupID:    groupID,
								CreatedAt:  now,
								UpdatedAt:  now,
								ValidFrom:  now,
								EntityType: "Entity",
								Summary:    "",
								Metadata:   make(map[string]interface{}),
							}
							entities = append(entities, entity)
						}
					}
				}
				break
			}
		}
	}

	return entities, nil
}

// AddTriplet adds a single triplet (source node, edge, target node) to the knowledge graph.
func (c *Client) AddTriplet(ctx context.Context, sourceNode *types.Node, edge *types.Edge, targetNode *types.Node, createEmbeddings bool) (*types.AddTripletResults, error) {
	if sourceNode == nil || edge == nil || targetNode == nil {
		return nil, fmt.Errorf("source node, edge, and target node must not be nil")
	}

	// Step 1: Generate name embeddings for nodes if missing (lines 1024-1027)
	// Equivalent to: if source_node.name_embedding is None: await source_node.generate_name_embedding(self.embedder)
	if len(sourceNode.NameEmbedding) == 0 && c.embedder != nil {
		embedding, err := c.embedder.EmbedSingle(ctx, sourceNode.Name)
		if err != nil {
			return nil, fmt.Errorf("failed to generate name embedding for source node: %w", err)
		}
		sourceNode.NameEmbedding = embedding
	}

	if len(targetNode.NameEmbedding) == 0 && c.embedder != nil {
		embedding, err := c.embedder.EmbedSingle(ctx, targetNode.Name)
		if err != nil {
			return nil, fmt.Errorf("failed to generate name embedding for target node: %w", err)
		}
		targetNode.NameEmbedding = embedding
	}

	// Step 2: Generate fact embedding for edge if missing (lines 1028-1029)
	// Equivalent to: if edge.fact_embedding is None: await edge.generate_embedding(self.embedder)
	if len(edge.FactEmbedding) == 0 && c.embedder != nil {
		embedding, err := c.embedder.EmbedSingle(ctx, edge.Fact)
		if err != nil {
			return nil, fmt.Errorf("failed to generate fact embedding for edge: %w", err)
		}
		edge.FactEmbedding = embedding
	}

	// Step 3: Resolve extracted nodes (lines 1031-1034)
	nodeOps := maintenance.NewNodeOperations(c.driver, c.nlProcessor, c.embedder, prompts.NewLibrary())
	nodeOps.SetLogger(c.logger)
	nodes, uuidMap, _, err := nodeOps.ResolveExtractedNodes(ctx, []*types.Node{sourceNode, targetNode}, nil, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve extracted nodes: %w", err)
	}

	// Step 4: Update edge pointers to resolved node UUIDs (line 1036)
	utils.ResolveEdgePointers([]*types.Edge{edge}, uuidMap)
	updatedEdge := edge // The edge is updated in-place

	// Step 6: Search for related edges with edge UUID filters (lines 1042-1050)

	searchFilters := &search.SearchFilters{
		EdgeTypes: []types.EdgeType{types.EntityEdgeType}, // Filter for entity edges
	}

	// Use edge hybrid search RRF config
	edgeSearchConfig := &search.SearchConfig{
		EdgeConfig: &search.EdgeSearchConfig{
			SearchMethods: []search.SearchMethod{search.BM25, search.CosineSimilarity},
			Reranker:      search.RRFRerankType,
			MinScore:      0.0,
		},
		Limit:    20,
		MinScore: 0.0,
	}

	relatedResults, err := c.searcher.Search(ctx, updatedEdge.Summary, edgeSearchConfig, searchFilters, updatedEdge.GroupID)
	if err != nil {
		return nil, fmt.Errorf("failed to search for related edges: %w", err)
	}
	relatedEdges := relatedResults.Edges

	// Step 7: Search for existing edges without filters (lines 1051-1059)
	existingResults, err := c.searcher.Search(ctx, updatedEdge.Summary, edgeSearchConfig, &search.SearchFilters{}, updatedEdge.GroupID)
	if err != nil {
		return nil, fmt.Errorf("failed to search for existing edges: %w", err)
	}
	existingEdges := existingResults.Edges

	// Step 8: Create EpisodicNode exactly as in Python (lines 1066-1074)
	var validAt time.Time
	if !updatedEdge.ValidFrom.IsZero() {
		validAt = updatedEdge.ValidFrom
	} else {
		validAt = time.Now()
	}

	episodicNode := &types.Node{
		Name:        "",
		Type:        types.EpisodicNodeType,
		EpisodeType: types.DocumentEpisodeType, // Equivalent to Python's EpisodeType.text
		Content:     "",
		Summary:     "",
		ValidFrom:   validAt,
		GroupID:     updatedEdge.GroupID,
	}

	// Step 9: Resolve extracted edge (lines 1061-1077)
	resolvedEdge, invalidatedEdges, err := c.resolveExtractedEdgeExact(ctx, updatedEdge, relatedEdges, existingEdges, episodicNode, createEmbeddings)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve extracted edge: %w", err)
	}

	// Step 10: Combine all edges (line 1079)
	allEdges := []*types.Edge{resolvedEdge}
	allEdges = append(allEdges, invalidatedEdges...)

	// Step 11: Create entity edge embeddings (line 1081)
	err = c.createEntityEdgeEmbeddings(ctx, allEdges)
	if err != nil {
		return nil, fmt.Errorf("failed to create entity edge embeddings: %w", err)
	}

	// Step 12: Create entity node embeddings (line 1082)
	err = c.createEntityNodeEmbeddings(ctx, nodes)
	if err != nil {
		return nil, fmt.Errorf("failed to create entity node embeddings: %w", err)
	}

	// Step 13: Add nodes and edges in bulk (line 1084)
	_, err = utils.AddNodesAndEdgesBulk(ctx, c.driver, []*types.Node{}, []*types.Edge{}, nodes, allEdges, c.embedder)
	if err != nil {
		return nil, fmt.Errorf("failed to add nodes and edges to database: %w", err)
	}

	// Step 14: Return results (line 1085)
	return &types.AddTripletResults{
		Edges: allEdges,
		Nodes: nodes,
	}, nil
}

// resolveExtractedEdgeExact is an exact translation of Python's resolve_extracted_edge function
func (c *Client) resolveExtractedEdgeExact(ctx context.Context, extractedEdge *types.Edge, relatedEdges []*types.Edge, existingEdges []*types.Edge, episode *types.Node, createEmbeddings bool) (*types.Edge, []*types.Edge, error) {
	// Use the EdgeOperations to resolve the edge exactly as in Python
	edgeOps := maintenance.NewEdgeOperations(c.driver, c.nlProcessor, c.embedder, prompts.NewLibrary())
	edgeOps.SetLogger(c.logger)

	// The Go implementation wraps the private resolveExtractedEdge method
	// We'll use ResolveExtractedEdges which internally calls the same logic
	resolvedEdges, invalidatedEdges, err := edgeOps.ResolveExtractedEdges(ctx, []*types.Edge{extractedEdge}, episode, []*types.Node{}, createEmbeddings, c.config.EdgeTypes)
	if err != nil {
		return nil, nil, err
	}

	var resolvedEdge *types.Edge
	if len(resolvedEdges) > 0 {
		resolvedEdge = resolvedEdges[0]
	} else {
		resolvedEdge = extractedEdge
	}

	return resolvedEdge, invalidatedEdges, nil
}

// createEntityEdgeEmbeddings creates embeddings for entity edges (equivalent to Python's create_entity_edge_embeddings)
func (c *Client) createEntityEdgeEmbeddings(ctx context.Context, edges []*types.Edge) error {
	if c.embedder == nil {
		return nil
	}

	for _, edge := range edges {
		if edge.Type == types.EntityEdgeType && len(edge.Embedding) == 0 && edge.Summary != "" {
			embedding, err := c.embedder.EmbedSingle(ctx, edge.Summary)
			if err != nil {
				return fmt.Errorf("failed to create embedding for edge %s: %w", edge.Uuid, err)
			}
			edge.Embedding = embedding
		}
	}

	return nil
}

// createEntityNodeEmbeddings creates embeddings for entity nodes (equivalent to Python's create_entity_node_embeddings)
func (c *Client) createEntityNodeEmbeddings(ctx context.Context, nodes []*types.Node) error {
	if c.embedder == nil {
		return nil
	}

	for _, node := range nodes {
		if node.Type == types.EntityNodeType && len(node.Embedding) == 0 && node.Name != "" {
			embedding, err := c.embedder.EmbedSingle(ctx, node.Name)
			if err != nil {
				return fmt.Errorf("failed to create embedding for node %s: %w", node.Uuid, err)
			}
			node.Embedding = embedding
		}
	}

	return nil
}

func GenerateViaCsv[T any](ctx context.Context, client Predicato, messages []types.Message) ([]T, error) {
	return nil, nil
}

// getOrCreateSourceNode retrieves an existing source node or creates a new one if it doesn't exist.
// Returns the source node and a boolean indicating whether a new node was created.
func (c *Client) getOrCreateSourceNode(ctx context.Context, sourceName string, groupID string) (*types.Node, bool, error) {
	if sourceName == "" {
		return nil, false, nil
	}

	// Try to find an existing source node with this name
	// Search for source nodes by name in the group
	searchResults, err := c.driver.SearchNodes(ctx, sourceName, groupID, &driver.SearchOptions{
		Limit:       1,
		NodeTypes:   []types.NodeType{types.SourceNodeType},
		UseFullText: false,
		ExactMatch:  true,
	})
	if err != nil {
		c.logger.Warn("Failed to search for existing source node", "source", sourceName, "error", err)
	}

	// If we found an existing source node with exact name match, return it
	if len(searchResults) > 0 {
		for _, node := range searchResults {
			if node.Name == sourceName && node.Type == types.SourceNodeType {
				c.logger.Debug("Found existing source node", "source", sourceName, "node_id", node.Uuid)
				return node, false, nil
			}
		}
	}

	// Create a new source node
	now := time.Now()
	sourceNode := &types.Node{
		Uuid:      generateID(),
		Name:      sourceName,
		Type:      types.SourceNodeType,
		GroupID:   groupID,
		CreatedAt: now,
		UpdatedAt: now,
		ValidFrom: now,
		Metadata:  make(map[string]interface{}),
		Summary:   fmt.Sprintf("Content source: %s", sourceName),
	}

	// Persist the source node
	if err := c.driver.UpsertNode(ctx, sourceNode); err != nil {
		return nil, false, fmt.Errorf("failed to create source node: %w", err)
	}

	c.logger.Info("Created new source node", "source", sourceName, "node_id", sourceNode.Uuid)
	return sourceNode, true, nil
}

// createSourceEdge creates an edge connecting a source node to an episode node.
func (c *Client) createSourceEdge(ctx context.Context, sourceNode *types.Node, episodeNode *types.Node) (*types.Edge, error) {
	if sourceNode == nil || episodeNode == nil {
		return nil, nil
	}

	now := time.Now()
	edge := &types.Edge{
		BaseEdge: types.BaseEdge{
			Uuid:         generateID(),
			GroupID:      episodeNode.GroupID,
			SourceNodeID: sourceNode.Uuid,
			TargetNodeID: episodeNode.Uuid,
			CreatedAt:    now,
			Metadata:     make(map[string]interface{}),
		},
		Name:      "SOURCED_FROM",
		Fact:      fmt.Sprintf("Episode '%s' is sourced from '%s'", episodeNode.Name, sourceNode.Name),
		UpdatedAt: now,
		ValidFrom: now,
		Episodes:  []string{episodeNode.Uuid},
	}

	// Set the type
	edge.Type = types.SourceEdgeType

	// Sync backward compatibility fields
	edge.SourceID = edge.SourceNodeID
	edge.TargetID = edge.TargetNodeID

	// Persist the edge
	if err := c.driver.UpsertEdge(ctx, edge); err != nil {
		return nil, fmt.Errorf("failed to create source edge: %w", err)
	}

	c.logger.Debug("Created source edge", "source", sourceNode.Name, "episode", episodeNode.Uuid, "edge_id", edge.Uuid)
	return edge, nil
}
