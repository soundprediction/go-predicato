package utils

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/embedder"
	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/prompts"
	"github.com/soundprediction/predicato/pkg/types"
)

// NodeOperations interface to avoid import cycle with maintenance package
// The duplicate pairs are returned as simple string pair tuples [source_uuid, target_uuid]
type NodeOperations interface {
	ResolveExtractedNodes(ctx context.Context, extractedNodes []*types.Node, episode *types.Node, previousEpisodes []*types.Node, entityTypes map[string]interface{}) ([]*types.Node, map[string]string, interface{}, error)
}

// Clients represents the set of clients needed for bulk operations
type Clients struct {
	Driver   driver.GraphDriver
	NLP      nlp.Client
	Embedder embedder.Client
	Prompts  prompts.Library
}

// ExtractNodesAndEdgesResult represents the result of bulk node and edge extraction
type ExtractNodesAndEdgesResult struct {
	ExtractedNodes []*types.Node
	ExtractedEdges []*types.Edge
}

// AddNodesAndEdgesResult represents the result of bulk add operations
type AddNodesAndEdgesResult struct {
	EpisodicNodes []*types.Node
	EpisodicEdges []*types.Edge
	EntityNodes   []*types.Node
	EntityEdges   []*types.Edge
	Errors        []error
}

// RetrievePreviousEpisodesBulk retrieves previous episodes for a list of episodes
// This matches the Python function signature: retrieve_previous_episodes_bulk(driver, episodes)
func RetrievePreviousEpisodesBulk(ctx context.Context, driver driver.GraphDriver, episodes []*types.Episode) ([]EpisodeTuple, error) {
	var episodeTuples []EpisodeTuple

	for _, episode := range episodes {
		// Get previous episodes using temporal search
		// Get nodes in the time range before this episode
		previousNodes, err := driver.GetNodesInTimeRange(ctx, episode.CreatedAt.Add(-24*time.Hour), episode.CreatedAt, episode.GroupID)
		if err != nil {
			return nil, fmt.Errorf("failed to get previous episodes for group %s: %w", episode.GroupID, err)
		}

		// Convert Node results to Episodes and filter for episodic nodes
		var prevEpisodes []*types.Episode
		for _, node := range previousNodes {
			if node.Type == types.EpisodicNodeType && node.Uuid != episode.ID {
				prevEpisodes = append(prevEpisodes, &types.Episode{
					ID:        node.Uuid,
					Name:      node.Name,
					Content:   node.Content,
					Reference: node.Reference,
					CreatedAt: node.CreatedAt,
					GroupID:   node.GroupID,
					Metadata:  node.Metadata,
				})
			}
		}

		episodeTuples = append(episodeTuples, EpisodeTuple{
			Episode:          episode,
			PreviousEpisodes: prevEpisodes,
		})
	}

	return episodeTuples, nil
}

// AddNodesAndEdgesBulk adds nodes and edges to the graph database in bulk
// This matches the Python function signature: add_nodes_and_edges_bulk(driver, episodic_nodes, episodic_edges, entity_nodes, entity_edges, embedder)
func AddNodesAndEdgesBulk(
	ctx context.Context,
	driver driver.GraphDriver,
	episodicNodes []*types.Node,
	episodicEdges []*types.Edge,
	entityNodes []*types.Node,
	entityEdges []*types.Edge,
	embedder embedder.Client,
) (*AddNodesAndEdgesResult, error) {
	result := &AddNodesAndEdgesResult{}

	// Add episodic nodes
	if len(episodicNodes) > 0 {
		for _, node := range episodicNodes {
			if err := driver.UpsertNode(ctx, node); err != nil {
				result.Errors = append(result.Errors, fmt.Errorf("failed to upsert episodic node %s: %w", node.Uuid, err))
			} else {
				result.EpisodicNodes = append(result.EpisodicNodes, node)
			}
		}
	}

	// Add entity nodes with embeddings
	if len(entityNodes) > 0 {
		// Generate embeddings for entity nodes if needed
		var textsToEmbed []string
		var nodeIndices []int
		for i, node := range entityNodes {
			if len(node.Embedding) == 0 && node.Name != "" {
				textsToEmbed = append(textsToEmbed, node.Name)
				nodeIndices = append(nodeIndices, i)
			}
		}

		if len(textsToEmbed) > 0 && embedder != nil {
			embeddings, err := embedder.Embed(ctx, textsToEmbed)
			if err != nil {
				result.Errors = append(result.Errors, fmt.Errorf("failed to generate embeddings: %w", err))
			} else {
				for i, embedding := range embeddings {
					if i < len(nodeIndices) {
						entityNodes[nodeIndices[i]].Embedding = embedding
					}
				}
			}
		}

		// Upsert entity nodes
		for _, node := range entityNodes {
			if err := driver.UpsertNode(ctx, node); err != nil {
				result.Errors = append(result.Errors, fmt.Errorf("failed to upsert entity node %s: %w", node.Uuid, err))
			} else {
				result.EntityNodes = append(result.EntityNodes, node)
			}
		}
	}

	// Add episodic edges
	if len(episodicEdges) > 0 {
		for _, edge := range episodicEdges {
			if err := driver.UpsertEdge(ctx, edge); err != nil {
				result.Errors = append(result.Errors, fmt.Errorf("failed to upsert episodic edge %s: %w", edge.Uuid, err))
			} else {
				result.EpisodicEdges = append(result.EpisodicEdges, edge)
			}
		}
	}

	// Add entity edges with embeddings
	if len(entityEdges) > 0 {
		// Generate embeddings for entity edges if needed
		var textsToEmbed []string
		var edgeIndices []int
		for i, edge := range entityEdges {
			if len(edge.Embedding) == 0 && edge.Summary != "" {
				textsToEmbed = append(textsToEmbed, edge.Summary)
				edgeIndices = append(edgeIndices, i)
			}
		}

		if len(textsToEmbed) > 0 && embedder != nil {
			embeddings, err := embedder.Embed(ctx, textsToEmbed)
			if err != nil {
				result.Errors = append(result.Errors, fmt.Errorf("failed to generate edge embeddings: %w", err))
			} else {
				for i, embedding := range embeddings {
					if i < len(edgeIndices) {
						entityEdges[edgeIndices[i]].Embedding = embedding
					}
				}
			}
		}

		// Upsert entity edges
		for _, edge := range entityEdges {
			if err := driver.UpsertEdge(ctx, edge); err != nil {
				result.Errors = append(result.Errors, fmt.Errorf("failed to upsert entity edge %s: %w", edge.Uuid, err))
			} else {
				result.EntityEdges = append(result.EntityEdges, edge)
			}
		}
	}

	// Report database statistics for all affected groups
	logger := slog.Default()

	// Collect unique group IDs from all added nodes and edges
	groupIDs := make(map[string]bool)
	for _, node := range episodicNodes {
		if node.GroupID != "" {
			groupIDs[node.GroupID] = true
		}
	}
	for _, node := range entityNodes {
		if node.GroupID != "" {
			groupIDs[node.GroupID] = true
		}
	}
	for _, edge := range episodicEdges {
		if edge.GroupID != "" {
			groupIDs[edge.GroupID] = true
		}
	}
	for _, edge := range entityEdges {
		if edge.GroupID != "" {
			groupIDs[edge.GroupID] = true
		}
	}

	// Query and log statistics for each affected group
	for groupID := range groupIDs {
		stats, err := driver.GetStats(ctx, groupID)
		if err != nil {
			logger.Warn("Failed to retrieve graph statistics",
				"group_id", groupID,
				"error", err.Error())
			continue
		}

		// Extract counts by type
		episodicNodeCount := int64(0)
		entityNodeCount := int64(0)
		relatesToEdgeCount := int64(0)
		mentionsEdgeCount := int64(0)
		hasMemberEdgeCount := int64(0)

		if stats.NodesByType != nil {
			episodicNodeCount = stats.NodesByType["Episodic"]
			entityNodeCount = stats.NodesByType["Entity"]
		}

		if stats.EdgesByType != nil {
			relatesToEdgeCount = stats.EdgesByType["RELATES_TO"]
			mentionsEdgeCount = stats.EdgesByType["MENTIONS"]
			hasMemberEdgeCount = stats.EdgesByType["HAS_MEMBER"]
		}

		logger.Info("Graph database statistics after bulk operation",
			"group_id", groupID,
			"total_nodes", stats.NodeCount,
			"total_edges", stats.EdgeCount,
			"episodic_nodes", episodicNodeCount,
			"entity_nodes", entityNodeCount,
			"relates_to_edges", relatesToEdgeCount,
			"mentions_edges", mentionsEdgeCount,
			"has_member_edges", hasMemberEdgeCount,
			"communities", stats.CommunityCount,
		)
	}

	return result, nil
}

// ExtractNodesAndEdgesBulk extracts nodes and edges from episodes in bulk
// This matches the Python function signature: extract_nodes_and_edges_bulk(clients, episode_tuples, edge_type_map, ...)
func ExtractNodesAndEdgesBulk(
	ctx context.Context,
	clients *Clients,
	episodeTuples []EpisodeTuple,
	edgeTypeMap map[string]string,
	entityTypes []string,
	excludedEntityTypes []string,
	edgeTypes []string,
	logger *slog.Logger,
) (*ExtractNodesAndEdgesResult, error) {
	var allExtractedNodes []*types.Node
	var allExtractedEdges []*types.Edge

	// Process episodes in batches for better performance
	batchProcessor := NewBatchProcessor(
		GetSemaphoreLimit(),
		GetSemaphoreLimit(),
		func(ctx context.Context, batch []EpisodeTuple) ([]*ExtractNodesAndEdgesResult, error) {
			var results []*ExtractNodesAndEdgesResult
			for _, episodeTuple := range batch {
				result, err := extractFromSingleEpisode(ctx, clients, episodeTuple, edgeTypeMap, entityTypes, excludedEntityTypes, edgeTypes, logger)
				if err != nil {
					return nil, err
				}
				results = append(results, result)
			}
			return results, nil
		},
	)

	batchResults, err := batchProcessor.Process(ctx, episodeTuples)
	if err != nil {
		return nil, fmt.Errorf("failed to process episode batches: %w", err)
	}

	// Aggregate results
	for _, result := range batchResults {
		allExtractedNodes = append(allExtractedNodes, result.ExtractedNodes...)
		allExtractedEdges = append(allExtractedEdges, result.ExtractedEdges...)
	}

	return &ExtractNodesAndEdgesResult{
		ExtractedNodes: allExtractedNodes,
		ExtractedEdges: allExtractedEdges,
	}, nil
}

// extractFromSingleEpisode extracts nodes and edges from a single episode
func extractFromSingleEpisode(
	ctx context.Context,
	clients *Clients,
	episodeTuple EpisodeTuple,
	edgeTypeMap map[string]string,
	entityTypes []string,
	excludedEntityTypes []string,
	edgeTypes []string,
	logger *slog.Logger,
) (*ExtractNodesAndEdgesResult, error) {
	// This is a simplified implementation - in practice this would use
	// the LLM client and prompts library to extract entities and relationships

	// Create context from episode and previous episodes
	content := episodeTuple.Episode.Content
	for _, prevEpisode := range episodeTuple.PreviousEpisodes {
		content += "\nPrevious: " + prevEpisode.Content
	}

	// Use LLM to extract entities (simplified)
	// Build context for the prompt
	promptContext := map[string]interface{}{
		"episode_content":       content,
		"entity_types":          entityTypes,
		"excluded_entity_types": excludedEntityTypes,
		"previous_episodes":     episodeTuple.PreviousEpisodes,
	}

	entityMessages, err := clients.Prompts.ExtractNodes().ExtractMessage().Call(promptContext)
	if err != nil {
		return nil, fmt.Errorf("failed to create entity extraction prompt: %w", err)
	}

	entityResponse, err := clients.NLP.Chat(ctx, entityMessages)
	prompts.LogResponses(logger, *entityResponse)
	if err != nil {
		return nil, fmt.Errorf("failed to extract entities: %w", err)
	}

	// Parse entities from response (this would need proper JSON parsing)
	var extractedNodes []*types.Node
	// Simplified parsing - in practice this would parse JSON response
	entities := strings.Split(entityResponse.Content, ",")
	for i, entity := range entities {
		entity = strings.TrimSpace(entity)
		if entity != "" {
			node := &types.Node{
				Uuid:      fmt.Sprintf("entity_%d_%s", i, episodeTuple.Episode.ID),
				Name:      entity,
				Type:      types.EntityNodeType,
				GroupID:   episodeTuple.Episode.GroupID,
				CreatedAt: episodeTuple.Episode.CreatedAt,
				Summary:   entity,
			}
			extractedNodes = append(extractedNodes, node)
		}
	}

	// Use LLM to extract relationships (simplified)
	edgeContext := map[string]interface{}{
		"episode_content": content,
		"extracted_nodes": extractedNodes,
		"edge_types":      edgeTypes,
	}

	edgeMessages, err := clients.Prompts.ExtractEdges().Edge().Call(edgeContext)
	if err != nil {
		return nil, fmt.Errorf("failed to create edge extraction prompt: %w", err)
	}

	edgeResponse, err := clients.NLP.Chat(ctx, edgeMessages)
	prompts.LogResponses(logger, *edgeResponse)
	if err != nil {
		return nil, fmt.Errorf("failed to extract edges: %w \nprompt: %s \nresponse: \n %s", err, edgeMessages[1].Content, edgeResponse.Content)
	}

	// Parse edges from response (simplified)
	var extractedEdges []*types.Edge
	// This would need proper JSON parsing in practice
	relationships := strings.Split(edgeResponse.Content, ";")
	for i, rel := range relationships {
		rel = strings.TrimSpace(rel)
		if rel != "" && len(extractedNodes) >= 2 {
			edge := types.NewEntityEdge(
				fmt.Sprintf("edge_%d_%s", i, episodeTuple.Episode.ID),
				extractedNodes[0].Uuid,
				extractedNodes[min(1, len(extractedNodes)-1)].Uuid,
				episodeTuple.Episode.GroupID,
				rel,
				types.EntityEdgeType,
			)
			edge.CreatedAt = episodeTuple.Episode.CreatedAt
			extractedEdges = append(extractedEdges, edge)
		}
	}

	return &ExtractNodesAndEdgesResult{
		ExtractedNodes: extractedNodes,
		ExtractedEdges: extractedEdges,
	}, nil
}

// DedupeNodesBulk resolves entity duplicates across an in-memory batch using a two-pass strategy.
//
// Pass 1: Run ResolveExtractedNodes for every episode in parallel so each batch item is
// reconciled against the live graph just like the non-batch flow.
//
// Pass 2: Re-run the deterministic similarity heuristics across the union of resolved nodes
// to catch duplicates that only co-occur inside this batch, emitting a canonical UUID map
// that callers can apply to edges and persistence.
//
// This matches the Python function: dedupe_nodes_bulk(clients, extracted_nodes, episode_tuples, entity_types)
func DedupeNodesBulk(
	ctx context.Context,
	clients *Clients,
	extractedNodesByEpisode [][]*types.Node, // List of lists - one list per episode
	episodeTuples []EpisodeTuple,
	entityTypes map[string]interface{},
	nodeOps NodeOperations,
) (*DedupeNodesResult, error) {
	if len(extractedNodesByEpisode) == 0 {
		return &DedupeNodesResult{
			NodesByEpisode: make(map[string][]*types.Node),
			UUIDMap:        make(map[string]string),
		}, nil
	}

	// PASS 1: Resolve each episode's nodes against the live graph in parallel
	type firstPassResult struct {
		duplicatePairs interface{} // Can be []NodePair from maintenance or any compatible type
		uuidMap        map[string]string
		resolvedNodes  []*types.Node
	}

	firstPassResults := make([]firstPassResult, len(extractedNodesByEpisode))
	functions := make([]func() (firstPassResult, error), len(extractedNodesByEpisode))

	for i, nodes := range extractedNodesByEpisode {
		idx := i
		// Make a proper copy of the slice to avoid sharing references across closures
		nodesCopy := make([]*types.Node, len(nodes))
		copy(nodesCopy, nodes)

		// Capture episode data at closure creation time, not execution time
		episodeCopy := episodeTuples[idx].Episode
		prevEpisodesCopy := episodeTuples[idx].PreviousEpisodes

		functions[idx] = func() (firstPassResult, error) {
			if idx >= len(episodeTuples) {
				return firstPassResult{}, fmt.Errorf("episode tuple index %d out of range", idx)
			}

			episode := episodeCopy
			previousEpisodes := prevEpisodesCopy

			// Convert Episode to Node for compatibility
			episodeNode := &types.Node{
				Uuid:      episode.ID,
				Name:      episode.Name,
				Type:      types.EpisodicNodeType,
				Content:   episode.Content,
				GroupID:   episode.GroupID,
				CreatedAt: episode.CreatedAt,
				ValidFrom: episode.Reference,
			}

			// Convert Episode slice to Node slice for previous episodes
			var prevEpisodeNodes []*types.Node
			for _, prevEp := range previousEpisodes {
				prevEpisodeNodes = append(prevEpisodeNodes, &types.Node{
					Uuid:      prevEp.ID,
					Name:      prevEp.Name,
					Type:      types.EpisodicNodeType,
					Content:   prevEp.Content,
					GroupID:   prevEp.GroupID,
					CreatedAt: prevEp.CreatedAt,
					ValidFrom: prevEp.Reference,
				})
			}

			// Resolve against live graph
			resolved, uuidMap, duplicates, err := nodeOps.ResolveExtractedNodes(
				ctx, nodesCopy, episodeNode, prevEpisodeNodes, entityTypes,
			)
			if err != nil {
				return firstPassResult{}, fmt.Errorf("failed to resolve episode %d nodes: %w", idx, err)
			}

			return firstPassResult{
				resolvedNodes:  resolved,
				uuidMap:        uuidMap,
				duplicatePairs: duplicates,
			}, nil
		}
	}

	// Execute first pass in parallel
	results, errors := SemaphoreGatherWithResults(ctx, GetSemaphoreLimit(), functions...)
	for i, err := range errors {
		if err != nil {
			return nil, fmt.Errorf("first pass failed for episode %d: %w", i, err)
		}
		firstPassResults[i] = results[i]
	}

	// Collect episode resolutions from first pass
	// Note: duplicate pairs from Pass 1 are already captured in the UUID maps,
	// so we don't need to extract them separately
	episodeResolutions := make([]struct {
		episodeUUID   string
		resolvedNodes []*types.Node
	}, len(episodeTuples))

	perEpisodeUUIDMaps := make([]map[string]string, len(firstPassResults))

	for i, result := range firstPassResults {
		episodeResolutions[i].episodeUUID = episodeTuples[i].Episode.ID
		episodeResolutions[i].resolvedNodes = result.resolvedNodes
		perEpisodeUUIDMaps[i] = result.uuidMap
	}

	// PASS 2: Re-run deduplication across the union of resolved nodes to catch
	// duplicates that only co-occur inside this batch
	canonicalNodes := make(map[string]*types.Node)
	var duplicatePairsFromPass2 [][2]string

	for _, resolution := range episodeResolutions {
		for _, node := range resolution.resolvedNodes {
			// This loop is O(n^2) but caching keeps it manageable for typical batch sizes
			if len(canonicalNodes) == 0 {
				canonicalNodes[node.Uuid] = node
				continue
			}

			// Build indexes for existing canonical nodes
			existingCandidates := make([]*types.Node, 0, len(canonicalNodes))
			for _, n := range canonicalNodes {
				existingCandidates = append(existingCandidates, n)
			}

			// Check for exact name match
			normalized := NormalizeStringExact(node.Name)
			var exactMatch *types.Node
			for _, candidate := range existingCandidates {
				if NormalizeStringExact(candidate.Name) == normalized {
					exactMatch = candidate
					break
				}
			}

			if exactMatch != nil {
				if exactMatch.Uuid != node.Uuid {
					duplicatePairsFromPass2 = append(duplicatePairsFromPass2, [2]string{node.Uuid, exactMatch.Uuid})
				}
				continue
			}

			// Try fuzzy matching using similarity heuristics
			indexes := BuildCandidateIndexes(existingCandidates)
			state := &DedupResolutionState{
				ResolvedNodes:     []*types.Node{nil},
				UUIDMap:           make(map[string]string),
				UnresolvedIndices: []int{},
				DuplicatePairs:    []NodePair{},
			}

			ResolveWithSimilarity([]*types.Node{node}, indexes, state)

			resolved := state.ResolvedNodes[0]
			if resolved == nil {
				// No match found - add as new canonical node
				canonicalNodes[node.Uuid] = node
				continue
			}

			// Found a match
			canonicalUUID := resolved.Uuid
			if _, exists := canonicalNodes[canonicalUUID]; !exists {
				canonicalNodes[canonicalUUID] = resolved
			}
			if canonicalUUID != node.Uuid {
				duplicatePairsFromPass2 = append(duplicatePairsFromPass2, [2]string{node.Uuid, canonicalUUID})
			}
		}
	}

	// Combine UUID maps from both passes using directed union-find
	// Pass 1 duplicate information is already in perEpisodeUUIDMaps
	var unionPairs [][2]string
	for _, uuidMap := range perEpisodeUUIDMaps {
		for oldUUID, newUUID := range uuidMap {
			if oldUUID != newUUID {
				unionPairs = append(unionPairs, [2]string{oldUUID, newUUID})
			}
		}
	}
	unionPairs = append(unionPairs, duplicatePairsFromPass2...)

	compressedMap := BuildDirectedUUIDMap(unionPairs)

	// Group nodes by episode with canonical UUIDs
	nodesByEpisode := make(map[string][]*types.Node)
	for _, resolution := range episodeResolutions {
		// Pre-allocate with capacity hint from resolved nodes
		dedupedNodes := make([]*types.Node, 0, len(resolution.resolvedNodes))
		seen := make(map[string]bool, len(resolution.resolvedNodes))

		for _, node := range resolution.resolvedNodes {
			canonicalUUID := compressedMap[node.Uuid]
			if canonicalUUID == "" {
				canonicalUUID = node.Uuid
			}

			if seen[canonicalUUID] {
				continue
			}
			seen[canonicalUUID] = true

			canonicalNode := canonicalNodes[canonicalUUID]
			if canonicalNode == nil {
				// Fallback to original node if canonical not found
				canonicalNode = node
			}
			dedupedNodes = append(dedupedNodes, canonicalNode)
		}

		nodesByEpisode[resolution.episodeUUID] = dedupedNodes
	}

	return &DedupeNodesResult{
		NodesByEpisode: nodesByEpisode,
		UUIDMap:        compressedMap,
	}, nil
}

// DedupeEdgesBulk deduplicates extracted edges across episodes
// This matches the Python function signature: dedupe_edges_bulk(clients, extracted_edges, episode_tuples, ...)
func DedupeEdgesBulk(
	ctx context.Context,
	clients *Clients,
	extractedEdges []*types.Edge,
	episodeTuples []EpisodeTuple,
	embedder embedder.Client,
	logger *slog.Logger,
) (*DedupeEdgesResult, error) {
	if len(extractedEdges) == 0 {
		return &DedupeEdgesResult{
			EdgesByEpisode: make(map[string][]*types.Edge),
			UUIDMap:        make(map[string]string),
		}, nil
	}

	// Generate embeddings for edges if not present
	var edgesToEmbed []*types.Edge
	var textsToEmbed []string
	for _, edge := range extractedEdges {
		if len(edge.Embedding) == 0 && edge.Summary != "" {
			edgesToEmbed = append(edgesToEmbed, edge)
			textsToEmbed = append(textsToEmbed, edge.Summary)
		}
	}

	if len(textsToEmbed) > 0 && embedder != nil {
		embeddings, err := embedder.Embed(ctx, textsToEmbed)
		if err != nil {
			return nil, fmt.Errorf("failed to generate embeddings for edge deduplication: %w", err)
		}
		for i, embedding := range embeddings {
			if i < len(edgesToEmbed) {
				edgesToEmbed[i].Embedding = embedding
			}
		}
	}

	// Find duplicates using similarity comparison
	var duplicatePairs [][]string
	processed := make(map[string]bool)

	for i, edge1 := range extractedEdges {
		if processed[edge1.Uuid] {
			continue
		}

		similar := FindSimilarEdges(edge1, extractedEdges[i+1:], MinScoreEdges)
		if len(similar) > 0 {
			for _, edge2 := range similar {
				if !processed[edge2.Uuid] {
					duplicatePairs = append(duplicatePairs, []string{edge1.Uuid, edge2.Uuid})
					processed[edge2.Uuid] = true
				}
			}
		}
		processed[edge1.Uuid] = true
	}

	// Use LLM to confirm duplicates (simplified)
	if len(duplicatePairs) > 0 && clients != nil && clients.NLP != nil {
		confirmedPairs := make([][]string, 0, len(duplicatePairs))

		for _, pair := range duplicatePairs {
			// Find the actual edges
			var edge1, edge2 *types.Edge
			for _, edge := range extractedEdges {
				if edge.Uuid == pair[0] {
					edge1 = edge
				} else if edge.Uuid == pair[1] {
					edge2 = edge
				}
			}

			if edge1 != nil && edge2 != nil {
				// Use LLM to confirm if they are duplicates
				dedupeContext := map[string]interface{}{
					"edges": []*types.Edge{edge1, edge2},
				}

				dedupeMessages, err := clients.Prompts.DedupeEdges().Edge().Call(dedupeContext)
				if err == nil {
					response, err := clients.NLP.Chat(ctx, dedupeMessages)
					prompts.LogResponses(logger, *response)
					if err == nil && strings.Contains(strings.ToLower(response.Content), "duplicate") {
						confirmedPairs = append(confirmedPairs, pair)
					}
				}
			}
		}
		duplicatePairs = confirmedPairs
	}

	// Create UUID mapping using UnionFind
	uuidMap := CompressUUIDMap(duplicatePairs)

	// Group edges by episode
	edgesByEpisode := make(map[string][]*types.Edge)
	edgeMap := make(map[string]*types.Edge)

	// Create edge map and apply UUID mappings
	for _, edge := range extractedEdges {
		canonicalID := uuidMap[edge.Uuid]
		if canonicalID == "" {
			canonicalID = edge.Uuid
		}

		// Use the canonical edge (lexicographically smallest ID)
		if existingEdge, exists := edgeMap[canonicalID]; exists {
			// Merge properties if needed (simplified)
			if existingEdge.Summary == "" && edge.Summary != "" {
				existingEdge.Summary = edge.Summary
			}
			if existingEdge.Name == "" && edge.Name != "" {
				existingEdge.Name = edge.Name
			}
		} else {
			// Create a copy with canonical ID
			canonicalEdge := *edge
			canonicalEdge.Uuid = canonicalID
			edgeMap[canonicalID] = &canonicalEdge
		}
	}

	// Group edges by their source episodes
	for _, episodeTuple := range episodeTuples {
		var episodeEdges []*types.Edge
		seen := make(map[string]bool)

		// Find edges that came from this episode
		for _, edge := range extractedEdges {
			// This is simplified - in practice you'd track which episode each edge came from
			if strings.Contains(edge.Uuid, episodeTuple.Episode.ID) {
				canonicalID := uuidMap[edge.Uuid]
				if canonicalID == "" {
					canonicalID = edge.Uuid
				}

				if !seen[canonicalID] && edgeMap[canonicalID] != nil {
					episodeEdges = append(episodeEdges, edgeMap[canonicalID])
					seen[canonicalID] = true
				}
			}
		}

		edgesByEpisode[episodeTuple.Episode.ID] = episodeEdges
	}

	return &DedupeEdgesResult{
		EdgesByEpisode: edgesByEpisode,
		UUIDMap:        uuidMap,
	}, nil
}

// Helper function to get minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
