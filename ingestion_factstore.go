package predicato

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/soundprediction/predicato/pkg/factstore"
	"github.com/soundprediction/predicato/pkg/modeler"
	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/prompts"
	"github.com/soundprediction/predicato/pkg/ruleschema"
	"github.com/soundprediction/predicato/pkg/types"
	"github.com/soundprediction/predicato/pkg/utils"
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

	// 4b. Generate Node Embeddings
	if c.embedder != nil {
		for _, nodes := range extractedNodesByChunk {
			if len(nodes) == 0 {
				continue
			}
			texts := make([]string, len(nodes))
			for i, n := range nodes {
				if n.Summary != "" {
					texts[i] = n.Name + " " + n.Summary
				} else {
					texts[i] = n.Name
				}
			}
			embeddings, err := c.embedder.Embed(ctx, texts)
			if err != nil {
				c.logger.Warn("Failed to generate node embeddings, continuing without",
					"error", err,
					"episode_id", episode.ID)
			} else {
				for i, emb := range embeddings {
					nodes[i].Embedding = emb
				}
			}
		}
	} else {
		c.logger.Warn("No embedder configured, skipping node embedding generation",
			"episode_id", episode.ID)
	}

	// 5. Prepare Facts Data (Nodes)
	var flattenedNodes []*types.Node
	var factsNodes []*factstore.ExtractedNode

	for chunkIdx, nodes := range extractedNodesByChunk {
		for _, n := range nodes {
			flattenedNodes = append(flattenedNodes, n)
			nodeModel := ""
			if c.nlpModels.NodeExtraction != nil {
				nodeModel = c.nlpModels.NodeExtraction.GetModel()
			} else if c.nlProcessor != nil {
				nodeModel = c.nlProcessor.GetModel()
			}
			factsNodes = append(factsNodes, &factstore.ExtractedNode{
				ID:             n.Uuid,
				SourceID:       episode.ID,
				Name:           n.Name,
				NormalizedName: utils.NormalizeStringExact(n.Name),
				Type:           string(n.Type),
				Description:    n.Summary,
				Model:          nodeModel,
				Embedding:      n.Embedding,
				ChunkIndex:     chunkIdx,
			})
		}
	}

	// Verbose logging for extracted entities
	if options.Verbose {
		c.logger.Info("[VERBOSE] Extracted entities",
			"episode_id", episode.ID,
			"entity_count", len(factsNodes))
		for _, n := range factsNodes {
			c.logger.Info("[VERBOSE] Entity",
				"name", n.Name,
				"type", n.Type,
				"description", n.Description,
				"chunk", n.ChunkIndex)
		}
	}

	// 6. Extract Relationships (Raw) and Prepare Knowledge Triples
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

	var knowledgeTriples []*factstore.ExtractedTriple

	edgeModel := ""
	if c.nlpModels.EdgeExtraction != nil {
		edgeModel = c.nlpModels.EdgeExtraction.GetModel()
	} else if c.nlProcessor != nil {
		edgeModel = c.nlProcessor.GetModel()
	}

	for chunkIdx, nodes := range extractedNodesByChunk {
		if len(nodes) > 0 {
			// Extract edges using the chunk's context
			extracted, err := edgeOps.ExtractEdges(ctx, chunkData.chunkEpisodeNodes[chunkIdx], nodes, previousEpisodes, edgeTypeMap, options.EdgeTypes, episode.GroupID)
			if err != nil {
				return nil, err
			}

			for _, e := range extracted {
				// Resolve Names for Facts DB
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

				// If still not found, check previousEpisodes
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

				knowledgeTriples = append(knowledgeTriples, &factstore.ExtractedTriple{
					ID:          e.Uuid,
					SourceID:    episode.ID,
					Subject:     utils.NormalizeStringExact(sourceName),
					SubjectType: sourceType,
					Predicate:   e.Name,
					Object:      utils.NormalizeStringExact(targetName),
					ObjectType:  targetType,
					Description: e.Summary,
					Model:       edgeModel,
					Confidence:  e.Strength,
					ChunkIndex:  chunkIdx,
				})
			}
		}
	}

	// 6b. Generate Triple Embeddings
	if c.embedder != nil && len(knowledgeTriples) > 0 {
		texts := make([]string, len(knowledgeTriples))
		for i, t := range knowledgeTriples {
			texts[i] = t.Subject + " " + t.Predicate + " " + t.Object
			if t.Description != "" {
				texts[i] = t.Description
			}
		}
		embeddings, err := c.embedder.Embed(ctx, texts)
		if err != nil {
			c.logger.Warn("Failed to generate triple embeddings, continuing without",
				"error", err,
				"episode_id", episode.ID)
		} else {
			for i, emb := range embeddings {
				knowledgeTriples[i].Embedding = emb
			}
		}
	} else if c.embedder == nil && len(knowledgeTriples) > 0 {
		c.logger.Warn("No embedder configured, skipping triple embedding generation",
			"episode_id", episode.ID)
	}

	// Verbose logging for extracted relations
	if options.Verbose {
		c.logger.Info("[VERBOSE] Extracted relations",
			"episode_id", episode.ID,
			"relation_count", len(knowledgeTriples))
		for _, t := range knowledgeTriples {
			c.logger.Info("[VERBOSE] Relation",
				"subject", t.Subject,
				"predicate", t.Predicate,
				"object", t.Object,
				"description", t.Description,
				"chunk", t.ChunkIndex)
		}
	}

	// 6c. Extended Extraction (enriches triples with context, extracts rules) — optional
	var knowledgeRules []*factstore.ExtractedRule

	if options.ExtendedExtraction {
		extractionClient := c.nlpModels.NodeExtraction
		if extractionClient == nil {
			extractionClient = c.nlProcessor
		}

		// Check if the extraction client supports extended extraction
		if !nlp.HasCapability(extractionClient, nlp.TaskExtendedExtraction) {
			c.logger.Warn("Extended extraction enabled but the configured extraction model does not support it — falling back to regular extraction pipeline",
				"model", extractionClient.GetModel(),
				"episode_id", episode.ID)
		} else {

			// Collect entity/relation type keys from config for schema guidance
			var entityTypeKeys, relationTypeKeys []string
			if options.EntityTypes != nil {
				for k := range options.EntityTypes {
					entityTypeKeys = append(entityTypeKeys, k)
				}
			}
			if options.EdgeTypes != nil {
				for k := range options.EdgeTypes {
					relationTypeKeys = append(relationTypeKeys, k)
				}
			}

			for chunkIdx, chunk := range chunks {
				result, err := extractionClient.ExtractExtended(ctx, chunk, entityTypeKeys, relationTypeKeys)
				if err != nil {
					c.logger.Warn("Extended extraction failed for chunk, skipping",
						"episode_id", episode.ID,
						"chunk", chunkIdx,
						"error", err)
					continue
				}

				extModel := extractionClient.GetModel()

				// Convert nlp.ExtendedTriple → types.ExtractedTriple
				for _, t := range result.Triples {
					knowledgeTriples = append(knowledgeTriples, &types.ExtractedTriple{
						ID:                fmt.Sprintf("%s-triple-%d-%d", episode.ID, chunkIdx, len(knowledgeTriples)),
						SourceID:          episode.ID,
						Subject:           t.Subject,
						Predicate:         t.Predicate,
						Object:            t.Object,
						Condition:         t.Condition,
						Temporal:          t.Temporal,
						Location:          t.Location,
						Certainty:         t.Certainty,
						Scope:             t.Scope,
						SourceAttribution: t.SourceAttribution,
						Confidence:        t.Confidence,
						Model:             extModel,
						ChunkIndex:        chunkIdx,
						CreatedAt:         time.Now(),
					})
				}

				// Convert nlp.Rule → types.ExtractedRule
				for _, r := range result.Rules {
					ruleID := fmt.Sprintf("%s-rule-%d-%d", episode.ID, chunkIdx, len(knowledgeRules))
					nlp.CompileRuleStructure(&r, nlp.RuleStructureMetadata{
						ID:            ruleID,
						SourceID:      episode.ID,
						Model:         extModel,
						ChunkIndex:    chunkIdx,
						HasChunkIndex: true,
					})
					knowledgeRules = append(knowledgeRules, &types.ExtractedRule{
						ID:                ruleID,
						SourceID:          episode.ID,
						Antecedent:        r.Antecedent,
						Consequent:        r.Consequent,
						Exception:         r.Exception,
						RuleType:          r.RuleType,
						Scope:             r.Scope,
						SourceAttribution: r.SourceAttribution,
						Confidence:        r.Confidence,
						Model:             extModel,
						ChunkIndex:        chunkIdx,
						CreatedAt:         time.Now(),
						Structured:        r.Structured,
						StructureStatus:   r.StructureStatus,
						StructureError:    r.StructureError,
					})
				}
			}

			// Generate embeddings for extended triples (only the newly added ones)
			if c.embedder != nil && len(knowledgeTriples) > 0 {
				// Find triples that don't have embeddings yet (from extended extraction)
				var needsEmbedding []int
				for i, t := range knowledgeTriples {
					if len(t.Embedding) == 0 {
						needsEmbedding = append(needsEmbedding, i)
					}
				}
				if len(needsEmbedding) > 0 {
					texts := make([]string, len(needsEmbedding))
					for j, idx := range needsEmbedding {
						t := knowledgeTriples[idx]
						texts[j] = t.Subject + " " + t.Predicate + " " + t.Object
					}
					embeddings, err := c.embedder.Embed(ctx, texts)
					if err != nil {
						c.logger.Warn("Failed to generate extended triple embeddings, continuing without",
							"error", err, "episode_id", episode.ID)
					} else {
						for j, emb := range embeddings {
							knowledgeTriples[needsEmbedding[j]].Embedding = emb
						}
					}
				}
			}

			// Generate embeddings for rules
			if c.embedder != nil && len(knowledgeRules) > 0 {
				texts := make([]string, len(knowledgeRules))
				for i, r := range knowledgeRules {
					texts[i] = r.Antecedent + " " + r.Consequent
				}
				embeddings, err := c.embedder.Embed(ctx, texts)
				if err != nil {
					c.logger.Warn("Failed to generate rule embeddings, continuing without",
						"error", err, "episode_id", episode.ID)
				} else {
					for i, emb := range embeddings {
						knowledgeRules[i].Embedding = emb
					}
				}
			}

			if options.Verbose {
				c.logger.Info("[VERBOSE] Extended extraction complete",
					"episode_id", episode.ID,
					"triple_count", len(knowledgeTriples),
					"rule_count", len(knowledgeRules))
			}
		} // end capability check else
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

	// Verbose logging for source being saved
	if options.Verbose {
		c.logger.Info("[VERBOSE] Saving source to factstore",
			"source_id", source.ID,
			"source_name", source.Name,
			"content_length", len(source.Content),
			"group_id", source.GroupID)
	}

	if err := c.factStore.SaveSource(ctx, source); err != nil {
		return nil, err
	}

	if err := c.factStore.SaveExtractedKnowledge(ctx, episode.ID, factsNodes, knowledgeTriples); err != nil {
		return nil, err
	}

	// Save rules
	if len(knowledgeRules) > 0 {
		if err := c.factStore.SaveExtractedRules(ctx, episode.ID, knowledgeRules); err != nil {
			c.logger.Warn("Failed to save extracted rules, continuing",
				"error", err, "episode_id", episode.ID)
		}
	}

	// 8. Return ExtractionResults
	return &types.ExtractionResults{
		SourceID:         episode.ID,
		ExtractedNodes:   factsNodes,
		ExtractedTriples: knowledgeTriples,
		ExtractedRules:   knowledgeRules,
		ChunkCount:       len(chunks),
		ExtractionTime:   time.Since(startTime),
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

	extTriples, err := c.factStore.GetExtractedTriples(ctx, sourceID)
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

	// 4. Filter low-value triples before graph promotion
	extTriples, removedTriples := types.FilterLowValueTriples(extTriples)
	if len(removedTriples) > 0 {
		c.logger.Info("Filtered low-value triples during promotion",
			"source_id", sourceID,
			"kept", len(extTriples),
			"removed", len(removedTriples))
	}

	// 5. Reconstruct Edges from Triples
	var allExtractedEdges []*types.Edge
	for _, t := range extTriples {
		var sUUID, tUUID string
		for _, n := range uuidToNode {
			if n.Name == t.Subject {
				sUUID = n.Uuid
			}
			if n.Name == t.Object {
				tUUID = n.Uuid
			}
		}

		if sUUID != "" && tUUID != "" {
			te := &types.Edge{
				BaseEdge: types.BaseEdge{
					Uuid:         t.ID,
					SourceNodeID: sUUID,
					TargetNodeID: tUUID,
					GroupID:      source.GroupID,
					CreatedAt:    time.Now(),
				},
				Name:     t.Predicate,
				Summary:  t.Description,
				Fact:     t.Description,
				Strength: t.Confidence,
				Type:     types.EntityEdgeType,
				SourceID: sUUID,
				TargetID: tUUID,
			}
			allExtractedEdges = append(allExtractedEdges, te)
		}
	}

	// 5.5: Load and promote extracted rules
	extRules, err := c.factStore.GetExtractedRules(ctx, sourceID)
	if err != nil {
		c.logger.Warn("Failed to load extracted rules", "error", err)
		extRules = nil // Non-fatal: proceed without rules
	}

	var ruleNodes []*types.Node
	var ruleEdges []*types.Edge

	for _, rule := range extRules {
		ruleNode := &types.Node{
			Uuid:       rule.ID,
			Name:       buildRuleName(rule),
			Summary:    buildRuleSummary(rule),
			Content:    buildRuleContent(rule),
			Type:       types.RuleNodeType,
			EntityType: rule.RuleType,
			Embedding:  rule.Embedding,
			GroupID:    source.GroupID,
			SourceIDs:  []string{source.ID},
			Metadata:   buildRuleMetadata(rule, source.ID),
			CreatedAt:  time.Now(),
			UpdatedAt:  time.Now(),
			ValidFrom:  time.Now(),
		}
		ruleNodes = append(ruleNodes, ruleNode)

		// Entity linking (approach B) — default
		if !options.SkipRuleEntityLinking {
			antEntities := matchEntitiesInText(rule.Antecedent, uuidToNode)
			conEntities := matchEntitiesInText(rule.Consequent, uuidToNode)
			excEntities := matchEntitiesInText(rule.Exception, uuidToNode)

			// REFERENCED_BY edges: rule → entity
			for _, e := range antEntities {
				edge := types.NewEntityEdge(utils.GenerateUUID(), ruleNode.Uuid, e.Uuid, source.GroupID, "REFERENCED_BY", types.RuleEdgeType)
				edge.Metadata = map[string]interface{}{"part": "antecedent"}
				edge.Fact = rule.Antecedent
				ruleEdges = append(ruleEdges, edge)
			}
			for _, e := range conEntities {
				edge := types.NewEntityEdge(utils.GenerateUUID(), ruleNode.Uuid, e.Uuid, source.GroupID, "REFERENCED_BY", types.RuleEdgeType)
				edge.Metadata = map[string]interface{}{"part": "consequent"}
				edge.Fact = rule.Consequent
				ruleEdges = append(ruleEdges, edge)
			}
			for _, e := range excEntities {
				edge := types.NewEntityEdge(utils.GenerateUUID(), ruleNode.Uuid, e.Uuid, source.GroupID, "REFERENCED_BY", types.RuleEdgeType)
				edge.Metadata = map[string]interface{}{"part": "exception"}
				edge.Fact = rule.Exception
				ruleEdges = append(ruleEdges, edge)
			}

			// IMPLIES edges: antecedent entity → consequent entity
			for _, ae := range antEntities {
				for _, ce := range conEntities {
					edge := types.NewEntityEdge(utils.GenerateUUID(), ae.Uuid, ce.Uuid, source.GroupID, "IMPLIES", types.EntityEdgeType)
					edge.Fact = ruleNode.Summary
					edge.Metadata = map[string]interface{}{
						"rule_id":   rule.ID,
						"rule_type": rule.RuleType,
					}
					edge.Strength = rule.Confidence
					ruleEdges = append(ruleEdges, edge)
				}
			}
		}
	}

	// Add rule nodes and edges to the extracted sets so they participate in the pipeline
	allExtractedNodes = append(allExtractedNodes, ruleNodes...)
	allExtractedEdges = append(allExtractedEdges, ruleEdges...)

	if len(ruleNodes) > 0 {
		c.logger.Info("Rule promotion",
			"source_id", sourceID,
			"rule_nodes", len(ruleNodes),
			"rule_edges", len(ruleEdges),
			"entity_linking", !options.SkipRuleEntityLinking)
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

	// Verbose logging for resolved entities
	if options.Verbose {
		c.logger.Info("[VERBOSE] Resolved entities",
			"source_id", sourceID,
			"entity_count", len(entityOutput.ResolvedNodes))
		for _, n := range entityOutput.ResolvedNodes {
			c.logger.Info("[VERBOSE] Resolved Entity",
				"name", n.Name,
				"type", n.Type,
				"summary", n.Summary,
				"uuid", n.Uuid)
		}
	}

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

	// Verbose logging for resolved relationships
	if options.Verbose {
		c.logger.Info("[VERBOSE] Resolved relationships",
			"source_id", sourceID,
			"edge_count", len(relOutput.ResolvedEdges))
		for _, e := range relOutput.ResolvedEdges {
			c.logger.Info("[VERBOSE] Resolved Edge",
				"name", e.Name,
				"source_id", e.SourceNodeID,
				"target_id", e.TargetNodeID,
				"fact", e.Fact,
				"summary", e.Summary)
		}
	}

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

		// Verbose logging for community summaries
		if options.Verbose {
			c.logger.Info("[VERBOSE] Community summaries",
				"source_id", sourceID,
				"community_count", len(communities))
			for _, comm := range communities {
				c.logger.Info("[VERBOSE] Community",
					"name", comm.Name,
					"summary", comm.Summary,
					"uuid", comm.Uuid)
			}
		}
	}

	return &types.AddEpisodeResults{
		Episode:        chunkData.mainEpisodeNode,
		EpisodicEdges:  relOutput.EpisodicEdges,
		Nodes:          entityOutput.ResolvedNodes,
		Edges:          relOutput.ResolvedEdges,
		Communities:    communities,
		CommunityEdges: communityEdges,
		Rules:          ruleNodes,
		RuleEdges:      ruleEdges,
	}, nil
}

// PromoteToGraphBatched reads extracted knowledge from the fact store in batches
// and writes directly to the graph database. Unlike PromoteToGraph, this method
// streams data in configurable batch sizes to handle sources with millions of
// entities without loading everything into memory.
//
// The batch size controls how many nodes/triples/rules are loaded from the fact
// store per iteration. The default is 100,000 if batchSize <= 0.
//
// This method skips the modeler pipeline (entity resolution, community detection)
// and writes nodes/edges directly, making it suitable for pre-extracted bulk data
// like Wikidata imports.
func (c *Client) PromoteToGraphBatched(ctx context.Context, sourceID string, batchSize int, options *AddEpisodeOptions) (*PromoteBatchedResult, error) {
	if c.factStore == nil {
		return nil, fmt.Errorf("facts DB not configured")
	}
	if c.driver == nil {
		return nil, fmt.Errorf("graph driver not configured")
	}
	if batchSize <= 0 {
		batchSize = 100_000
	}
	if options == nil {
		options = &AddEpisodeOptions{}
	}

	source, err := c.factStore.GetSource(ctx, sourceID)
	if err != nil {
		return nil, fmt.Errorf("failed to get source %s: %w", sourceID, err)
	}

	// Get counts for progress reporting
	nodeCount, err := c.factStore.CountExtractedNodes(ctx, sourceID)
	if err != nil {
		return nil, fmt.Errorf("failed to count nodes: %w", err)
	}
	tripleCount, err := c.factStore.CountExtractedTriples(ctx, sourceID)
	if err != nil {
		return nil, fmt.Errorf("failed to count triples: %w", err)
	}
	ruleCount, err := c.factStore.CountExtractedRules(ctx, sourceID)
	if err != nil {
		return nil, fmt.Errorf("failed to count rules: %w", err)
	}

	c.logger.Info("Starting batched promotion",
		"source_id", sourceID,
		"nodes", nodeCount,
		"triples", tripleCount,
		"rules", ruleCount,
		"batch_size", batchSize)

	result := &PromoteBatchedResult{}

	// Phase 1: Upsert nodes in batches, building name→uuid index
	nameToUUID := make(map[string]string)

	for offset := 0; ; offset += batchSize {
		extNodes, err := c.factStore.GetExtractedNodesPaginated(ctx, sourceID, offset, batchSize)
		if err != nil {
			return nil, fmt.Errorf("failed to get nodes at offset %d: %w", offset, err)
		}
		if len(extNodes) == 0 {
			break
		}

		graphNodes := make([]*types.Node, 0, len(extNodes))
		for _, n := range extNodes {
			tn := &types.Node{
				Uuid:      n.ID,
				Name:      n.Name,
				Summary:   n.Description,
				Type:      types.EntityNodeType,
				Embedding: n.Embedding,
				GroupID:   source.GroupID,
				SourceIDs: []string{sourceID},
				CreatedAt: n.CreatedAt,
				UpdatedAt: n.CreatedAt,
				ValidFrom: n.CreatedAt,
			}
			if n.Type != "" {
				tn.Type = types.NodeType(n.Type)
			}
			nameToUUID[n.Name] = n.ID
			graphNodes = append(graphNodes, tn)
		}

		if err := c.driver.UpsertNodes(ctx, graphNodes); err != nil {
			return nil, fmt.Errorf("failed to upsert nodes at offset %d: %w", offset, err)
		}
		result.Nodes += int64(len(graphNodes))

		c.logger.Info("Promoted nodes batch",
			"offset", offset,
			"batch", len(graphNodes),
			"total", result.Nodes,
			"of", nodeCount)
	}

	// Phase 2: Upsert edges from triples in batches
	var totalFiltered int64
	for offset := 0; ; offset += batchSize {
		extTriples, err := c.factStore.GetExtractedTriplesPaginated(ctx, sourceID, offset, batchSize)
		if err != nil {
			return nil, fmt.Errorf("failed to get triples at offset %d: %w", offset, err)
		}
		if len(extTriples) == 0 {
			break
		}

		// Filter low-value triples before promotion
		extTriples, removed := types.FilterLowValueTriples(extTriples)
		totalFiltered += int64(len(removed))

		graphEdges := make([]*types.Edge, 0, len(extTriples))
		skipped := 0
		for _, t := range extTriples {
			sUUID := nameToUUID[t.Subject]
			tUUID := nameToUUID[t.Object]
			if sUUID == "" || tUUID == "" {
				skipped++
				continue
			}
			te := &types.Edge{
				BaseEdge: types.BaseEdge{
					Uuid:         t.ID,
					SourceNodeID: sUUID,
					TargetNodeID: tUUID,
					GroupID:      source.GroupID,
					CreatedAt:    t.CreatedAt,
				},
				Name:     t.Predicate,
				Summary:  t.Description,
				Fact:     t.Description,
				Strength: t.Confidence,
				Type:     types.EntityEdgeType,
				SourceID: sUUID,
				TargetID: tUUID,
			}
			if len(t.Embedding) > 0 {
				te.Embedding = t.Embedding
			}
			graphEdges = append(graphEdges, te)
		}

		if len(graphEdges) > 0 {
			if err := c.driver.UpsertEdges(ctx, graphEdges); err != nil {
				return nil, fmt.Errorf("failed to upsert edges at offset %d: %w", offset, err)
			}
		}
		result.Edges += int64(len(graphEdges))
		result.SkippedEdges += int64(skipped)

		c.logger.Info("Promoted triples batch",
			"offset", offset,
			"batch", len(graphEdges),
			"skipped", skipped,
			"total", result.Edges,
			"of", tripleCount)
	}

	// Phase 3: Upsert rules in batches
	for offset := 0; ; offset += batchSize {
		extRules, err := c.factStore.GetExtractedRulesPaginated(ctx, sourceID, offset, batchSize)
		if err != nil {
			return nil, fmt.Errorf("failed to get rules at offset %d: %w", offset, err)
		}
		if len(extRules) == 0 {
			break
		}

		ruleNodes := make([]*types.Node, 0, len(extRules))
		for _, rule := range extRules {
			ruleNode := &types.Node{
				Uuid:       rule.ID,
				Name:       buildRuleName(rule),
				Summary:    buildRuleSummary(rule),
				Content:    buildRuleContent(rule),
				Type:       types.RuleNodeType,
				EntityType: rule.RuleType,
				Embedding:  rule.Embedding,
				GroupID:    source.GroupID,
				SourceIDs:  []string{sourceID},
				Metadata:   buildRuleMetadata(rule, sourceID),
				CreatedAt:  rule.CreatedAt,
				UpdatedAt:  rule.CreatedAt,
				ValidFrom:  rule.CreatedAt,
			}
			ruleNodes = append(ruleNodes, ruleNode)
		}

		if err := c.driver.UpsertNodes(ctx, ruleNodes); err != nil {
			return nil, fmt.Errorf("failed to upsert rule nodes at offset %d: %w", offset, err)
		}
		result.Rules += int64(len(ruleNodes))

		c.logger.Info("Promoted rules batch",
			"offset", offset,
			"batch", len(ruleNodes),
			"total", result.Rules,
			"of", ruleCount)
	}

	c.logger.Info("Batched promotion complete",
		"source_id", sourceID,
		"nodes", result.Nodes,
		"edges", result.Edges,
		"skipped_edges", result.SkippedEdges,
		"filtered_low_value", totalFiltered,
		"rules", result.Rules)

	return result, nil
}

// PromoteBatchedResult holds the results of a batched promotion.
type PromoteBatchedResult struct {
	Nodes        int64
	Edges        int64
	SkippedEdges int64
	Rules        int64
}

// ValidateModeler tests a GraphModeler implementation with sample data to verify
// it works correctly before using it in production.
func (c *Client) ValidateModeler(ctx context.Context, gm modeler.GraphModeler) (*modeler.ModelerValidationResult, error) {
	opts := &modeler.ValidateModelerOptions{
		GroupID: c.config.GroupID,
	}
	return modeler.ValidateModeler(ctx, gm, opts)
}

// buildRuleName returns a truncated display name for a rule node.
func buildRuleName(rule *types.ExtractedRule) string {
	name := "IF " + rule.Antecedent + " THEN " + rule.Consequent
	if len(name) > 100 {
		return name[:97] + "..."
	}
	return name
}

// buildRuleSummary returns the full human-readable rule text.
func buildRuleSummary(rule *types.ExtractedRule) string {
	s := "IF " + rule.Antecedent + " THEN " + rule.Consequent
	if rule.Exception != "" {
		s += " UNLESS " + rule.Exception
	}
	return s
}

// buildRuleContent returns a JSON-encoded representation of the rule parts.
func buildRuleContent(rule *types.ExtractedRule) string {
	m := map[string]string{
		"antecedent": rule.Antecedent,
		"consequent": rule.Consequent,
	}
	if rule.Exception != "" {
		m["exception"] = rule.Exception
	}
	b, _ := json.Marshal(m)
	return string(b)
}

func buildRuleMetadata(rule *types.ExtractedRule, sourceID string) map[string]interface{} {
	metadata := map[string]interface{}{
		"scope":              rule.Scope,
		"source_attribution": rule.SourceAttribution,
		"confidence":         rule.Confidence,
		"source_id":          sourceID,
	}
	if rule.SourceID != "" {
		metadata["source_id"] = rule.SourceID
	}
	if rule.StructureStatus != "" {
		metadata["structure_status"] = rule.StructureStatus
	}
	if rule.StructureError != "" {
		metadata["structure_error"] = rule.StructureError
	}
	if rule.Structured != nil {
		if data, err := json.Marshal(rule.Structured); err == nil {
			metadata["structured_rule"] = string(data)
			if _, ok := metadata["structure_status"]; !ok {
				metadata["structure_status"] = ruleschema.StructureStatusParsed
			}
		} else {
			metadata["structure_status"] = ruleschema.StructureStatusInvalid
			metadata["structure_error"] = err.Error()
		}
	}
	return metadata
}

// matchEntitiesInText performs case-insensitive substring matching of node names
// against the given text, returning matching nodes.
func matchEntitiesInText(text string, uuidToNode map[string]*types.Node) []*types.Node {
	if text == "" {
		return nil
	}
	lower := strings.ToLower(text)
	var matches []*types.Node
	for _, n := range uuidToNode {
		if n.Name != "" && strings.Contains(lower, strings.ToLower(n.Name)) {
			matches = append(matches, n)
		}
	}
	return matches
}
