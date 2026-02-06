package modeler

import (
	"context"
	"fmt"
	"time"

	"github.com/soundprediction/predicato/pkg/types"
	"github.com/soundprediction/predicato/pkg/utils"
)

// ValidateModelerOptions configures modeler validation.
type ValidateModelerOptions struct {

	// GroupID is the group identifier for sample data
	GroupID string

	// NodeCount is the number of sample nodes to generate (default: 5)
	NodeCount int

	// EdgeCount is the number of sample edges to generate (default: 3)
	EdgeCount int

	// Timeout for each validation step (default: 30s)
	Timeout time.Duration

	// SkipCommunityValidation skips testing BuildCommunities
	SkipCommunityValidation bool
}

// ValidateModeler tests a GraphModeler implementation with sample data.
// This helps verify custom implementations work correctly before use.
//
// Returns a ModelerValidationResult with pass/fail status for each method.
func ValidateModeler(ctx context.Context, gm GraphModeler, opts *ValidateModelerOptions) (*ModelerValidationResult, error) {
	if gm == nil {
		return nil, fmt.Errorf("modeler is required")
	}

	// Apply defaults
	if opts == nil {
		opts = &ValidateModelerOptions{}
	}
	if opts.NodeCount <= 0 {
		opts.NodeCount = 5
	}
	if opts.EdgeCount <= 0 {
		opts.EdgeCount = 3
	}
	if opts.GroupID == "" {
		opts.GroupID = "validation-test"
	}
	if opts.Timeout <= 0 {
		opts.Timeout = 30 * time.Second
	}

	result := &ModelerValidationResult{
		Valid:    true,
		Warnings: []string{},
	}

	// Generate sample data
	episode := generateSampleEpisode(opts.GroupID)
	nodes := generateSampleNodes(opts.NodeCount, opts.GroupID)
	edges := generateSampleEdges(opts.EdgeCount, nodes, opts.GroupID)

	// Test ResolveEntities
	entityResult := validateResolveEntities(ctx, gm, nodes, episode, opts)
	result.EntityResolution = entityResult
	if !entityResult.Passed {
		result.Valid = false
	}

	// Test ResolveRelationships (only if entity resolution passed)
	var resolvedNodes []*types.Node
	var uuidMap map[string]string
	if entityResult.Passed {
		// Use actual resolved nodes if entity resolution worked
		resolvedNodes = nodes // Fallback
		uuidMap = make(map[string]string)
		for _, n := range nodes {
			uuidMap[n.Uuid] = n.Uuid
		}
	} else {
		resolvedNodes = nodes
		uuidMap = make(map[string]string)
		for _, n := range nodes {
			uuidMap[n.Uuid] = n.Uuid
		}
	}

	relResult := validateResolveRelationships(ctx, gm, edges, resolvedNodes, uuidMap, episode, opts)
	result.RelationshipResolution = relResult
	if !relResult.Passed {
		result.Valid = false
	}

	// Test BuildCommunities (optional)
	if !opts.SkipCommunityValidation {
		commResult := validateBuildCommunities(ctx, gm, resolvedNodes, edges, opts)
		result.CommunityDetection = commResult
		// Community detection failure is a warning, not a validation failure
		if !commResult.Passed {
			result.Warnings = append(result.Warnings, fmt.Sprintf("BuildCommunities failed: %v", commResult.Error))
		}
	}

	return result, nil
}

// validateResolveEntities tests the ResolveEntities method.
func validateResolveEntities(ctx context.Context, gm GraphModeler, nodes []*types.Node, episode *types.Node, opts *ValidateModelerOptions) *ValidationStepResult {
	ctx, cancel := context.WithTimeout(ctx, opts.Timeout)
	defer cancel()

	start := time.Now()

	input := &EntityResolutionInput{
		ExtractedNodes:   nodes,
		Episode:          episode,
		PreviousEpisodes: []*types.Node{},
		EntityTypes:      map[string]interface{}{},
		GroupID:          opts.GroupID,
		Options: &EntityResolutionOptions{
			SkipResolution: true, // Don't actually hit DB during validation
			SkipReflexion:  true,
			SkipAttributes: true,
		},
	}

	output, err := gm.ResolveEntities(ctx, input)
	latency := time.Since(start).Milliseconds()

	if err != nil {
		return &ValidationStepResult{
			Passed:  false,
			Error:   err,
			Latency: latency,
		}
	}

	// Validate output structure
	if output == nil {
		return &ValidationStepResult{
			Passed:  false,
			Error:   fmt.Errorf("ResolveEntities returned nil output"),
			Latency: latency,
		}
	}

	if output.UUIDMap == nil {
		return &ValidationStepResult{
			Passed:  false,
			Error:   fmt.Errorf("ResolveEntities returned nil UUIDMap"),
			Latency: latency,
		}
	}

	return &ValidationStepResult{
		Passed:  true,
		Latency: latency,
	}
}

// validateResolveRelationships tests the ResolveRelationships method.
func validateResolveRelationships(ctx context.Context, gm GraphModeler, edges []*types.Edge, resolvedNodes []*types.Node, uuidMap map[string]string, episode *types.Node, opts *ValidateModelerOptions) *ValidationStepResult {
	ctx, cancel := context.WithTimeout(ctx, opts.Timeout)
	defer cancel()

	start := time.Now()

	input := &RelationshipResolutionInput{
		ExtractedEdges: edges,
		ResolvedNodes:  resolvedNodes,
		UUIDMap:        uuidMap,
		Episode:        episode,
		GroupID:        opts.GroupID,
		EdgeTypes:      map[string]interface{}{},
		Options: &RelationshipResolutionOptions{
			SkipEdgeResolution: true, // Don't actually hit DB during validation
		},
	}

	output, err := gm.ResolveRelationships(ctx, input)
	latency := time.Since(start).Milliseconds()

	if err != nil {
		return &ValidationStepResult{
			Passed:  false,
			Error:   err,
			Latency: latency,
		}
	}

	// Validate output structure
	if output == nil {
		return &ValidationStepResult{
			Passed:  false,
			Error:   fmt.Errorf("ResolveRelationships returned nil output"),
			Latency: latency,
		}
	}

	return &ValidationStepResult{
		Passed:  true,
		Latency: latency,
	}
}

// validateBuildCommunities tests the BuildCommunities method.
func validateBuildCommunities(ctx context.Context, gm GraphModeler, nodes []*types.Node, edges []*types.Edge, opts *ValidateModelerOptions) *ValidationStepResult {
	ctx, cancel := context.WithTimeout(ctx, opts.Timeout)
	defer cancel()

	start := time.Now()

	input := &CommunityInput{
		Nodes:     nodes,
		Edges:     edges,
		GroupID:   opts.GroupID,
		EpisodeID: utils.GenerateUUID(),
	}

	// Note: BuildCommunities may return nil output (skip), which is valid
	_, err := gm.BuildCommunities(ctx, input)
	latency := time.Since(start).Milliseconds()

	if err != nil {
		return &ValidationStepResult{
			Passed:  false,
			Error:   err,
			Latency: latency,
		}
	}

	return &ValidationStepResult{
		Passed:  true,
		Latency: latency,
	}
}

// generateSampleEpisode creates a sample episode node for testing.
func generateSampleEpisode(groupID string) *types.Node {
	return &types.Node{
		Uuid:    utils.GenerateUUID(),
		Name:    "Validation Test Episode",
		Type:    types.EpisodicNodeType,
		GroupID: groupID,
		Content: "This is a sample episode for modeler validation.",
		Summary: "Validation test episode",
		Metadata: map[string]interface{}{
			"source": "validation",
		},
	}
}

// generateSampleNodes creates sample entity nodes for testing.
func generateSampleNodes(count int, groupID string) []*types.Node {
	names := []string{
		"Alice",
		"Bob",
		"Acme Corp",
		"New York",
		"Project Alpha",
		"John",
		"TechCo",
		"San Francisco",
		"Product X",
		"Charlie",
	}

	nodeTypes := []string{
		"PERSON",
		"PERSON",
		"ORGANIZATION",
		"LOCATION",
		"PROJECT",
		"PERSON",
		"ORGANIZATION",
		"LOCATION",
		"PRODUCT",
		"PERSON",
	}

	nodes := make([]*types.Node, 0, count)
	for i := 0; i < count && i < len(names); i++ {
		node := &types.Node{
			Uuid:       utils.GenerateUUID(),
			Name:       names[i],
			Type:       types.EntityNodeType,
			EntityType: nodeTypes[i],
			GroupID:    groupID,
			Metadata: map[string]interface{}{
				"source": "validation",
			},
		}
		nodes = append(nodes, node)
	}

	return nodes
}

// generateSampleEdges creates sample relationship edges for testing.
func generateSampleEdges(count int, nodes []*types.Node, groupID string) []*types.Edge {
	if len(nodes) < 2 {
		return []*types.Edge{}
	}

	relationTypes := []string{
		"WORKS_FOR",
		"KNOWS",
		"LOCATED_IN",
		"MANAGES",
		"OWNS",
	}

	edges := make([]*types.Edge, 0, count)
	for i := 0; i < count && i < len(relationTypes); i++ {
		// Create edges between consecutive nodes (with wrap-around)
		sourceIdx := i % len(nodes)
		targetIdx := (i + 1) % len(nodes)

		// Use NewEntityEdge helper which properly initializes all fields
		edge := types.NewEntityEdge(
			utils.GenerateUUID(),
			nodes[sourceIdx].Uuid,
			nodes[targetIdx].Uuid,
			groupID,
			relationTypes[i],
			types.EntityEdgeType,
		)
		edges = append(edges, edge)
	}

	return edges
}
