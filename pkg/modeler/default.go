package modeler

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/soundprediction/predicato/pkg/community"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/embedder"
	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/prompts"
	"github.com/soundprediction/predicato/pkg/types"
	"github.com/soundprediction/predicato/pkg/utils"
	"github.com/soundprediction/predicato/pkg/utils/maintenance"
)

// NlpModels holds specialized NLP clients for different pipeline steps.
// This mirrors the structure in predicato.go for consistency.
//
// NOTE: This type is intentionally duplicated from predicato.NlpModels to avoid
// an import cycle (modeler cannot import root package). Both types should be
// kept in sync. If adding fields here, also add them to predicato.NlpModels.
type NlpModels struct {
	NodeExtraction nlp.Client
	NodeReflexion  nlp.Client
	NodeResolution nlp.Client
	NodeAttribute  nlp.Client
	EdgeExtraction nlp.Client
	EdgeResolution nlp.Client
	Summarization  nlp.Client
	TextGeneration nlp.Client // Added for parity with predicato.NlpModels
}

// DefaultModelerOptions configures the DefaultModeler.
type DefaultModelerOptions struct {
	// Driver is the graph database driver (required)
	Driver driver.GraphDriver

	// NlpClient is the default NLP client for all operations
	NlpClient nlp.Client

	// Embedder generates vector embeddings
	Embedder embedder.Client

	// NlpModels provides specialized NLP clients per step (optional)
	// If nil or a specific field is nil, falls back to NlpClient
	NlpModels *NlpModels

	// Logger for debug output (optional, defaults to slog.Default())
	Logger *slog.Logger

	// UseYAML uses YAML format for NLP model prompts instead of TSV
	UseYAML bool
}

// DefaultModeler implements GraphModeler using the existing NodeOperations,
// EdgeOperations, and community.Builder logic.
//
// This is the standard graph modeling implementation that:
// - Resolves entities using embedding similarity and NLP confirmation
// - Deduplicates relationships between resolved entities
// - Detects communities using label propagation
type DefaultModeler struct {
	driver    driver.GraphDriver
	nlpClient nlp.Client
	embedder  embedder.Client
	nlpModels *NlpModels
	community *community.Builder
	logger    *slog.Logger
	useYAML   bool
}

// NewDefaultModeler creates a new DefaultModeler with the given options.
func NewDefaultModeler(opts *DefaultModelerOptions) (*DefaultModeler, error) {
	if opts == nil {
		return nil, fmt.Errorf("options are required")
	}
	if opts.Driver == nil {
		return nil, fmt.Errorf("driver is required")
	}
	if opts.NlpClient == nil {
		return nil, fmt.Errorf("nlp client is required")
	}
	if opts.Embedder == nil {
		return nil, fmt.Errorf("embedder is required")
	}

	logger := opts.Logger
	if logger == nil {
		logger = slog.Default()
	}

	// Create community builder
	var summarizer nlp.Client
	if opts.NlpModels != nil && opts.NlpModels.Summarization != nil {
		summarizer = opts.NlpModels.Summarization
	} else {
		summarizer = opts.NlpClient
	}
	communityBuilder, err := community.NewBuilder(opts.Driver, opts.NlpClient, summarizer, opts.Embedder)
	if err != nil {
		return nil, fmt.Errorf("failed to create community builder: %w", err)
	}

	return &DefaultModeler{
		driver:    opts.Driver,
		nlpClient: opts.NlpClient,
		embedder:  opts.Embedder,
		nlpModels: opts.NlpModels,
		community: communityBuilder,
		logger:    logger,
		useYAML:   opts.UseYAML,
	}, nil
}

// ResolveEntities performs entity resolution using the existing NodeOperations logic.
func (m *DefaultModeler) ResolveEntities(ctx context.Context, input *EntityResolutionInput) (*EntityResolutionOutput, error) {
	if input == nil {
		return nil, fmt.Errorf("input is required")
	}

	if len(input.ExtractedNodes) == 0 {
		return &EntityResolutionOutput{
			ResolvedNodes: []*types.Node{},
			UUIDMap:       make(map[string]string),
		}, nil
	}

	// Create NodeOperations with the configured NLP clients
	nodeOps := maintenance.NewNodeOperations(m.driver, m.nlpClient, m.embedder, prompts.NewLibrary())
	nodeOps.SetLogger(m.logger)
	nodeOps.UseYAML = m.useYAML

	// Apply NLP model specialization
	if m.nlpModels != nil {
		if m.nlpModels.NodeExtraction != nil {
			nodeOps.ExtractionNLP = m.nlpModels.NodeExtraction
		}
		if m.nlpModels.NodeReflexion != nil {
			nodeOps.ReflexionNLP = m.nlpModels.NodeReflexion
		}
		if m.nlpModels.NodeResolution != nil {
			nodeOps.ResolutionNLP = m.nlpModels.NodeResolution
		}
		if m.nlpModels.NodeAttribute != nil {
			nodeOps.AttributeNLP = m.nlpModels.NodeAttribute
		}
	}

	// Apply skip flags from options
	if input.Options != nil {
		nodeOps.SkipResolution = input.Options.SkipResolution
		nodeOps.SkipReflexion = input.Options.SkipReflexion
		nodeOps.SkipAttributes = input.Options.SkipAttributes
	}

	// Call the existing resolution logic
	resolvedNodes, uuidMap, _, err := nodeOps.ResolveExtractedNodes(
		ctx,
		input.ExtractedNodes,
		input.Episode,
		input.PreviousEpisodes,
		input.EntityTypes,
	)
	if err != nil {
		return nil, fmt.Errorf("entity resolution failed: %w", err)
	}

	// Count new vs merged
	mergedCount := 0
	for origUUID, resolvedUUID := range uuidMap {
		if origUUID != resolvedUUID {
			mergedCount++
		}
	}

	return &EntityResolutionOutput{
		ResolvedNodes: resolvedNodes,
		UUIDMap:       uuidMap,
		MergedCount:   mergedCount,
		NewCount:      len(resolvedNodes) - mergedCount,
	}, nil
}

// ResolveRelationships performs edge resolution using the existing EdgeOperations logic.
func (m *DefaultModeler) ResolveRelationships(ctx context.Context, input *RelationshipResolutionInput) (*RelationshipResolutionOutput, error) {
	if input == nil {
		return nil, fmt.Errorf("input is required")
	}

	if len(input.ExtractedEdges) == 0 {
		return &RelationshipResolutionOutput{
			ResolvedEdges: []*types.Edge{},
			EpisodicEdges: []*types.Edge{},
		}, nil
	}

	// Create EdgeOperations with configured NLP clients
	edgeOps := maintenance.NewEdgeOperations(m.driver, m.nlpClient, m.embedder, prompts.NewLibrary())
	edgeOps.SetLogger(m.logger)
	edgeOps.UseYAML = m.useYAML

	// Apply NLP model specialization
	if m.nlpModels != nil {
		if m.nlpModels.EdgeExtraction != nil {
			edgeOps.ExtractionNLP = m.nlpModels.EdgeExtraction
		}
		if m.nlpModels.EdgeResolution != nil {
			edgeOps.ResolutionNLP = m.nlpModels.EdgeResolution
		}
	}

	// Apply skip flags
	if input.Options != nil {
		edgeOps.SkipResolution = input.Options.SkipEdgeResolution
	}

	// Apply UUID mapping to edges
	mappedEdges := make([]*types.Edge, 0, len(input.ExtractedEdges))
	for _, edge := range input.ExtractedEdges {
		mappedEdge := edge // Copy
		if newUUID, ok := input.UUIDMap[edge.SourceNodeID]; ok {
			mappedEdge.SourceNodeID = newUUID
			mappedEdge.SourceID = newUUID
		}
		if newUUID, ok := input.UUIDMap[edge.TargetNodeID]; ok {
			mappedEdge.TargetNodeID = newUUID
			mappedEdge.TargetID = newUUID
		}
		mappedEdges = append(mappedEdges, mappedEdge)
	}

	// Resolve edges using existing logic
	resolvedEdges, newEdges, err := edgeOps.ResolveExtractedEdges(
		ctx,
		mappedEdges,
		input.Episode,
		input.ResolvedNodes,
		input.EdgeTypes,
	)
	if err != nil {
		return nil, fmt.Errorf("relationship resolution failed: %w", err)
	}

	// Build episodic edges
	episodicEdges, err := edgeOps.BuildEpisodicEdges(ctx, input.ResolvedNodes, input.Episode.Uuid, time.Now())
	if err != nil {
		return nil, fmt.Errorf("failed to build episodic edges: %w", err)
	}

	return &RelationshipResolutionOutput{
		ResolvedEdges: resolvedEdges,
		EpisodicEdges: episodicEdges,
		NewCount:      len(newEdges),
		UpdatedCount:  len(resolvedEdges) - len(newEdges),
	}, nil
}

// BuildCommunities detects communities using the existing community.Builder.
func (m *DefaultModeler) BuildCommunities(ctx context.Context, input *CommunityInput) (*CommunityOutput, error) {
	if input == nil {
		return nil, nil // Skip community detection if no input
	}

	if len(input.Nodes) == 0 {
		return nil, nil // Nothing to cluster
	}

	// Build communities using existing logic
	result, err := m.community.BuildCommunities(ctx, []string{input.GroupID}, m.logger)
	if err != nil {
		return nil, fmt.Errorf("community detection failed: %w", err)
	}

	if result == nil {
		return nil, nil
	}

	return &CommunityOutput{
		Communities:    result.CommunityNodes,
		CommunityEdges: result.CommunityEdges,
	}, nil
}

// Ensure DefaultModeler implements GraphModeler
var _ GraphModeler = (*DefaultModeler)(nil)

// NoOpModeler is a GraphModeler that does nothing.
// Useful for testing or when you want to skip all graph modeling.
type NoOpModeler struct{}

// ResolveEntities returns the input nodes without any resolution.
func (m *NoOpModeler) ResolveEntities(ctx context.Context, input *EntityResolutionInput) (*EntityResolutionOutput, error) {
	if input == nil {
		return &EntityResolutionOutput{
			ResolvedNodes: []*types.Node{},
			UUIDMap:       make(map[string]string),
		}, nil
	}

	// Create identity UUID map
	uuidMap := make(map[string]string)
	for _, node := range input.ExtractedNodes {
		uuidMap[node.Uuid] = node.Uuid
	}

	return &EntityResolutionOutput{
		ResolvedNodes: input.ExtractedNodes,
		UUIDMap:       uuidMap,
		NewCount:      len(input.ExtractedNodes),
	}, nil
}

// ResolveRelationships returns the input edges without any resolution.
func (m *NoOpModeler) ResolveRelationships(ctx context.Context, input *RelationshipResolutionInput) (*RelationshipResolutionOutput, error) {
	if input == nil {
		return &RelationshipResolutionOutput{
			ResolvedEdges: []*types.Edge{},
			EpisodicEdges: []*types.Edge{},
		}, nil
	}

	// Build basic episodic edges
	episodicEdges := make([]*types.Edge, 0, len(input.ResolvedNodes))
	for _, node := range input.ResolvedNodes {
		edge := types.NewEntityEdge(
			utils.GenerateUUID(),
			input.Episode.Uuid,
			node.Uuid,
			node.GroupID,
			"MENTIONED_IN",
			types.EpisodicEdgeType,
		)
		episodicEdges = append(episodicEdges, edge)
	}

	return &RelationshipResolutionOutput{
		ResolvedEdges: input.ExtractedEdges,
		EpisodicEdges: episodicEdges,
		NewCount:      len(input.ExtractedEdges),
	}, nil
}

// BuildCommunities returns nil (skips community detection).
func (m *NoOpModeler) BuildCommunities(ctx context.Context, input *CommunityInput) (*CommunityOutput, error) {
	return nil, nil
}

// Ensure NoOpModeler implements GraphModeler
var _ GraphModeler = (*NoOpModeler)(nil)
