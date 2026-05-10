package utils

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/soundprediction/predicato/pkg/types"
)

const (
	DefaultChunkSize = 10
	MinScoreNodes    = 0.8
	MinScoreEdges    = 0.6
)

// UnionFind implements the Union-Find data structure for efficient duplicate resolution
type UnionFind struct {
	parent map[string]string
}

// NewUnionFind creates a new UnionFind data structure
func NewUnionFind(elements []string) *UnionFind {
	parent := make(map[string]string)
	for _, element := range elements {
		parent[element] = element
	}
	return &UnionFind{parent: parent}
}

// Find returns the root of the set containing x (with path compression)
func (uf *UnionFind) Find(x string) string {
	if uf.parent[x] != x {
		uf.parent[x] = uf.Find(uf.parent[x]) // Path compression
	}
	return uf.parent[x]
}

// Union merges the sets containing a and b
func (uf *UnionFind) Union(a, b string) {
	rootA, rootB := uf.Find(a), uf.Find(b)
	if rootA == rootB {
		return
	}
	// Attach the lexicographically larger root under the smaller
	if rootA < rootB {
		uf.parent[rootB] = rootA
	} else {
		uf.parent[rootA] = rootB
	}
}

// CompressUUIDMap compresses duplicate pairs into a mapping from each UUID to the lexicographically smallest UUID in its duplicate set
func CompressUUIDMap(duplicatePairs [][]string) map[string]string {
	if len(duplicatePairs) == 0 {
		return make(map[string]string)
	}

	// Collect all UUIDs
	allUUIDs := make(map[string]bool)
	for _, pair := range duplicatePairs {
		if len(pair) >= 2 {
			allUUIDs[pair[0]] = true
			allUUIDs[pair[1]] = true
		}
	}

	// Convert to slice
	uuidSlice := make([]string, 0, len(allUUIDs))
	for uuid := range allUUIDs {
		uuidSlice = append(uuidSlice, uuid)
	}

	// Create UnionFind
	uf := NewUnionFind(uuidSlice)

	// Union duplicate pairs
	for _, pair := range duplicatePairs {
		if len(pair) >= 2 {
			uf.Union(pair[0], pair[1])
		}
	}

	// Build final mapping
	result := make(map[string]string)
	for _, uuid := range uuidSlice {
		result[uuid] = uf.Find(uuid)
	}

	return result
}

// BuildDirectedUUIDMap collapses alias -> canonical chains while preserving direction.
// This is used by dedupe_nodes_bulk to handle directed mappings discovered during node dedupe.
func BuildDirectedUUIDMap(pairs [][2]string) map[string]string {
	if len(pairs) == 0 {
		return make(map[string]string)
	}

	parent := make(map[string]string)

	// find performs directed union-find lookup using iterative path compression
	find := func(uuid string) string {
		if _, exists := parent[uuid]; !exists {
			parent[uuid] = uuid
		}
		root := uuid
		for parent[root] != root {
			root = parent[root]
		}

		// Path compression
		for parent[uuid] != root {
			nextUUID := parent[uuid]
			parent[uuid] = root
			uuid = nextUUID
		}

		return root
	}

	// Build the directed mapping
	for _, pair := range pairs {
		sourceUUID, targetUUID := pair[0], pair[1]
		if _, exists := parent[sourceUUID]; !exists {
			parent[sourceUUID] = sourceUUID
		}
		if _, exists := parent[targetUUID]; !exists {
			parent[targetUUID] = targetUUID
		}
		parent[find(sourceUUID)] = find(targetUUID)
	}

	// Build final mapping
	result := make(map[string]string)
	for uuid := range parent {
		result[uuid] = find(uuid)
	}

	return result
}

// ResolveEdgePointers updates edge source and target node UUIDs according to the UUID mapping
func ResolveEdgePointers(edges []*types.Edge, uuidMap map[string]string) {
	for _, edge := range edges {
		if newSourceID, exists := uuidMap[edge.SourceID]; exists {
			edge.SourceID = newSourceID
		}
		if newTargetID, exists := uuidMap[edge.TargetID]; exists {
			edge.TargetID = newTargetID
		}
	}
}

// EpisodeTuple represents an episode with its previous episodes for context
type EpisodeTuple struct {
	Episode          *types.Episode
	PreviousEpisodes []*types.Episode
}

// BulkProcessingResult represents the result of bulk processing operations
type BulkProcessingResult struct {
	UUIDMappings   map[string]string
	Errors         []error
	ProcessedItems int
}

// DedupeNodesResult represents the result of node deduplication
type DedupeNodesResult struct {
	NodesByEpisode map[string][]*types.Node
	UUIDMap        map[string]string
}

// DedupeEdgesResult represents the result of edge deduplication
type DedupeEdgesResult struct {
	EdgesByEpisode map[string][]*types.Edge
	UUIDMap        map[string]string
}

// HasWordOverlap checks if two strings have overlapping words (case-insensitive)
func HasWordOverlap(text1, text2 string) bool {
	words1 := strings.Fields(strings.ToLower(text1))
	words2 := strings.Fields(strings.ToLower(text2))

	word1Set := make(map[string]bool)
	for _, word := range words1 {
		word1Set[word] = true
	}

	for _, word := range words2 {
		if word1Set[word] {
			return true
		}
	}

	return false
}

// FindSimilarNodes finds nodes that are potentially duplicates based on word overlap and semantic similarity
func FindSimilarNodes(node *types.Node, candidates []*types.Node, minScore float64) []*types.Node {
	var similar []*types.Node

	for _, candidate := range candidates {
		// Check for word overlap first (faster)
		if HasWordOverlap(node.Name, candidate.Name) {
			similar = append(similar, candidate)
			continue
		}

		// Check semantic similarity if embeddings are available
		if len(node.Embedding) > 0 && len(candidate.Embedding) > 0 {
			similarity := CosineSimilarity(node.Embedding, candidate.Embedding)
			if similarity >= minScore {
				similar = append(similar, candidate)
			}
		}
	}

	return similar
}

// FindSimilarEdges finds edges that are potentially duplicates
func FindSimilarEdges(edge *types.Edge, candidates []*types.Edge, minScore float64) []*types.Edge {
	var similar []*types.Edge

	for _, candidate := range candidates {
		// Must have same source and target
		if edge.SourceID != candidate.SourceID || edge.TargetID != candidate.TargetID {
			continue
		}

		// Check for word overlap in fact text
		if HasWordOverlap(edge.Summary, candidate.Summary) {
			similar = append(similar, candidate)
			continue
		}

		// Check semantic similarity if embeddings are available
		if len(edge.Embedding) > 0 && len(candidate.Embedding) > 0 {
			similarity := CosineSimilarity(edge.Embedding, candidate.Embedding)
			if similarity >= minScore {
				similar = append(similar, candidate)
			}
		}
	}

	return similar
}

// GroupDuplicatesByEpisode groups duplicates back into episode-specific collections
func GroupDuplicatesByEpisode(
	originalItems map[string][]string, // episode UUID -> item UUIDs
	compressedMap map[string]string, // old UUID -> new UUID
	itemMap map[string]interface{}, // UUID -> item
) map[string][]interface{} {
	result := make(map[string][]interface{})

	for episodeUUID, itemUUIDs := range originalItems {
		var items []interface{}
		seen := make(map[string]bool)

		for _, uuid := range itemUUIDs {
			// Get the canonical UUID
			canonicalUUID := compressedMap[uuid]
			if canonicalUUID == "" {
				canonicalUUID = uuid
			}

			// Avoid duplicates within the same episode
			if !seen[canonicalUUID] {
				if item, exists := itemMap[canonicalUUID]; exists {
					items = append(items, item)
					seen[canonicalUUID] = true
				}
			}
		}

		result[episodeUUID] = items
	}

	return result
}

// BatchProcessor processes items in batches with concurrent execution
type BatchProcessor[T any, R any] struct {
	ProcessBatch   func(ctx context.Context, batch []T) ([]R, error)
	BatchSize      int
	MaxConcurrency int
}

// NewBatchProcessor creates a new batch processor
func NewBatchProcessor[T any, R any](batchSize, maxConcurrency int, processBatch func(ctx context.Context, batch []T) ([]R, error)) *BatchProcessor[T, R] {
	if batchSize <= 0 {
		batchSize = DefaultChunkSize
	}
	if maxConcurrency <= 0 {
		maxConcurrency = GetSemaphoreLimit()
	}
	return &BatchProcessor[T, R]{
		BatchSize:      batchSize,
		MaxConcurrency: maxConcurrency,
		ProcessBatch:   processBatch,
	}
}

// Process processes all items in batches
func (bp *BatchProcessor[T, R]) Process(ctx context.Context, items []T) ([]R, error) {
	if len(items) == 0 {
		return nil, nil
	}

	// Create batches
	batches := Batch(items, bp.BatchSize)

	// Process batches concurrently
	var allResults []R
	batchFunctions := make([]func() ([]R, error), len(batches))
	for i, batch := range batches {
		batchCopy := batch // Capture loop variable
		batchFunctions[i] = func() ([]R, error) {
			return bp.ProcessBatch(ctx, batchCopy)
		}
	}

	results, errors := SemaphoreGatherWithResults(ctx, bp.MaxConcurrency, batchFunctions...)

	// Collect all results and check for errors
	for i, err := range errors {
		if err != nil {
			return nil, fmt.Errorf("batch %d failed: %w", i, err)
		}
		allResults = append(allResults, results[i]...)
	}

	return allResults, nil
}

// SortStringSlice sorts a string slice in place
func SortStringSlice(slice []string) {
	sort.Strings(slice)
}

// RemoveDuplicateStrings removes duplicate strings from a slice while preserving order
func RemoveDuplicateStrings(slice []string) []string {
	seen := make(map[string]bool)
	var result []string

	for _, item := range slice {
		if !seen[item] {
			seen[item] = true
			result = append(result, item)
		}
	}

	return result
}

// ChunkSlice splits a slice into smaller chunks of the specified size
func ChunkSlice[T any](slice []T, chunkSize int) [][]T {
	if chunkSize <= 0 {
		return [][]T{slice}
	}

	var chunks [][]T
	for i := 0; i < len(slice); i += chunkSize {
		end := i + chunkSize
		if end > len(slice) {
			end = len(slice)
		}
		chunks = append(chunks, slice[i:end])
	}

	return chunks
}
