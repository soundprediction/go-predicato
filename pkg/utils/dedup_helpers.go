package utils

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"regexp"
	"strings"
	"sync"

	"github.com/soundprediction/predicato/pkg/types"
)

// Constants for deduplication heuristics
const (
	NameEntropyThreshold  = 1.5
	MinNameLength         = 6
	MinTokenCount         = 2
	FuzzyJaccardThreshold = 0.9
	MinHashPermutations   = 32
	MinHashBandSize       = 4
)

var (
	// Cache for shingles to avoid recomputation
	shingleCache sync.Map

	whitespaceRe  = regexp.MustCompile(`\s+`)
	nonAlphaNumRe = regexp.MustCompile(`[^a-z0-9' ]`)
)

// NormalizeStringExact lowercases text and collapses whitespace so equal names map to the same key
func NormalizeStringExact(name string) string {
	normalized := whitespaceRe.ReplaceAllString(strings.ToLower(name), " ")
	return strings.TrimSpace(normalized)
}

// normalizeNameForFuzzy produces a fuzzier form that keeps alphanumerics and apostrophes for n-gram shingles
func normalizeNameForFuzzy(name string) string {
	normalized := NormalizeStringExact(name)
	normalized = nonAlphaNumRe.ReplaceAllString(normalized, " ")
	normalized = whitespaceRe.ReplaceAllString(strings.TrimSpace(normalized), " ")
	return normalized
}

// nameEntropy approximates text specificity using Shannon entropy over characters
func nameEntropy(normalizedName string) float64 {
	if normalizedName == "" {
		return 0.0
	}

	// Strip spaces and count character frequencies
	text := strings.ReplaceAll(normalizedName, " ", "")
	counts := make(map[rune]int)
	for _, char := range text {
		counts[char]++
	}

	total := len(text)
	if total == 0 {
		return 0.0
	}

	var entropy float64
	for _, count := range counts {
		probability := float64(count) / float64(total)
		entropy -= probability * math.Log2(probability)
	}

	return entropy
}

// HasHighEntropy filters out very short or low-entropy names that are unreliable for fuzzy matching
func HasHighEntropy(normalizedName string) bool {
	tokenCount := len(strings.Fields(normalizedName))
	if len(normalizedName) < MinNameLength && tokenCount < MinTokenCount {
		return false
	}

	return nameEntropy(normalizedName) >= NameEntropyThreshold
}

// shingles creates 3-gram shingles from the normalized name for MinHash calculations
func shingles(normalizedName string) []string {
	cleaned := strings.ReplaceAll(normalizedName, " ", "")
	if len(cleaned) < 2 {
		if cleaned == "" {
			return []string{}
		}
		return []string{cleaned}
	}

	shingleSet := make([]string, 0, len(cleaned)-2)
	for i := 0; i < len(cleaned)-2; i++ {
		shingleSet = append(shingleSet, cleaned[i:i+3])
	}
	return shingleSet
}

// CachedShingles caches shingle sets per normalized name to avoid recomputation
func CachedShingles(name string) []string {
	if cached, ok := shingleCache.Load(name); ok {
		return cached.([]string)
	}

	result := shingles(name)
	shingleCache.Store(name, result)
	return result
}

// hashShingle generates a deterministic 64-bit hash for a shingle given the permutation seed
func hashShingle(shingle string, seed int) uint64 {
	h := sha256.New()
	h.Write([]byte(fmt.Sprintf("%d:%s", seed, shingle)))
	hash := h.Sum(nil)
	return binary.BigEndian.Uint64(hash[:8])
}

// MinHashSignature computes the MinHash signature for the shingle set across predefined permutations
func MinHashSignature(shingleSet []string) []uint64 {
	if len(shingleSet) == 0 {
		return []uint64{}
	}

	signature := make([]uint64, MinHashPermutations)
	for seed := 0; seed < MinHashPermutations; seed++ {
		minHash := uint64(math.MaxUint64)
		for _, shingle := range shingleSet {
			hash := hashShingle(shingle, seed)
			if hash < minHash {
				minHash = hash
			}
		}
		signature[seed] = minHash
	}

	return signature
}

// LSHBands splits the MinHash signature into fixed-size bands for locality-sensitive hashing
func LSHBands(signature []uint64) [][]uint64 {
	if len(signature) == 0 {
		return [][]uint64{}
	}

	// Pre-allocate bands slice with expected capacity
	numBands := len(signature) / MinHashBandSize
	bands := make([][]uint64, 0, numBands)
	for start := 0; start < len(signature); start += MinHashBandSize {
		end := start + MinHashBandSize
		if end > len(signature) {
			break
		}
		band := make([]uint64, MinHashBandSize)
		copy(band, signature[start:end])
		if len(band) == MinHashBandSize {
			bands = append(bands, band)
		}
	}

	return bands
}

// JaccardSimilarity returns the Jaccard similarity between two shingle sets
func JaccardSimilarity(a, b []string) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1.0
	}
	if len(a) == 0 || len(b) == 0 {
		return 0.0
	}

	// Convert to sets
	setA := make(map[string]bool)
	for _, s := range a {
		setA[s] = true
	}

	setB := make(map[string]bool)
	for _, s := range b {
		setB[s] = true
	}

	// Calculate intersection
	intersection := 0
	for s := range setA {
		if setB[s] {
			intersection++
		}
	}

	// Calculate union
	union := len(setA) + len(setB) - intersection

	if union == 0 {
		return 0.0
	}

	return float64(intersection) / float64(union)
}

// NodePair represents a pair of duplicate nodes
type NodePair struct {
	Source *types.Node
	Target *types.Node
}

// DedupCandidateIndexes holds precomputed lookup structures that drive entity deduplication heuristics
type DedupCandidateIndexes struct {
	NodesByUUID         map[string]*types.Node
	NormalizedExisting  map[string][]*types.Node // normalized name -> nodes
	ShinglesByCandidate map[string][]string      // uuid -> shingles
	LSHBuckets          map[string][]string      // band key -> []uuid
	ExistingNodes       []*types.Node
}

// DedupResolutionState holds mutable resolution bookkeeping shared across deterministic and LLM passes
type DedupResolutionState struct {
	ResolvedNodes     []*types.Node
	UUIDMap           map[string]string
	UnresolvedIndices []int
	DuplicatePairs    []NodePair
}

// BuildCandidateIndexes precomputes exact and fuzzy lookup structures once per dedupe run
func BuildCandidateIndexes(existingNodes []*types.Node) *DedupCandidateIndexes {
	indexes := &DedupCandidateIndexes{
		ExistingNodes:       existingNodes,
		NodesByUUID:         make(map[string]*types.Node),
		NormalizedExisting:  make(map[string][]*types.Node),
		ShinglesByCandidate: make(map[string][]string),
		LSHBuckets:          make(map[string][]string),
	}

	for _, candidate := range existingNodes {
		normalized := NormalizeStringExact(candidate.Name)
		indexes.NormalizedExisting[normalized] = append(indexes.NormalizedExisting[normalized], candidate)
		indexes.NodesByUUID[candidate.Uuid] = candidate

		shingles := CachedShingles(normalizeNameForFuzzy(candidate.Name))
		indexes.ShinglesByCandidate[candidate.Uuid] = shingles

		signature := MinHashSignature(shingles)
		bands := LSHBands(signature)
		for bandIndex, band := range bands {
			// Create a key for this band
			bandKey := fmt.Sprintf("%d:%v", bandIndex, band)
			indexes.LSHBuckets[bandKey] = append(indexes.LSHBuckets[bandKey], candidate.Uuid)
		}
	}

	return indexes
}

// ResolveWithSimilarity attempts deterministic resolution using exact name hits and fuzzy MinHash comparisons
func ResolveWithSimilarity(
	extractedNodes []*types.Node,
	indexes *DedupCandidateIndexes,
	state *DedupResolutionState,
) {
	for idx, node := range extractedNodes {
		normalizedExact := NormalizeStringExact(node.Name)
		normalizedFuzzy := normalizeNameForFuzzy(node.Name)

		if !HasHighEntropy(normalizedFuzzy) {
			state.UnresolvedIndices = append(state.UnresolvedIndices, idx)
			continue
		}

		// Check for exact matches
		existingMatches := indexes.NormalizedExisting[normalizedExact]
		if len(existingMatches) == 1 {
			match := existingMatches[0]
			state.ResolvedNodes[idx] = match
			state.UUIDMap[node.Uuid] = match.Uuid
			if match.Uuid != node.Uuid {
				state.DuplicatePairs = append(state.DuplicatePairs, NodePair{
					Source: node,
					Target: match,
				})
			}
			continue
		}
		if len(existingMatches) > 1 {
			// Multiple matches - needs LLM resolution
			state.UnresolvedIndices = append(state.UnresolvedIndices, idx)
			continue
		}

		// Try fuzzy matching using MinHash + LSH
		shingles := CachedShingles(normalizedFuzzy)
		signature := MinHashSignature(shingles)
		candidateIDs := make(map[string]bool)

		bands := LSHBands(signature)
		for bandIndex, band := range bands {
			bandKey := fmt.Sprintf("%d:%v", bandIndex, band)
			for _, candidateID := range indexes.LSHBuckets[bandKey] {
				candidateIDs[candidateID] = true
			}
		}

		// Find best candidate based on Jaccard similarity
		var bestCandidate *types.Node
		bestScore := 0.0
		for candidateID := range candidateIDs {
			candidateShingles := indexes.ShinglesByCandidate[candidateID]
			score := JaccardSimilarity(shingles, candidateShingles)
			if score > bestScore {
				bestScore = score
				bestCandidate = indexes.NodesByUUID[candidateID]
			}
		}

		if bestCandidate != nil && bestScore >= FuzzyJaccardThreshold {
			state.ResolvedNodes[idx] = bestCandidate
			state.UUIDMap[node.Uuid] = bestCandidate.Uuid
			if bestCandidate.Uuid != node.Uuid {
				state.DuplicatePairs = append(state.DuplicatePairs, NodePair{
					Source: node,
					Target: bestCandidate,
				})
			}
			continue
		}

		// No match found
		state.UnresolvedIndices = append(state.UnresolvedIndices, idx)
	}
}
