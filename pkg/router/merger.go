//go:build cgo

package router

import (
	"sort"
	"strings"

	"github.com/soundprediction/predicato/pkg/types"
)

// MergedSearchResults contains the merged results from multiple predicato clients.
type MergedSearchResults struct {
	// Sources maps node IDs to the clients they came from.
	Sources map[string][]string `json:"sources"`

	// ClientResults contains the individual results from each client.
	ClientResults map[string]*types.SearchResults `json:"client_results,omitempty"`

	// Errors contains any errors that occurred during search.
	Errors map[string]error `json:"errors,omitempty"`

	// Nodes is the merged list of nodes, sorted by combined score.
	Nodes []*types.Node `json:"nodes"`

	// Edges is the merged list of edges.
	Edges []*types.Edge `json:"edges"`

	// Total is the total number of results before limiting.
	Total int `json:"total"`
}

// ScoredNode wraps a node with its combined score for sorting.
type ScoredNode struct {
	Node    *types.Node
	Sources []string
	Score   float64
}

// ResultMerger merges search results from multiple predicato clients.
type ResultMerger struct {
	strategy string
	k        int // RRF constant (default: 60)
}

// NewResultMerger creates a new result merger with the given strategy.
func NewResultMerger(strategy string) *ResultMerger {
	if strategy == "" {
		strategy = "rrf"
	}
	return &ResultMerger{
		strategy: strategy,
		k:        60,
	}
}

// Merge combines results from multiple clients according to the configured strategy.
// preferredPredicates optionally boosts edges with matching names to the top.
func (rm *ResultMerger) Merge(results map[string]*types.SearchResults, limit int, preferredPredicates []string) *MergedSearchResults {
	merged := &MergedSearchResults{
		Sources:       make(map[string][]string),
		ClientResults: results,
		Errors:        make(map[string]error),
	}

	if len(results) == 0 {
		return merged
	}

	switch rm.strategy {
	case "rrf":
		merged = rm.mergeRRF(results, limit)
	case "simple_concat":
		merged = rm.mergeSimpleConcat(results, limit)
	case "max_score":
		merged = rm.mergeMaxScore(results, limit)
	default:
		merged = rm.mergeRRF(results, limit)
	}

	// Deduplicate nodes by normalized name (collapses "Renal failure" / "RENAL FAILURE" etc.)
	merged.Nodes, merged.Sources = deduplicateNodesByName(merged.Nodes, merged.Sources)
	if limit > 0 && len(merged.Nodes) > limit {
		merged.Nodes = merged.Nodes[:limit]
	}

	// Filter low-value edges: tautological, overly generic, duplicate facts
	merged.Edges = filterLowValueEdges(merged.Edges)

	// Sort edges: preferred predicates first, then original order
	if len(preferredPredicates) > 0 {
		merged.Edges = boostPreferredEdges(merged.Edges, preferredPredicates)
	}

	return merged
}

func (rm *ResultMerger) mergeRRF(results map[string]*types.SearchResults, limit int) *MergedSearchResults {
	merged := &MergedSearchResults{
		Sources:       make(map[string][]string),
		ClientResults: results,
		Errors:        make(map[string]error),
	}

	nodeScores := make(map[string]*ScoredNode)

	for clientName, clientResults := range results {
		if clientResults == nil {
			continue
		}

		for rank, node := range clientResults.Nodes {
			if node == nil || node.Uuid == "" {
				continue
			}

			rrfScore := 1.0 / float64(rm.k+rank+1)

			if existing, ok := nodeScores[node.Uuid]; ok {
				existing.Score += rrfScore
				existing.Sources = appendUnique(existing.Sources, clientName)
			} else {
				nodeScores[node.Uuid] = &ScoredNode{
					Node:    node,
					Score:   rrfScore,
					Sources: []string{clientName},
				}
			}
		}

		for _, edge := range clientResults.Edges {
			if edge != nil {
				merged.Edges = append(merged.Edges, edge)
			}
		}
	}

	scoredNodes := make([]*ScoredNode, 0, len(nodeScores))
	for _, sn := range nodeScores {
		scoredNodes = append(scoredNodes, sn)
	}

	sort.Slice(scoredNodes, func(i, j int) bool {
		return scoredNodes[i].Score > scoredNodes[j].Score
	})

	merged.Total = len(scoredNodes)
	if limit > 0 && len(scoredNodes) > limit {
		scoredNodes = scoredNodes[:limit]
	}

	merged.Nodes = make([]*types.Node, len(scoredNodes))
	for i, sn := range scoredNodes {
		merged.Nodes[i] = sn.Node
		merged.Sources[sn.Node.Uuid] = sn.Sources
	}

	merged.Edges = deduplicateEdges(merged.Edges)

	return merged
}

func (rm *ResultMerger) mergeSimpleConcat(results map[string]*types.SearchResults, limit int) *MergedSearchResults {
	merged := &MergedSearchResults{
		Sources:       make(map[string][]string),
		ClientResults: results,
		Errors:        make(map[string]error),
	}

	nodeMap := make(map[string]*types.Node)

	for clientName, clientResults := range results {
		if clientResults == nil {
			continue
		}

		for _, node := range clientResults.Nodes {
			if node == nil || node.Uuid == "" {
				continue
			}

			if _, ok := nodeMap[node.Uuid]; !ok {
				nodeMap[node.Uuid] = node
			}
			merged.Sources[node.Uuid] = appendUnique(merged.Sources[node.Uuid], clientName)
		}

		for _, edge := range clientResults.Edges {
			if edge != nil {
				merged.Edges = append(merged.Edges, edge)
			}
		}
	}

	merged.Nodes = make([]*types.Node, 0, len(nodeMap))
	for _, node := range nodeMap {
		merged.Nodes = append(merged.Nodes, node)
	}

	merged.Total = len(merged.Nodes)

	if limit > 0 && len(merged.Nodes) > limit {
		merged.Nodes = merged.Nodes[:limit]
	}

	merged.Edges = deduplicateEdges(merged.Edges)

	return merged
}

func (rm *ResultMerger) mergeMaxScore(results map[string]*types.SearchResults, limit int) *MergedSearchResults {
	merged := &MergedSearchResults{
		Sources:       make(map[string][]string),
		ClientResults: results,
		Errors:        make(map[string]error),
	}

	nodeScores := make(map[string]*ScoredNode)

	for clientName, clientResults := range results {
		if clientResults == nil {
			continue
		}

		for rank, node := range clientResults.Nodes {
			if node == nil || node.Uuid == "" {
				continue
			}

			score := 1.0 / float64(rank+1)

			if existing, ok := nodeScores[node.Uuid]; ok {
				if score > existing.Score {
					existing.Score = score
					existing.Node = node
				}
				existing.Sources = appendUnique(existing.Sources, clientName)
			} else {
				nodeScores[node.Uuid] = &ScoredNode{
					Node:    node,
					Score:   score,
					Sources: []string{clientName},
				}
			}
		}

		for _, edge := range clientResults.Edges {
			if edge != nil {
				merged.Edges = append(merged.Edges, edge)
			}
		}
	}

	scoredNodes := make([]*ScoredNode, 0, len(nodeScores))
	for _, sn := range nodeScores {
		scoredNodes = append(scoredNodes, sn)
	}

	sort.Slice(scoredNodes, func(i, j int) bool {
		return scoredNodes[i].Score > scoredNodes[j].Score
	})

	merged.Total = len(scoredNodes)
	if limit > 0 && len(scoredNodes) > limit {
		scoredNodes = scoredNodes[:limit]
	}

	merged.Nodes = make([]*types.Node, len(scoredNodes))
	for i, sn := range scoredNodes {
		merged.Nodes[i] = sn.Node
		merged.Sources[sn.Node.Uuid] = sn.Sources
	}

	merged.Edges = deduplicateEdges(merged.Edges)

	return merged
}

// deduplicateNodesByName merges nodes that have different UUIDs but the same
// normalized name (case-insensitive, trimmed). The first occurrence (highest
// ranked) is kept; subsequent duplicates have their sources merged in.
func deduplicateNodesByName(nodes []*types.Node, sources map[string][]string) ([]*types.Node, map[string][]string) {
	if len(nodes) <= 1 {
		return nodes, sources
	}

	type seenEntry struct {
		index int
		uuid  string
	}
	seen := make(map[string]*seenEntry, len(nodes))
	result := make([]*types.Node, 0, len(nodes))

	for _, node := range nodes {
		key := strings.ToLower(strings.TrimSpace(node.Name))
		if entry, exists := seen[key]; exists {
			// Merge sources from the duplicate into the kept node
			for _, src := range sources[node.Uuid] {
				sources[entry.uuid] = appendUnique(sources[entry.uuid], src)
			}
			delete(sources, node.Uuid)
		} else {
			seen[key] = &seenEntry{index: len(result), uuid: node.Uuid}
			result = append(result, node)
		}
	}

	return result, sources
}

// boostPreferredEdges re-sorts edges so that edges with preferred predicate
// names appear first, preserving relative order within each group.
func boostPreferredEdges(edges []*types.Edge, preferred []string) []*types.Edge {
	if len(edges) == 0 || len(preferred) == 0 {
		return edges
	}

	// Build a set for O(1) lookup; store index for sub-ordering preferred edges
	prefIdx := make(map[string]int, len(preferred))
	for i, p := range preferred {
		prefIdx[strings.ToUpper(p)] = i
	}

	type ranked struct {
		edge     *types.Edge
		prefRank int // -1 means not preferred
		origRank int
	}

	ranked_ := make([]ranked, len(edges))
	for i, e := range edges {
		pr := -1
		if idx, ok := prefIdx[strings.ToUpper(e.Name)]; ok {
			pr = idx
		}
		ranked_[i] = ranked{edge: e, prefRank: pr, origRank: i}
	}

	sort.SliceStable(ranked_, func(i, j int) bool {
		pi, pj := ranked_[i].prefRank, ranked_[j].prefRank
		// Preferred edges before non-preferred
		if (pi >= 0) != (pj >= 0) {
			return pi >= 0
		}
		// Among preferred, sort by predicate priority
		if pi >= 0 && pj >= 0 && pi != pj {
			return pi < pj
		}
		// Otherwise preserve original order
		return ranked_[i].origRank < ranked_[j].origRank
	})

	result := make([]*types.Edge, len(edges))
	for i, r := range ranked_ {
		result[i] = r.edge
	}
	return result
}

func deduplicateEdges(edges []*types.Edge) []*types.Edge {
	seen := make(map[string]bool)
	result := make([]*types.Edge, 0, len(edges))

	for _, edge := range edges {
		if edge == nil || edge.Uuid == "" {
			continue
		}
		if !seen[edge.Uuid] {
			seen[edge.Uuid] = true
			result = append(result, edge)
		}
	}

	return result
}
