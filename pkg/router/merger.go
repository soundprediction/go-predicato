//go:build cgo

package router

import (
	"sort"

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
func (rm *ResultMerger) Merge(results map[string]*types.SearchResults, limit int) *MergedSearchResults {
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
		return rm.mergeRRF(results, limit)
	case "simple_concat":
		return rm.mergeSimpleConcat(results, limit)
	case "max_score":
		return rm.mergeMaxScore(results, limit)
	default:
		return rm.mergeRRF(results, limit)
	}
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
