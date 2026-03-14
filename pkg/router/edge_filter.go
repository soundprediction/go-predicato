//go:build cgo

package router

import (
	"github.com/soundprediction/predicato/pkg/types"
)

// filterLowValueEdges removes edges that are tautological, overly generic,
// or duplicate by fact content. Delegates to the shared types package.
func filterLowValueEdges(edges []*types.Edge) []*types.Edge {
	return types.FilterLowValueEdges(edges)
}
