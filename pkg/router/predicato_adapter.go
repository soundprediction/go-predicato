//go:build cgo

package router

import (
	"context"

	predicato "github.com/soundprediction/predicato"
	"github.com/soundprediction/predicato/pkg/types"
)

// RouterPredicato adapts a multi-graph Router to the predicato.Predicato
// interface so it can be served by the standard gRPC server (pkg/grpcsvc).
//
// It embeds a default predicato.Predicato (the router's default client) so that
// all interface methods are satisfied automatically; only Search is overridden
// to route the query across topic graphs and merge the results.
type RouterPredicato struct {
	predicato.Predicato // default client: provides all non-Search methods
	r                   *Router
}

// NewRouterPredicato wraps a Router and a default predicato.Predicato.
// The default client backs every method except Search (which the router
// fans out across the matched topic graphs and merges).
func NewRouterPredicato(r *Router, def predicato.Predicato) *RouterPredicato {
	return &RouterPredicato{
		Predicato: def,
		r:         r,
	}
}

// Search routes the query across the topic graphs and merges the results into a
// single types.SearchResults. Per-node graph provenance (GroupID) is preserved;
// when a merged node has an empty GroupID it is backfilled from the merger's
// per-node Sources map so downstream consumers keep graph provenance.
func (a *RouterPredicato) Search(ctx context.Context, query string, cfg *types.SearchConfig) (*types.SearchResults, error) {
	m, err := a.r.Search(ctx, query, cfg)
	if err != nil {
		return nil, err
	}

	// Backfill node provenance from Sources when GroupID is empty so that
	// downstream consumers can still tell which topic graph a node came from.
	for _, n := range m.Nodes {
		if n == nil || n.GroupID != "" {
			continue
		}
		if srcs, ok := m.Sources[n.Uuid]; ok && len(srcs) > 0 {
			n.GroupID = srcs[0]
		}
	}

	return &types.SearchResults{
		Query: query,
		Nodes: m.Nodes,
		Edges: m.Edges,
		Total: m.Total,
	}, nil
}
