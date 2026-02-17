package driver

import (
	"context"

	"github.com/soundprediction/predicato/pkg/types"
)

// GraphExporter is an optional interface that drivers can implement to support
// streaming iteration over all nodes and edges. This is used by the graph
// conversion tool to efficiently migrate data between drivers.
//
// Drivers that do not implement this interface can still be used as conversion
// sources via the fallback path: GetAllGroupIDs + GetEntityNodesByGroup.
type GraphExporter interface {
	// IterateNodes calls fn for each node in the specified group.
	// If groupID is empty, iterates over all groups.
	// Iteration stops early if fn returns a non-nil error.
	IterateNodes(ctx context.Context, groupID string, fn func(*types.Node) error) error

	// IterateEdges calls fn for each edge in the specified group.
	// If groupID is empty, iterates over all groups.
	// Iteration stops early if fn returns a non-nil error.
	IterateEdges(ctx context.Context, groupID string, fn func(*types.Edge) error) error
}
