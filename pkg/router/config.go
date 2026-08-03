package router

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/soundprediction/predicato/pkg/driver"
)

// GraphDbType represents the type of graph database.
type GraphDbType string

const (
	GraphDbTypeDuckPGQ GraphDbType = "duckpgq"
	GraphDbTypeLadybug GraphDbType = "ladybug"
)

// DriverFactory creates a GraphDriver from client config, db path, and embedding dimensions.
type DriverFactory func(c *ClientConfig, dbPath string, embDim int) (driver.GraphDriver, error)

// driverFactories holds registered driver factories for build-tag-gated backends.
var driverFactories = map[GraphDbType]DriverFactory{}

// RegisterDriverFactory registers a driver factory for a graph database type.
// Called from build-tagged init() functions (e.g., driver_ladybug.go).
func RegisterDriverFactory(dbType GraphDbType, factory DriverFactory) {
	driverFactories[dbType] = factory
}

// ClientConfig defines configuration for a single predicato client.
type ClientConfig struct {
	// GraphDb is the graph database configuration map.
	// Must contain "type" (string) and "db_path" (string) at minimum.
	GraphDb map[string]any `json:"graph_db"`

	// Name is the unique identifier for this client (required).
	Name string `json:"name"`

	// GroupID is the predicato group ID for data isolation.
	GroupID string `json:"group_id"`

	// Description is a human-readable summary of what this knowledge base contains.
	Description string `json:"description"`

	// Topics are keywords used for topic-based routing.
	Topics []string `json:"topics"`

	// Default marks this client as the default (used when no specific client is requested).
	Default bool `json:"default"`

	// Fallback marks this client as fallback (queried when no topic matches).
	Fallback bool `json:"fallback"`

	// ExcludeFromSecondary prevents this client from being selected as a
	// secondary (2nd, 3rd, …) graph. It can still be the primary pick.
	ExcludeFromSecondary bool `json:"exclude_from_secondary"`

	// ReadOnly opens the database in read-only mode.
	// Topic graph databases should be read-only; user-specific DBs should be read-write.
	ReadOnly bool `json:"read_only"`
}

// Validate checks if the client config has all required fields.
func (c *ClientConfig) Validate() error {
	if c.Name == "" {
		return fmt.Errorf("client name is required")
	}
	if c.GraphDb == nil {
		return fmt.Errorf("graph_db configuration is required for client %q", c.Name)
	}
	return nil
}

// CreateDriver creates a GraphDriver from the client's GraphDb map.
func (c *ClientConfig) CreateDriver(ctx context.Context) (driver.GraphDriver, error) {
	dbType, ok := c.GraphDb["type"].(string)
	if !ok {
		return nil, fmt.Errorf("graph_db type is required")
	}

	dbPath, _ := c.GraphDb["db_path"].(string)
	if dbPath == "" {
		return nil, fmt.Errorf("graph_db db_path is required")
	}

	embDim := 1024
	if dim, ok := c.GraphDb["embedding_dim"].(int); ok && dim > 0 {
		embDim = dim
	} else if dim, ok := c.GraphDb["embedding_dim"].(int64); ok && dim > 0 {
		embDim = int(dim)
	}

	switch GraphDbType(dbType) {
	case GraphDbTypeDuckPGQ:
		return driver.NewDuckPGQDriverWithConfig(driver.DuckPGQDriverConfig{
			URI:          dbPath,
			EmbeddingDim: embDim,
			ReadOnly:     c.ReadOnly,
		})
	default:
		// Check platform-specific drivers (e.g., Ladybug behind build tags)
		if factory, ok := driverFactories[GraphDbType(dbType)]; ok {
			return factory(c, dbPath, embDim)
		}
		return nil, fmt.Errorf("unsupported graph database type: %s", dbType)
	}
}

// RouterConfig defines configuration for the multi-client router.
type RouterConfig struct {
	// Strategy is the routing algorithm: "topic_based", "keyword", "embedding", "cross_encoder"
	Strategy string `json:"strategy"`

	// MergeStrategy is the algorithm for merging results: "rrf", "simple_concat", "max_score"
	MergeStrategy string `json:"merge_strategy"`

	// MinConfidence is the minimum confidence score to include a client in routing.
	MinConfidence float64 `json:"min_confidence"`

	// MaxGraphs is the maximum number of graphs to query per request.
	MaxGraphs int `json:"max_graphs"`
}

// GetStrategyOrDefault returns the strategy or the default value.
func (r *RouterConfig) GetStrategyOrDefault() string {
	if r.Strategy == "" {
		return "topic_based"
	}
	return r.Strategy
}

// GetMinConfidenceOrDefault returns the min confidence or the default value.
func (r *RouterConfig) GetMinConfidenceOrDefault() float64 {
	if r.MinConfidence == 0 {
		return 0.3
	}
	return r.MinConfidence
}

// GetMaxGraphsOrDefault returns the max graphs or the default value.
func (r *RouterConfig) GetMaxGraphsOrDefault() int {
	if r.MaxGraphs == 0 {
		return 2
	}
	return r.MaxGraphs
}

// defaultTopicBufferPoolBytes is the per-graph buffer pool for a read-only topic
// graph. The previous size ladder (768 MB / 1 GB / 2 GB by file size) was sized
// for a handful of open graphs; across a whole topic corpus it asks for tens of
// gigabytes and is the reason the open-graph cap had to stay small. These graphs
// are read-only and memory-mapped, so the OS page cache — shared across all of
// them and evictable under pressure — does the caching that matters, and a large
// private pool per graph mostly buys address space and accounting.
//
// Override with PREDICATO_TOPIC_BUFFER_POOL_BYTES (the name humn already
// documents) when a specific deployment's working set justifies more.
const defaultTopicBufferPoolBytes uint64 = 256 * 1024 * 1024

// defaultTopicMaxDbSize bounds the VIRTUAL address space each open graph
// reserves via mmap. See driver_ladybug.go for why this must not be left at the
// driver's 8 TiB default. Override with PREDICATO_TOPIC_MAX_DB_SIZE_BYTES.
const defaultTopicMaxDbSize uint64 = 64 * 1024 * 1024 * 1024

// topicBufferPoolBytes returns the per-graph buffer pool size. It no longer
// scales with file size: with a whole corpus open at once the ladder's totals
// were the binding constraint, and the page cache serves the same purpose.
func topicBufferPoolBytes() uint64 {
	if v := envBytes("PREDICATO_TOPIC_BUFFER_POOL_BYTES"); v > 0 {
		return v
	}
	return defaultTopicBufferPoolBytes
}

// defaultTopicQueryThreads is the intra-query thread count for a read-only topic
// graph. Kept well below the host core count because searches fan out (max-graphs
// per request, and callers issue several concurrently), so this multiplies rather
// than replaces existing parallelism.
const defaultTopicQueryThreads = 4

// topicQueryThreads returns the per-query thread count, overridable via
// PREDICATO_TOPIC_QUERY_THREADS.
func topicQueryThreads() int {
	if v := envBytes("PREDICATO_TOPIC_QUERY_THREADS"); v > 0 && v <= 64 {
		return int(v)
	}
	return defaultTopicQueryThreads
}

// defaultTopicReadPoolSize is how many connections each read-only topic graph
// opens for concurrent reads. Matches the default query-thread count so a graph
// can service a full fan-out without queueing.
const defaultTopicReadPoolSize = 4

// topicReadPoolSize returns the per-graph read connection count, overridable via
// PREDICATO_TOPIC_READ_POOL.
func topicReadPoolSize() int {
	if v := envBytes("PREDICATO_TOPIC_READ_POOL"); v > 0 && v <= 32 {
		return int(v)
	}
	return defaultTopicReadPoolSize
}

// maxDbSizeForTopicGraph returns the per-graph virtual-size reservation.
func maxDbSizeForTopicGraph() uint64 {
	if v := envBytes("PREDICATO_TOPIC_MAX_DB_SIZE_BYTES"); v > 0 {
		return v
	}
	return defaultTopicMaxDbSize
}

// envBytes reads a byte count from the environment, returning 0 when unset or
// unparseable so the caller falls back to its default.
func envBytes(name string) uint64 {
	s := strings.TrimSpace(os.Getenv(name))
	if s == "" {
		return 0
	}
	v, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		return 0
	}
	return v
}

// GetMergeStrategyOrDefault returns the merge strategy or the default value.
func (r *RouterConfig) GetMergeStrategyOrDefault() string {
	if r.MergeStrategy == "" {
		return "rrf"
	}
	return r.MergeStrategy
}
