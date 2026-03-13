package router

import (
	"context"
	"fmt"

	"github.com/soundprediction/predicato/pkg/driver"
)

// GraphDbType represents the type of graph database.
type GraphDbType string

const (
	GraphDbTypeDuckPGQ GraphDbType = "duckpgq"
	GraphDbTypeLadybug GraphDbType = "ladybug"
)

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
	case GraphDbTypeLadybug:
		cfg := driver.DefaultLadybugDriverConfig()
		cfg.DBPath = dbPath
		cfg.ReadOnly = c.ReadOnly
		if c.ReadOnly {
			// Use a smaller buffer pool for read-only topic graphs to allow
			// many DBs to be open simultaneously (default 1GB is too much × N DBs).
			cfg.BufferPoolSize = 256 * 1024 * 1024 // 256MB
		}
		return driver.NewLadybugDriverWithConfig(cfg)
	default:
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
		return 3
	}
	return r.MaxGraphs
}

// GetMergeStrategyOrDefault returns the merge strategy or the default value.
func (r *RouterConfig) GetMergeStrategyOrDefault() string {
	if r.MergeStrategy == "" {
		return "rrf"
	}
	return r.MergeStrategy
}
