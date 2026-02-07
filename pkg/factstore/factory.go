package factstore

import (
	"fmt"
)

// NewFactsDB creates a new FactsDB instance based on the configuration.
// If config.DB is set, it is returned directly (no new connection is opened).
// Otherwise a new connection is created based on config.Type.
func NewFactsDB(config *DbConfig) (FactsDB, error) {
	if config == nil {
		return nil, fmt.Errorf("storage config is required")
	}

	// If a pre-built DB is provided, use it directly
	if config.DB != nil {
		return config.DB, nil
	}

	// Default values
	if config.EmbeddingDimensions <= 0 {
		config.EmbeddingDimensions = 1024 // Default for qwen3-embedding
	}

	if config.ConnectionString == "" {
		return nil, fmt.Errorf("connection string is required")
	}

	switch config.Type {
	case DbTypePostgres:
		return NewPostgresDB(config.ConnectionString, config.EmbeddingDimensions)

	case DbTypeDoltGres, "":
		return NewDoltGresDB(config.ConnectionString, config.EmbeddingDimensions)

	default:
		return nil, fmt.Errorf("unsupported storage type: %s (supported: postgres, doltgres, mysql, sqlite, duckdb)", config.Type)
	}
}

// NewFactsDBFromURL creates a FactsDB from a simple connection URL.
// This is a convenience function for quick setup with external PostgreSQL + VectorChord.
func NewFactsDBFromURL(connectionURL string, embeddingDimensions int) (FactsDB, error) {
	return NewFactsDB(&DbConfig{
		Type:                DbTypePostgres,
		ConnectionString:    connectionURL,
		EmbeddingDimensions: embeddingDimensions,
	})
}

// NewDoltGresFactsDB creates a FactsDB for DoltGres (without VectorChord).
func NewDoltGresFactsDB(connectionURL string, embeddingDimensions int) (FactsDB, error) {
	return NewFactsDB(&DbConfig{
		Type:                DbTypeDoltGres,
		ConnectionString:    connectionURL,
		EmbeddingDimensions: embeddingDimensions,
	})
}
