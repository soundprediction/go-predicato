//go:build cgo

// Package factstore provides fact storage backends.
package factstore

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
)

const (
	// DefaultDoltDataPath is the default path for embedded Dolt data.
	DefaultDoltDataPath = ".predicato/factstore"
)

// EmbeddedFactStoreConfig configures the embedded factstore.
type EmbeddedFactStoreConfig struct {
	// DataPath is the path to store the database files.
	// If empty, uses ~/.predicato/factstore
	DataPath string
	// EmbeddingDimensions is the vector dimension (e.g., 1024 for qwen3-embedding)
	EmbeddingDimensions int
}

// NewEmbeddedFactsDB creates a FactsDB using embedded Dolt storage.
// This provides persistent fact storage without requiring an external database.
func NewEmbeddedFactsDB(cfg *EmbeddedFactStoreConfig) (FactsDB, error) {
	if cfg == nil {
		cfg = &EmbeddedFactStoreConfig{}
	}

	// Set defaults
	if cfg.DataPath == "" {
		homeDir, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("failed to get home directory: %w", err)
		}
		cfg.DataPath = filepath.Join(homeDir, DefaultDoltDataPath)
	}

	if cfg.EmbeddingDimensions <= 0 {
		cfg.EmbeddingDimensions = 1024 // Default for qwen3-embedding
	}

	// Ensure data directory exists
	if err := os.MkdirAll(cfg.DataPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create data directory: %w", err)
	}

	// Build the Dolt connection string
	// Format: file:///path/to/databases?commitname=User&commitemail=user@example.com&database=facts
	connString := fmt.Sprintf(
		"file://%s?commitname=predicato&commitemail=predicato@localhost&database=facts",
		cfg.DataPath,
	)

	// Create the embedded Dolt database
	db, err := NewDoltDB(connString)
	if err != nil {
		return nil, fmt.Errorf("failed to create embedded Dolt database: %w", err)
	}

	// Initialize the database schema
	ctx := context.Background()
	if err := db.Initialize(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to initialize factstore schema: %w", err)
	}

	return db, nil
}

// DataPath returns the default data path for the embedded factstore.
func DefaultDataPath() (string, error) {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get home directory: %w", err)
	}
	return filepath.Join(homeDir, DefaultDoltDataPath), nil
}
