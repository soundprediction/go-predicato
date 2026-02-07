// Package factstore provides persistent storage for extracted facts and entities.
//
// This package implements the FactsDB interface for storing and searching
// extracted nodes, edges, and sources from the knowledge graph processing.
//
// # Supported Backends
//
// The following storage backends are supported:
//   - PostgresDB: PostgreSQL with optional VectorChord extension for vector search
//   - DoltDB: Dolt SQL database (deprecated, use PostgresDB with DoltGres)
//   - MySQL, SQLite, DuckDB: via migration SQL files (see migrations/ directory)
//
// # Usage
//
// Create from config:
//
//	config := &factstore.DbConfig{
//	    Type:             "postgres",
//	    ConnectionString: "postgres://user:pass@localhost:5432/facts",
//	}
//	db, err := factstore.NewFactsDB(config)
//	if err != nil {
//	    return err
//	}
//	defer db.Close()
//
// Or pass a pre-built connection:
//
//	config := &factstore.DbConfig{
//	    DB: factstore.NewPostgresDBFromConn(existingDB, 1024),
//	}
//	db, err := factstore.NewFactsDB(config)
//
// # Search Capabilities
//
// The package provides hybrid search combining:
//   - Vector similarity search (using embeddings)
//   - Keyword/fulltext search
//   - Reciprocal Rank Fusion (RRF) for result merging
//
// # Vector Search
//
// For best performance, use PostgreSQL with VectorChord extension.
// Without VectorChord (e.g., DoltGres), vector search falls back to
// in-memory cosine similarity calculation.
package factstore
