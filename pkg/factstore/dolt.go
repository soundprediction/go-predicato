package factstore

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"

	_ "github.com/dolthub/driver"
)

// DoltDB implements FactsDB using a Dolt SQL database.
type DoltDB struct {
	db *sql.DB
}

// NewDoltDB creates a new DoltDB instance.
// connectionString should be a valid Dolt DSN, e.g., "file:///path/to/databases?commitname=User&commitemail=user@example.com&database=mydb"
func NewDoltDB(connectionString string) (*DoltDB, error) {
	db, err := sql.Open("dolt", connectionString)
	if err != nil {
		return nil, fmt.Errorf("failed to open database connection: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return &DoltDB{db: db}, nil
}

func (d *DoltDB) Initialize(ctx context.Context) error {
	// Ensure the database exists and we are using it
	if _, err := d.db.ExecContext(ctx, "CREATE DATABASE IF NOT EXISTS facts"); err != nil {
		return fmt.Errorf("failed to create database 'facts': %w", err)
	}
	if _, err := d.db.ExecContext(ctx, "USE facts"); err != nil {
		return fmt.Errorf("failed to use database 'facts': %w", err)
	}

	queries := []string{
		`CREATE TABLE IF NOT EXISTS sources (
			id VARCHAR(255) PRIMARY KEY,
			name TEXT,
			content TEXT,
			group_id VARCHAR(255),
			metadata JSON,
			created_at TIMESTAMP
		)`,
		// Table for identified entities
		`CREATE TABLE IF NOT EXISTS extracted_entities (
			id VARCHAR(255) PRIMARY KEY,
			source_id VARCHAR(255),
			name TEXT,
			entity_type VARCHAR(100),
			description TEXT,
			embedding JSON,
			chunk_index INT,
			FOREIGN KEY (source_id) REFERENCES sources(id)
		)`,
		// Table for other structural/generic nodes
		`CREATE TABLE IF NOT EXISTS extracted_other_nodes (
			id VARCHAR(255) PRIMARY KEY,
			source_id VARCHAR(255),
			name TEXT,
			type VARCHAR(50),
			description TEXT,
			embedding JSON,
			chunk_index INT,
			FOREIGN KEY (source_id) REFERENCES sources(id)
		)`,
		`CREATE TABLE IF NOT EXISTS extracted_edges (
			id VARCHAR(255) PRIMARY KEY,
			source_id VARCHAR(255),
			source_node_name TEXT,
			target_node_name TEXT,
			relation TEXT,
			description TEXT,
			weight FLOAT,
			chunk_index INT,
			FOREIGN KEY (source_id) REFERENCES sources(id)
		)`,
	}

	for _, query := range queries {
		if _, err := d.db.ExecContext(ctx, query); err != nil {
			return fmt.Errorf("failed to execute init query: %w", err)
		}
	}
	return nil
}

func (d *DoltDB) SaveSource(ctx context.Context, source *Source) error {
	metadataJSON, err := json.Marshal(source.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	query := `INSERT INTO sources (id, name, content, group_id, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)`
	_, err = d.db.ExecContext(ctx, query, source.ID, source.Name, source.Content, source.GroupID, metadataJSON, source.CreatedAt)
	if err != nil {
		return fmt.Errorf("failed to insert source: %w", err)
	}
	return nil
}

func (d *DoltDB) SaveExtractedKnowledge(ctx context.Context, sourceID string, nodes []*ExtractedNode, edges []*ExtractedEdge) error {
	tx, err := d.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	// Prepare statements for both tables
	entityStmt, err := tx.PrepareContext(ctx, `INSERT INTO extracted_entities (id, source_id, name, entity_type, description, embedding, chunk_index) VALUES (?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return fmt.Errorf("failed to prepare entity statement: %w", err)
	}
	defer entityStmt.Close()

	otherStmt, err := tx.PrepareContext(ctx, `INSERT INTO extracted_other_nodes (id, source_id, name, type, description, embedding, chunk_index) VALUES (?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return fmt.Errorf("failed to prepare other node statement: %w", err)
	}
	defer otherStmt.Close()

	// Helper to check existence
	nodeExists := func(table string, name string, typeOrCategory string, isEntity bool) (bool, error) {
		var exists bool
		var query string
		if isEntity {
			// For entities, we check name (and maybe entity_type, but name usually key)
			query = fmt.Sprintf("SELECT EXISTS(SELECT 1 FROM %s WHERE source_id = ? AND name = ?)", table)
			err := tx.QueryRowContext(ctx, query, sourceID, name).Scan(&exists)
			return exists, err
		} else {
			// For others, check type as well
			query = fmt.Sprintf("SELECT EXISTS(SELECT 1 FROM %s WHERE source_id = ? AND name = ? AND type = ?)", table)
			err := tx.QueryRowContext(ctx, query, sourceID, name, typeOrCategory).Scan(&exists)
			return exists, err
		}
	}

	for _, node := range nodes {
		embeddingJSON, err := json.Marshal(node.Embedding)
		if err != nil {
			return fmt.Errorf("failed to marshal embedding for node %s: %w", node.ID, err)
		}

		// Determine target table and logic
		if node.Type == "entity" || node.Type == "Entity" {
			// Check existence in extracted_entities
			exists, err := nodeExists("extracted_entities", node.Name, "", true)
			if err != nil {
				return fmt.Errorf("failed to check existence for entity %s: %w", node.Name, err)
			}
			if exists {
				continue // Skip duplicate
			}

			if _, err := entityStmt.ExecContext(ctx, node.ID, sourceID, node.Name, node.EntityType, node.Description, embeddingJSON, node.ChunkIndex); err != nil {
				return fmt.Errorf("failed to insert entity %s: %w", node.ID, err)
			}
		} else {
			// Check existence in extracted_other_nodes
			exists, err := nodeExists("extracted_other_nodes", node.Name, node.Type, false)
			if err != nil {
				return fmt.Errorf("failed to check existence for node %s: %w", node.Name, err)
			}
			if exists {
				continue // Skip duplicate
			}

			if _, err := otherStmt.ExecContext(ctx, node.ID, sourceID, node.Name, node.Type, node.Description, embeddingJSON, node.ChunkIndex); err != nil {
				return fmt.Errorf("failed to insert other node %s: %w", node.ID, err)
			}
		}
	}

	edgeStmt, err := tx.PrepareContext(ctx, `INSERT INTO extracted_edges (id, source_id, source_node_name, target_node_name, relation, description, weight, chunk_index) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return fmt.Errorf("failed to prepare edge statement: %w", err)
	}
	defer edgeStmt.Close()

	for _, edge := range edges {
		// Deduping edges? User didn't strictly ask, but good practice.
		// For now, let's just insert as edges are often redundant or many-to-many.
		// Or check existence? Let's keep it simple and insert.
		if _, err := edgeStmt.ExecContext(ctx, edge.ID, sourceID, edge.SourceNodeName, edge.TargetNodeName, edge.Relation, edge.Description, edge.Weight, edge.ChunkIndex); err != nil {
			return fmt.Errorf("failed to insert edge %s: %w", edge.ID, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

func (d *DoltDB) GetSource(ctx context.Context, sourceID string) (*Source, error) {
	row := d.db.QueryRowContext(ctx, "SELECT id, name, content, group_id, metadata, created_at FROM sources WHERE id = ?", sourceID)

	var s Source
	var metadataBytes []byte

	if err := row.Scan(&s.ID, &s.Name, &s.Content, &s.GroupID, &metadataBytes, &s.CreatedAt); err != nil {
		return nil, fmt.Errorf("failed to scan source: %w", err)
	}

	if len(metadataBytes) > 0 {
		if err := json.Unmarshal(metadataBytes, &s.Metadata); err != nil {
			return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
		}
	}

	return &s, nil
}

func (d *DoltDB) GetExtractedNodes(ctx context.Context, sourceID string) ([]*ExtractedNode, error) {
	// Union query to get all nodes
	query := `
		SELECT id, source_id, name, 'entity' as type, entity_type, description, embedding, chunk_index 
		FROM extracted_entities 
		WHERE source_id = ?
		UNION ALL
		SELECT id, source_id, name, type, '' as entity_type, description, embedding, chunk_index 
		FROM extracted_other_nodes 
		WHERE source_id = ?
	`
	rows, err := d.db.QueryContext(ctx, query, sourceID, sourceID)
	if err != nil {
		return nil, fmt.Errorf("failed to query extracted nodes: %w", err)
	}
	defer rows.Close()

	var nodes []*ExtractedNode
	for rows.Next() {
		var n ExtractedNode
		var embeddingBytes []byte
		var entityType sql.NullString // Handle potential nulls or mismatched columns safely?
		// Actually UNION should align columns. entity_type is string in first, '' in second.

		if err := rows.Scan(&n.ID, &n.SourceID, &n.Name, &n.Type, &entityType, &n.Description, &embeddingBytes, &n.ChunkIndex); err != nil {
			return nil, err
		}
		if entityType.Valid {
			n.EntityType = entityType.String
		}
		if len(embeddingBytes) > 0 {
			if err := json.Unmarshal(embeddingBytes, &n.Embedding); err != nil {
				return nil, fmt.Errorf("failed to unmarshal embedding: %w", err)
			}
		}
		nodes = append(nodes, &n)
	}
	return nodes, nil
}

func (d *DoltDB) GetExtractedEdges(ctx context.Context, sourceID string) ([]*ExtractedEdge, error) {
	rows, err := d.db.QueryContext(ctx, "SELECT id, source_id, source_node_name, target_node_name, relation, description, weight, chunk_index FROM extracted_edges WHERE source_id = ?", sourceID)
	if err != nil {
		return nil, fmt.Errorf("failed to query extracted edges: %w", err)
	}
	defer rows.Close()

	var edges []*ExtractedEdge
	for rows.Next() {
		var e ExtractedEdge
		if err := rows.Scan(&e.ID, &e.SourceID, &e.SourceNodeName, &e.TargetNodeName, &e.Relation, &e.Description, &e.Weight, &e.ChunkIndex); err != nil {
			return nil, err
		}
		edges = append(edges, &e)
	}
	return edges, nil
}

func (d *DoltDB) GetAllSources(ctx context.Context, limit int) ([]*Source, error) {
	query := "SELECT id, name, content, group_id, metadata, created_at FROM sources ORDER BY created_at DESC"
	var rows *sql.Rows
	var err error
	if limit > 0 {
		query += " LIMIT ?"
		rows, err = d.db.QueryContext(ctx, query, limit)
	} else {
		rows, err = d.db.QueryContext(ctx, query)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to query sources: %w", err)
	}
	defer rows.Close()

	var sources []*Source
	for rows.Next() {
		var s Source
		var metadataBytes []byte
		if err := rows.Scan(&s.ID, &s.Name, &s.Content, &s.GroupID, &metadataBytes, &s.CreatedAt); err != nil {
			return nil, err
		}
		if len(metadataBytes) > 0 {
			if err := json.Unmarshal(metadataBytes, &s.Metadata); err != nil {
				return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
			}
		}
		sources = append(sources, &s)
	}
	return sources, nil
}

func (d *DoltDB) GetAllNodes(ctx context.Context, limit int) ([]*ExtractedNode, error) {
	query := `
		SELECT id, source_id, name, 'entity' as type, entity_type, description, embedding, chunk_index 
		FROM extracted_entities
		UNION ALL
		SELECT id, source_id, name, type, '' as entity_type, description, embedding, chunk_index 
		FROM extracted_other_nodes
	`
	if limit > 0 {
		query += fmt.Sprintf(" LIMIT %d", limit)
	}

	rows, err := d.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to query nodes: %w", err)
	}
	defer rows.Close()

	var nodes []*ExtractedNode
	for rows.Next() {
		var n ExtractedNode
		var embeddingBytes []byte
		var entityType sql.NullString
		if err := rows.Scan(&n.ID, &n.SourceID, &n.Name, &n.Type, &entityType, &n.Description, &embeddingBytes, &n.ChunkIndex); err != nil {
			return nil, err
		}
		if entityType.Valid {
			n.EntityType = entityType.String
		}
		if len(embeddingBytes) > 0 {
			if err := json.Unmarshal(embeddingBytes, &n.Embedding); err != nil {
				return nil, fmt.Errorf("failed to unmarshal embedding: %w", err)
			}
		}
		nodes = append(nodes, &n)
	}
	return nodes, nil
}

func (d *DoltDB) GetAllEdges(ctx context.Context, limit int) ([]*ExtractedEdge, error) {
	query := "SELECT id, source_id, source_node_name, target_node_name, relation, description, weight, chunk_index FROM extracted_edges"
	var rows *sql.Rows
	var err error
	if limit > 0 {
		query += " LIMIT ?"
		rows, err = d.db.QueryContext(ctx, query, limit)
	} else {
		rows, err = d.db.QueryContext(ctx, query)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to query edges: %w", err)
	}
	defer rows.Close()

	var edges []*ExtractedEdge
	for rows.Next() {
		var e ExtractedEdge
		if err := rows.Scan(&e.ID, &e.SourceID, &e.SourceNodeName, &e.TargetNodeName, &e.Relation, &e.Description, &e.Weight, &e.ChunkIndex); err != nil {
			return nil, err
		}
		edges = append(edges, &e)
	}
	return edges, nil
}

func (d *DoltDB) GetStats(ctx context.Context) (*Stats, error) {
	stats := &Stats{}

	if err := d.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM sources").Scan(&stats.SourceCount); err != nil {
		return nil, fmt.Errorf("failed to count sources: %w", err)
	}

	var entityCount, otherCount int64
	if err := d.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM extracted_entities").Scan(&entityCount); err != nil {
		// Log error or ignore if table doesn't exist yet? (Should exist after Init)
		// return nil, fmt.Errorf("failed to count entities: %w", err)
		// For robustness, maybe 0 if missing? But Initialize runs on startup.
	}
	if err := d.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM extracted_other_nodes").Scan(&otherCount); err != nil {

	}
	stats.NodeCount = entityCount + otherCount

	if err := d.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM extracted_edges").Scan(&stats.EdgeCount); err != nil {
		return nil, fmt.Errorf("failed to count edges: %w", err)
	}

	return stats, nil
}

func (d *DoltDB) Close() error {
	return d.db.Close()
}
