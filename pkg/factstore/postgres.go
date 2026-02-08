package factstore

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"sort"
	"strconv"
	"strings"
	"time"

	_ "github.com/lib/pq"
	"github.com/soundprediction/predicato/pkg/utils"
)

// PostgresDB implements FactsDB using PostgreSQL with VectorChord extension.
// For external PostgreSQL: uses VectorChord for native vector search
// For DoltGres: uses in-memory vector search (VectorChord not available)
type PostgresDB struct {
	db                  *sql.DB
	embeddingDimensions int
	usePgVector         bool // true for PostgreSQL with VectorChord, false for DoltGres
}

// PostgresDBConfig holds configuration options for PostgresDB connection pool.
type PostgresDBConfig struct {
	// MaxOpenConns is the maximum number of open connections to the database.
	// Default: 25
	MaxOpenConns int

	// MaxIdleConns is the maximum number of connections in the idle connection pool.
	// Default: 5
	MaxIdleConns int

	// ConnMaxLifetime is the maximum amount of time a connection may be reused.
	// Default: 5 minutes
	ConnMaxLifetime time.Duration
}

// DefaultPostgresDBConfig returns the default PostgresDB configuration.
func DefaultPostgresDBConfig() *PostgresDBConfig {
	return &PostgresDBConfig{
		MaxOpenConns:    25,
		MaxIdleConns:    5,
		ConnMaxLifetime: 5 * time.Minute,
	}
}

// NewPostgresDB creates a new PostgresDB instance for external PostgreSQL with VectorChord.
// connectionString should be a valid PostgreSQL DSN, e.g.:
// "postgres://user:password@localhost:5432/dbname?sslmode=disable"
func NewPostgresDB(connectionString string, embeddingDimensions int) (*PostgresDB, error) {
	return NewPostgresDBWithConfig(connectionString, embeddingDimensions, true, nil)
}

// NewPostgresDBWithConfig creates a new PostgresDB instance with custom configuration.
// If config is nil, default configuration values are used.
func NewPostgresDBWithConfig(connectionString string, embeddingDimensions int, usePgVector bool, config *PostgresDBConfig) (*PostgresDB, error) {
	if embeddingDimensions <= 0 {
		embeddingDimensions = 1024 // Default for qwen3-embedding
	}

	if config == nil {
		config = DefaultPostgresDBConfig()
	}

	db, err := sql.Open("postgres", connectionString)
	if err != nil {
		return nil, fmt.Errorf("failed to open database connection: %w", err)
	}

	// Configure connection pool
	db.SetMaxOpenConns(config.MaxOpenConns)
	db.SetMaxIdleConns(config.MaxIdleConns)
	db.SetConnMaxLifetime(config.ConnMaxLifetime)

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	return &PostgresDB{
		db:                  db,
		embeddingDimensions: embeddingDimensions,
		usePgVector:         usePgVector,
	}, nil
}

// NewPostgresDBFromConn creates a PostgresDB that wraps an existing *sql.DB connection.
// Use this to share a connection pool with the rest of your application instead of
// opening a separate one. The caller retains ownership of the connection and is
// responsible for closing it.
func NewPostgresDBFromConn(db *sql.DB, embeddingDimensions int) *PostgresDB {
	if embeddingDimensions <= 0 {
		embeddingDimensions = 1024
	}
	return &PostgresDB{
		db:                  db,
		embeddingDimensions: embeddingDimensions,
		usePgVector:         true,
	}
}

// NewDoltGresDB creates a new PostgresDB instance for DoltGres (without VectorChord).
// Uses in-memory vector search since DoltGres doesn't support VectorChord extension.
// connectionString should be a valid PostgreSQL DSN for DoltGres server.
func NewDoltGresDB(connectionString string, embeddingDimensions int) (*PostgresDB, error) {
	return NewPostgresDBWithConfig(connectionString, embeddingDimensions, false, nil)
}

// NewDoltGresDBWithConfig creates a new PostgresDB instance for DoltGres with custom configuration.
// If config is nil, default configuration values are used.
func NewDoltGresDBWithConfig(connectionString string, embeddingDimensions int, config *PostgresDBConfig) (*PostgresDB, error) {
	return NewPostgresDBWithConfig(connectionString, embeddingDimensions, false, config)
}

func (p *PostgresDB) Initialize(ctx context.Context) error {
	// Enable VectorChord extension only for PostgreSQL (not DoltGres)
	if p.usePgVector {
		if _, err := p.db.ExecContext(ctx, "CREATE EXTENSION IF NOT EXISTS vector"); err != nil {
			return fmt.Errorf("failed to create vector extension: %w", err)
		}
	}

	// Create sources table
	sourcesTable := `
		CREATE TABLE IF NOT EXISTS sources (
			id VARCHAR(255) PRIMARY KEY,
			name TEXT,
			content TEXT,
			group_id VARCHAR(255),
			metadata JSONB,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`
	if _, err := p.db.ExecContext(ctx, sourcesTable); err != nil {
		return fmt.Errorf("failed to create sources table: %w", err)
	}

	// Create extracted_nodes table
	// Use vector type for PostgreSQL with VectorChord, JSONB for DoltGres
	var nodesTable string
	if p.usePgVector {
		nodesTable = fmt.Sprintf(`
			CREATE TABLE IF NOT EXISTS extracted_nodes (
				id VARCHAR(255) PRIMARY KEY,
				source_id VARCHAR(255) REFERENCES sources(id),
				group_id VARCHAR(255),
				name TEXT,
				type VARCHAR(50),
				description TEXT,
				embedding vector(%d),
				chunk_index INT,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)`, p.embeddingDimensions)
	} else {
		nodesTable = `
			CREATE TABLE IF NOT EXISTS extracted_nodes (
				id VARCHAR(255) PRIMARY KEY,
				source_id VARCHAR(255) REFERENCES sources(id),
				group_id VARCHAR(255),
				name TEXT,
				type VARCHAR(50),
				description TEXT,
				embedding JSONB,
				chunk_index INT,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)`
	}
	if _, err := p.db.ExecContext(ctx, nodesTable); err != nil {
		return fmt.Errorf("failed to create extracted_nodes table: %w", err)
	}

	// Create extracted_edges table
	var edgesTable string
	if p.usePgVector {
		edgesTable = fmt.Sprintf(`
			CREATE TABLE IF NOT EXISTS extracted_edges (
				id VARCHAR(255) PRIMARY KEY,
				source_id VARCHAR(255) REFERENCES sources(id),
				group_id VARCHAR(255),
				source_node_name TEXT,
				source_node_type VARCHAR(50),
				target_node_name TEXT,
				target_node_type VARCHAR(50),
				relation TEXT,
				description TEXT,
				embedding vector(%d),
				weight FLOAT,
				chunk_index INT,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)`, p.embeddingDimensions)
	} else {
		edgesTable = `
			CREATE TABLE IF NOT EXISTS extracted_edges (
				id VARCHAR(255) PRIMARY KEY,
				source_id VARCHAR(255) REFERENCES sources(id),
				group_id VARCHAR(255),
				source_node_name TEXT,
				source_node_type VARCHAR(50),
				target_node_name TEXT,
				target_node_type VARCHAR(50),
				relation TEXT,
				description TEXT,
				embedding JSONB,
				weight FLOAT,
				chunk_index INT,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)`
	}
	if _, err := p.db.ExecContext(ctx, edgesTable); err != nil {
		return fmt.Errorf("failed to create extracted_edges table: %w", err)
	}

	// Create indices for better query performance
	indices := []string{
		"CREATE INDEX IF NOT EXISTS idx_sources_group ON sources(group_id)",
		"CREATE INDEX IF NOT EXISTS idx_nodes_source ON extracted_nodes(source_id)",
		"CREATE INDEX IF NOT EXISTS idx_nodes_group ON extracted_nodes(group_id)",
		"CREATE INDEX IF NOT EXISTS idx_nodes_type ON extracted_nodes(type)",
		"CREATE INDEX IF NOT EXISTS idx_edges_source ON extracted_edges(source_id)",
		"CREATE INDEX IF NOT EXISTS idx_edges_group ON extracted_edges(group_id)",
		"CREATE INDEX IF NOT EXISTS idx_triples_source ON extracted_triples(source_id)",
		"CREATE INDEX IF NOT EXISTS idx_rules_source ON extracted_rules(source_id)",
	}

	for _, idx := range indices {
		if _, err := p.db.ExecContext(ctx, idx); err != nil {
			// Log warning but don't fail - indices are optional
			fmt.Printf("Warning: failed to create index: %v\n", err)
		}
	}

	// Create GIN indices for full-text search (keyword search performance)
	// These are optional but significantly improve keyword search performance
	ginIndices := []string{
		`CREATE INDEX IF NOT EXISTS idx_nodes_fts ON extracted_nodes 
		 USING GIN (to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(description, '')))`,
		`CREATE INDEX IF NOT EXISTS idx_edges_fts ON extracted_edges 
		 USING GIN (to_tsvector('english', COALESCE(relation, '') || ' ' || COALESCE(description, '')))`,
		`CREATE INDEX IF NOT EXISTS idx_sources_fts ON sources 
		 USING GIN (to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(content, '')))`,
	}

	for _, idx := range ginIndices {
		if _, err := p.db.ExecContext(ctx, idx); err != nil {
			// GIN indices may not be supported in all configurations (e.g., DoltGres)
			// Log warning but don't fail
			fmt.Printf("Warning: failed to create GIN index (keyword search may be slower): %v\n", err)
		}
	}

	// Create extracted_triples table
	var triplesTable string
	if p.usePgVector {
		triplesTable = fmt.Sprintf(`
			CREATE TABLE IF NOT EXISTS extracted_triples (
				id VARCHAR(255) PRIMARY KEY,
				source_id VARCHAR(255) REFERENCES sources(id),
				subject TEXT,
				predicate TEXT,
				object TEXT,
				condition TEXT,
				temporal TEXT,
				location TEXT,
				certainty TEXT,
				scope TEXT,
				source_attribution TEXT,
				confidence FLOAT,
				embedding vector(%d),
				chunk_index INT,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)`, p.embeddingDimensions)
	} else {
		triplesTable = `
			CREATE TABLE IF NOT EXISTS extracted_triples (
				id VARCHAR(255) PRIMARY KEY,
				source_id VARCHAR(255) REFERENCES sources(id),
				subject TEXT,
				predicate TEXT,
				object TEXT,
				condition TEXT,
				temporal TEXT,
				location TEXT,
				certainty TEXT,
				scope TEXT,
				source_attribution TEXT,
				confidence FLOAT,
				embedding JSONB,
				chunk_index INT,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)`
	}
	if _, err := p.db.ExecContext(ctx, triplesTable); err != nil {
		return fmt.Errorf("failed to create extracted_triples table: %w", err)
	}

	// Create extracted_rules table
	var rulesTable string
	if p.usePgVector {
		rulesTable = fmt.Sprintf(`
			CREATE TABLE IF NOT EXISTS extracted_rules (
				id VARCHAR(255) PRIMARY KEY,
				source_id VARCHAR(255) REFERENCES sources(id),
				antecedent TEXT,
				consequent TEXT,
				exception TEXT,
				rule_type TEXT,
				scope TEXT,
				source_attribution TEXT,
				confidence FLOAT,
				embedding vector(%d),
				chunk_index INT,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)`, p.embeddingDimensions)
	} else {
		rulesTable = `
			CREATE TABLE IF NOT EXISTS extracted_rules (
				id VARCHAR(255) PRIMARY KEY,
				source_id VARCHAR(255) REFERENCES sources(id),
				antecedent TEXT,
				consequent TEXT,
				exception TEXT,
				rule_type TEXT,
				scope TEXT,
				source_attribution TEXT,
				confidence FLOAT,
				embedding JSONB,
				chunk_index INT,
				created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
			)`
	}
	if _, err := p.db.ExecContext(ctx, rulesTable); err != nil {
		return fmt.Errorf("failed to create extracted_rules table: %w", err)
	}

	// --- Entity dedup migrations ---

	// Add normalized_name column to extracted_nodes
	if _, err := p.db.ExecContext(ctx, `ALTER TABLE extracted_nodes ADD COLUMN IF NOT EXISTS normalized_name TEXT`); err != nil {
		fmt.Printf("Warning: failed to add normalized_name column: %v\n", err)
	}

	// Backfill normalized_name for existing rows
	if _, err := p.db.ExecContext(ctx, `UPDATE extracted_nodes SET normalized_name = LOWER(TRIM(name)) WHERE normalized_name IS NULL`); err != nil {
		fmt.Printf("Warning: failed to backfill normalized_name: %v\n", err)
	}

	// Create node_sources junction table
	nodeSourcesTable := `
		CREATE TABLE IF NOT EXISTS node_sources (
			node_id VARCHAR(255) REFERENCES extracted_nodes(id),
			source_id VARCHAR(255) REFERENCES sources(id),
			chunk_index INT,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY (node_id, source_id, chunk_index)
		)`
	if _, err := p.db.ExecContext(ctx, nodeSourcesTable); err != nil {
		return fmt.Errorf("failed to create node_sources table: %w", err)
	}

	// Create dedup lookup index
	if _, err := p.db.ExecContext(ctx, `CREATE INDEX IF NOT EXISTS idx_nodes_dedup ON extracted_nodes(group_id, normalized_name, type)`); err != nil {
		fmt.Printf("Warning: failed to create dedup index: %v\n", err)
	}

	// Backfill node_sources from existing data
	if _, err := p.db.ExecContext(ctx, `
		INSERT INTO node_sources (node_id, source_id, chunk_index, created_at)
		SELECT id, source_id, chunk_index, created_at FROM extracted_nodes
		ON CONFLICT DO NOTHING`); err != nil {
		fmt.Printf("Warning: failed to backfill node_sources: %v\n", err)
	}

	return nil
}

// CreateVectorIndices creates IVFFlat indices for vector similarity search.
// This should be called after bulk data loading for optimal performance.
// lists parameter determines the number of clusters (recommended: sqrt(num_rows))
func (p *PostgresDB) CreateVectorIndices(ctx context.Context, lists int) error {
	if lists <= 0 {
		lists = 100 // Default
	}

	nodeIdx := fmt.Sprintf(`
		CREATE INDEX IF NOT EXISTS idx_nodes_embedding 
		ON extracted_nodes USING ivfflat (embedding vector_cosine_ops)
		WITH (lists = %d)`, lists)
	if _, err := p.db.ExecContext(ctx, nodeIdx); err != nil {
		return fmt.Errorf("failed to create node vector index: %w", err)
	}

	edgeIdx := fmt.Sprintf(`
		CREATE INDEX IF NOT EXISTS idx_edges_embedding 
		ON extracted_edges USING ivfflat (embedding vector_cosine_ops)
		WITH (lists = %d)`, lists)
	if _, err := p.db.ExecContext(ctx, edgeIdx); err != nil {
		return fmt.Errorf("failed to create edge vector index: %w", err)
	}

	return nil
}

func (p *PostgresDB) SaveSource(ctx context.Context, source *Source) error {
	metadataJSON, err := json.Marshal(source.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	query := `
		INSERT INTO sources (id, name, content, group_id, metadata, created_at) 
		VALUES ($1, $2, $3, $4, $5, $6)
		ON CONFLICT (id) DO UPDATE SET
			name = EXCLUDED.name,
			content = EXCLUDED.content,
			group_id = EXCLUDED.group_id,
			metadata = EXCLUDED.metadata`

	createdAt := source.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now()
	}

	_, err = p.db.ExecContext(ctx, query,
		source.ID, source.Name, source.Content, source.GroupID, metadataJSON, createdAt)
	if err != nil {
		return fmt.Errorf("failed to insert source: %w", err)
	}
	return nil
}

func (p *PostgresDB) SaveExtractedKnowledge(ctx context.Context, sourceID string, nodes []*ExtractedNode, edges []*ExtractedEdge) error {
	tx, err := p.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	// Get source's group_id for nodes/edges
	var groupID string
	err = tx.QueryRowContext(ctx, "SELECT group_id FROM sources WHERE id = $1", sourceID).Scan(&groupID)
	if err != nil && err != sql.ErrNoRows {
		return fmt.Errorf("failed to get source group_id: %w", err)
	}

	// Track original node ID -> canonical (deduplicated) node ID for edge consistency
	nodeIDMap := make(map[string]string)

	for _, node := range nodes {
		normalizedName := utils.NormalizeStringExact(node.Name)
		normalizedType := strings.ToLower(strings.TrimSpace(node.Type))
		isPerson := normalizedType == "person"

		nodeGroupID := node.GroupID
		if nodeGroupID == "" {
			nodeGroupID = groupID
		}

		// Look up existing entity
		var existingID string
		var existingDesc string
		if isPerson {
			// Person: dedup only within the same source
			err = tx.QueryRowContext(ctx, `
				SELECT en.id, en.description FROM extracted_nodes en
				JOIN node_sources ns ON ns.node_id = en.id
				WHERE ns.source_id = $1 AND en.normalized_name = $2 AND LOWER(en.type) = $3
				LIMIT 1`,
				sourceID, normalizedName, normalizedType).Scan(&existingID, &existingDesc)
		} else {
			// Non-person: dedup globally within group
			err = tx.QueryRowContext(ctx, `
				SELECT id, description FROM extracted_nodes
				WHERE group_id = $1 AND normalized_name = $2 AND LOWER(type) = $3
				LIMIT 1`,
				nodeGroupID, normalizedName, normalizedType).Scan(&existingID, &existingDesc)
		}

		var canonicalID string
		if err == nil {
			// Found existing entity
			canonicalID = existingID

			// Update description if new one is longer
			if len(node.Description) > len(existingDesc) {
				if _, err := tx.ExecContext(ctx,
					`UPDATE extracted_nodes SET description = $1 WHERE id = $2`,
					node.Description, existingID); err != nil {
					return fmt.Errorf("failed to update node description %s: %w", existingID, err)
				}
			}

			// Update embedding to latest
			if len(node.Embedding) > 0 {
				embeddingVal := p.embeddingToString(node.Embedding)
				if _, err := tx.ExecContext(ctx,
					`UPDATE extracted_nodes SET embedding = $1 WHERE id = $2`,
					embeddingVal, existingID); err != nil {
					return fmt.Errorf("failed to update node embedding %s: %w", existingID, err)
				}
			}
		} else if err == sql.ErrNoRows {
			// New entity — insert
			canonicalID = node.ID

			var embeddingVal interface{}
			if len(node.Embedding) > 0 {
				embeddingVal = p.embeddingToString(node.Embedding)
			}

			createdAt := node.CreatedAt
			if createdAt.IsZero() {
				createdAt = time.Now()
			}

			if _, err := tx.ExecContext(ctx, `
				INSERT INTO extracted_nodes (id, source_id, group_id, name, normalized_name, type, description, embedding, chunk_index, created_at)
				VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
				ON CONFLICT (id) DO UPDATE SET
					name = EXCLUDED.name,
					normalized_name = EXCLUDED.normalized_name,
					type = EXCLUDED.type,
					description = EXCLUDED.description,
					embedding = EXCLUDED.embedding,
					chunk_index = EXCLUDED.chunk_index`,
				canonicalID, sourceID, nodeGroupID, node.Name, normalizedName, node.Type, node.Description,
				embeddingVal, node.ChunkIndex, createdAt); err != nil {
				return fmt.Errorf("failed to insert node %s: %w", canonicalID, err)
			}
		} else {
			return fmt.Errorf("failed to look up existing node: %w", err)
		}

		// Record mapping
		nodeIDMap[node.ID] = canonicalID

		// Insert into node_sources junction table
		if _, err := tx.ExecContext(ctx, `
			INSERT INTO node_sources (node_id, source_id, chunk_index, created_at)
			VALUES ($1, $2, $3, CURRENT_TIMESTAMP)
			ON CONFLICT DO NOTHING`,
			canonicalID, sourceID, node.ChunkIndex); err != nil {
			return fmt.Errorf("failed to insert node_source for %s: %w", canonicalID, err)
		}
	}

	// Insert edges
	edgeStmt, err := tx.PrepareContext(ctx, `
		INSERT INTO extracted_edges (id, source_id, group_id, source_node_name, source_node_type, target_node_name, target_node_type, relation, description, embedding, weight, chunk_index, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
		ON CONFLICT (id) DO UPDATE SET
			source_node_name = EXCLUDED.source_node_name,
			source_node_type = EXCLUDED.source_node_type,
			target_node_name = EXCLUDED.target_node_name,
			target_node_type = EXCLUDED.target_node_type,
			relation = EXCLUDED.relation,
			description = EXCLUDED.description,
			embedding = EXCLUDED.embedding,
			weight = EXCLUDED.weight,
			chunk_index = EXCLUDED.chunk_index`)
	if err != nil {
		return fmt.Errorf("failed to prepare edge statement: %w", err)
	}
	defer edgeStmt.Close()

	for _, edge := range edges {
		var embeddingVal interface{}
		if len(edge.Embedding) > 0 {
			embeddingVal = p.embeddingToString(edge.Embedding)
		}
		edgeGroupID := edge.GroupID
		if edgeGroupID == "" {
			edgeGroupID = groupID
		}
		createdAt := edge.CreatedAt
		if createdAt.IsZero() {
			createdAt = time.Now()
		}

		if _, err := edgeStmt.ExecContext(ctx,
			edge.ID, sourceID, edgeGroupID, edge.SourceNodeName, edge.SourceNodeType, edge.TargetNodeName, edge.TargetNodeType,
			edge.Relation, edge.Description, embeddingVal, edge.Weight, edge.ChunkIndex, createdAt); err != nil {
			return fmt.Errorf("failed to insert edge %s: %w", edge.ID, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

func (p *PostgresDB) GetSource(ctx context.Context, sourceID string) (*Source, error) {
	row := p.db.QueryRowContext(ctx,
		"SELECT id, name, content, group_id, metadata, created_at FROM sources WHERE id = $1", sourceID)

	var s Source
	var metadataBytes []byte

	if err := row.Scan(&s.ID, &s.Name, &s.Content, &s.GroupID, &metadataBytes, &s.CreatedAt); err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("source not found: %s", sourceID)
		}
		return nil, fmt.Errorf("failed to scan source: %w", err)
	}

	if len(metadataBytes) > 0 {
		if err := json.Unmarshal(metadataBytes, &s.Metadata); err != nil {
			return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
		}
	}

	return &s, nil
}

func (p *PostgresDB) GetExtractedNodes(ctx context.Context, sourceID string) ([]*ExtractedNode, error) {
	rows, err := p.db.QueryContext(ctx, `
		SELECT en.id, en.source_id, en.group_id, en.name, en.normalized_name, en.type,
		       en.description, en.embedding, en.chunk_index, en.created_at
		FROM extracted_nodes en
		JOIN node_sources ns ON ns.node_id = en.id
		WHERE ns.source_id = $1`,
		sourceID)
	if err != nil {
		return nil, fmt.Errorf("failed to query extracted nodes: %w", err)
	}
	defer rows.Close()

	return p.scanNodes(rows)
}

func (p *PostgresDB) GetExtractedEdges(ctx context.Context, sourceID string) ([]*ExtractedEdge, error) {
	rows, err := p.db.QueryContext(ctx,
		"SELECT id, source_id, group_id, source_node_name, source_node_type, target_node_name, target_node_type, relation, description, embedding, weight, chunk_index, created_at FROM extracted_edges WHERE source_id = $1",
		sourceID)
	if err != nil {
		return nil, fmt.Errorf("failed to query extracted edges: %w", err)
	}
	defer rows.Close()

	return p.scanEdges(rows)
}

func (p *PostgresDB) GetAllSources(ctx context.Context, limit int) ([]*Source, error) {
	query := "SELECT id, name, content, group_id, metadata, created_at FROM sources ORDER BY created_at DESC"
	if limit > 0 {
		query += fmt.Sprintf(" LIMIT %d", limit)
	}

	rows, err := p.db.QueryContext(ctx, query)
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

func (p *PostgresDB) GetAllNodes(ctx context.Context, limit int) ([]*ExtractedNode, error) {
	query := "SELECT id, source_id, group_id, name, normalized_name, type, description, embedding, chunk_index, created_at FROM extracted_nodes"
	if limit > 0 {
		query += fmt.Sprintf(" LIMIT %d", limit)
	}

	rows, err := p.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to query nodes: %w", err)
	}
	defer rows.Close()

	return p.scanNodes(rows)
}

func (p *PostgresDB) GetAllEdges(ctx context.Context, limit int) ([]*ExtractedEdge, error) {
	query := "SELECT id, source_id, group_id, source_node_name, source_node_type, target_node_name, target_node_type, relation, description, embedding, weight, chunk_index, created_at FROM extracted_edges"
	if limit > 0 {
		query += fmt.Sprintf(" LIMIT %d", limit)
	}

	rows, err := p.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to query edges: %w", err)
	}
	defer rows.Close()

	return p.scanEdges(rows)
}

func (p *PostgresDB) GetStats(ctx context.Context) (*Stats, error) {
	stats := &Stats{}

	if err := p.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM sources").Scan(&stats.SourceCount); err != nil {
		return nil, fmt.Errorf("failed to count sources: %w", err)
	}
	if err := p.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM extracted_nodes").Scan(&stats.NodeCount); err != nil {
		return nil, fmt.Errorf("failed to count nodes: %w", err)
	}
	if err := p.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM extracted_edges").Scan(&stats.EdgeCount); err != nil {
		return nil, fmt.Errorf("failed to count edges: %w", err)
	}
	// Triple and rule counts are optional (tables may not exist on older schemas)
	_ = p.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM extracted_triples").Scan(&stats.TripleCount)
	_ = p.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM extracted_rules").Scan(&stats.RuleCount)

	return stats, nil
}

func (p *PostgresDB) Close() error {
	return p.db.Close()
}

// --- Extended Extraction Methods ---

func (p *PostgresDB) SaveExtractedTriples(ctx context.Context, sourceID string, triples []*ExtractedTriple) error {
	if len(triples) == 0 {
		return nil
	}

	tx, err := p.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	stmt, err := tx.PrepareContext(ctx, `
		INSERT INTO extracted_triples (id, source_id, subject, predicate, object, condition, temporal, location, certainty, scope, source_attribution, confidence, embedding, chunk_index, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
		ON CONFLICT (id) DO UPDATE SET
			subject = EXCLUDED.subject,
			predicate = EXCLUDED.predicate,
			object = EXCLUDED.object,
			condition = EXCLUDED.condition,
			temporal = EXCLUDED.temporal,
			location = EXCLUDED.location,
			certainty = EXCLUDED.certainty,
			scope = EXCLUDED.scope,
			source_attribution = EXCLUDED.source_attribution,
			confidence = EXCLUDED.confidence,
			embedding = EXCLUDED.embedding,
			chunk_index = EXCLUDED.chunk_index`)
	if err != nil {
		return fmt.Errorf("failed to prepare triple statement: %w", err)
	}
	defer stmt.Close()

	for _, t := range triples {
		var embeddingVal interface{}
		if len(t.Embedding) > 0 {
			embeddingVal = p.embeddingToString(t.Embedding)
		}
		createdAt := t.CreatedAt
		if createdAt.IsZero() {
			createdAt = time.Now()
		}

		if _, err := stmt.ExecContext(ctx,
			t.ID, sourceID, t.Subject, t.Predicate, t.Object,
			t.Condition, t.Temporal, t.Location, t.Certainty, t.Scope, t.SourceAttribution,
			t.Confidence, embeddingVal, t.ChunkIndex, createdAt); err != nil {
			return fmt.Errorf("failed to insert triple %s: %w", t.ID, err)
		}
	}

	return tx.Commit()
}

func (p *PostgresDB) SaveExtractedRules(ctx context.Context, sourceID string, rules []*ExtractedRule) error {
	if len(rules) == 0 {
		return nil
	}

	tx, err := p.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	stmt, err := tx.PrepareContext(ctx, `
		INSERT INTO extracted_rules (id, source_id, antecedent, consequent, exception, rule_type, scope, source_attribution, confidence, embedding, chunk_index, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
		ON CONFLICT (id) DO UPDATE SET
			antecedent = EXCLUDED.antecedent,
			consequent = EXCLUDED.consequent,
			exception = EXCLUDED.exception,
			rule_type = EXCLUDED.rule_type,
			scope = EXCLUDED.scope,
			source_attribution = EXCLUDED.source_attribution,
			confidence = EXCLUDED.confidence,
			embedding = EXCLUDED.embedding,
			chunk_index = EXCLUDED.chunk_index`)
	if err != nil {
		return fmt.Errorf("failed to prepare rule statement: %w", err)
	}
	defer stmt.Close()

	for _, r := range rules {
		var embeddingVal interface{}
		if len(r.Embedding) > 0 {
			embeddingVal = p.embeddingToString(r.Embedding)
		}
		createdAt := r.CreatedAt
		if createdAt.IsZero() {
			createdAt = time.Now()
		}

		if _, err := stmt.ExecContext(ctx,
			r.ID, sourceID, r.Antecedent, r.Consequent, r.Exception,
			r.RuleType, r.Scope, r.SourceAttribution,
			r.Confidence, embeddingVal, r.ChunkIndex, createdAt); err != nil {
			return fmt.Errorf("failed to insert rule %s: %w", r.ID, err)
		}
	}

	return tx.Commit()
}

func (p *PostgresDB) GetExtractedTriples(ctx context.Context, sourceID string) ([]*ExtractedTriple, error) {
	rows, err := p.db.QueryContext(ctx, `
		SELECT id, source_id, subject, predicate, object, condition, temporal, location,
		       certainty, scope, source_attribution, confidence, embedding, chunk_index, created_at
		FROM extracted_triples WHERE source_id = $1`, sourceID)
	if err != nil {
		return nil, fmt.Errorf("failed to query extracted triples: %w", err)
	}
	defer rows.Close()

	var triples []*ExtractedTriple
	for rows.Next() {
		var t ExtractedTriple
		var embeddingStr sql.NullString
		var condition, temporal, location, certainty, scope, sourceAttr sql.NullString
		var confidence sql.NullFloat64

		if err := rows.Scan(&t.ID, &t.SourceID, &t.Subject, &t.Predicate, &t.Object,
			&condition, &temporal, &location, &certainty, &scope, &sourceAttr,
			&confidence, &embeddingStr, &t.ChunkIndex, &t.CreatedAt); err != nil {
			return nil, err
		}

		if condition.Valid {
			t.Condition = condition.String
		}
		if temporal.Valid {
			t.Temporal = temporal.String
		}
		if location.Valid {
			t.Location = location.String
		}
		if certainty.Valid {
			t.Certainty = certainty.String
		}
		if scope.Valid {
			t.Scope = scope.String
		}
		if sourceAttr.Valid {
			t.SourceAttribution = sourceAttr.String
		}
		if confidence.Valid {
			t.Confidence = confidence.Float64
		}
		if embeddingStr.Valid {
			t.Embedding = p.parseEmbedding(embeddingStr.String)
		}

		triples = append(triples, &t)
	}
	return triples, nil
}

func (p *PostgresDB) GetExtractedRules(ctx context.Context, sourceID string) ([]*ExtractedRule, error) {
	rows, err := p.db.QueryContext(ctx, `
		SELECT id, source_id, antecedent, consequent, exception, rule_type, scope,
		       source_attribution, confidence, embedding, chunk_index, created_at
		FROM extracted_rules WHERE source_id = $1`, sourceID)
	if err != nil {
		return nil, fmt.Errorf("failed to query extracted rules: %w", err)
	}
	defer rows.Close()

	var rules []*ExtractedRule
	for rows.Next() {
		var r ExtractedRule
		var embeddingStr sql.NullString
		var exception, ruleType, scope, sourceAttr sql.NullString
		var confidence sql.NullFloat64

		if err := rows.Scan(&r.ID, &r.SourceID, &r.Antecedent, &r.Consequent,
			&exception, &ruleType, &scope, &sourceAttr,
			&confidence, &embeddingStr, &r.ChunkIndex, &r.CreatedAt); err != nil {
			return nil, err
		}

		if exception.Valid {
			r.Exception = exception.String
		}
		if ruleType.Valid {
			r.RuleType = ruleType.String
		}
		if scope.Valid {
			r.Scope = scope.String
		}
		if sourceAttr.Valid {
			r.SourceAttribution = sourceAttr.String
		}
		if confidence.Valid {
			r.Confidence = confidence.Float64
		}
		if embeddingStr.Valid {
			r.Embedding = p.parseEmbedding(embeddingStr.String)
		}

		rules = append(rules, &r)
	}
	return rules, nil
}

// --- Search Methods ---

func (p *PostgresDB) SearchNodes(ctx context.Context, query string, embedding []float32, config *FactSearchConfig) ([]*ExtractedNode, []float64, error) {
	if config == nil {
		config = &FactSearchConfig{Limit: 10}
	}
	if config.Limit <= 0 {
		config.Limit = 10
	}

	// Determine search methods
	useVector := len(embedding) > 0
	useKeyword := query != ""

	if len(config.SearchMethods) > 0 {
		useVector = false
		useKeyword = false
		for _, m := range config.SearchMethods {
			if m == VectorSearch {
				useVector = len(embedding) > 0
			}
			if m == KeywordSearch {
				useKeyword = query != ""
			}
		}
	}

	if !useVector && !useKeyword {
		return []*ExtractedNode{}, []float64{}, nil
	}

	// If both methods, use hybrid search internally
	if useVector && useKeyword {
		vectorNodes, vectorScores, err := p.vectorSearchNodes(ctx, embedding, config)
		if err != nil {
			return nil, nil, err
		}
		keywordNodes, keywordScores, err := p.keywordSearchNodes(ctx, query, config)
		if err != nil {
			return nil, nil, err
		}
		return p.rrfMergeNodes(vectorNodes, vectorScores, keywordNodes, keywordScores, config.Limit, config.MinScore)
	}

	if useVector {
		return p.vectorSearchNodes(ctx, embedding, config)
	}

	return p.keywordSearchNodes(ctx, query, config)
}

func (p *PostgresDB) SearchEdges(ctx context.Context, query string, embedding []float32, config *FactSearchConfig) ([]*ExtractedEdge, []float64, error) {
	if config == nil {
		config = &FactSearchConfig{Limit: 10}
	}
	if config.Limit <= 0 {
		config.Limit = 10
	}

	useVector := len(embedding) > 0
	useKeyword := query != ""

	if len(config.SearchMethods) > 0 {
		useVector = false
		useKeyword = false
		for _, m := range config.SearchMethods {
			if m == VectorSearch {
				useVector = len(embedding) > 0
			}
			if m == KeywordSearch {
				useKeyword = query != ""
			}
		}
	}

	if !useVector && !useKeyword {
		return []*ExtractedEdge{}, []float64{}, nil
	}

	if useVector && useKeyword {
		vectorEdges, vectorScores, err := p.vectorSearchEdges(ctx, embedding, config)
		if err != nil {
			return nil, nil, err
		}
		keywordEdges, keywordScores, err := p.keywordSearchEdges(ctx, query, config)
		if err != nil {
			return nil, nil, err
		}
		return p.rrfMergeEdges(vectorEdges, vectorScores, keywordEdges, keywordScores, config.Limit, config.MinScore)
	}

	if useVector {
		return p.vectorSearchEdges(ctx, embedding, config)
	}

	return p.keywordSearchEdges(ctx, query, config)
}

func (p *PostgresDB) SearchSources(ctx context.Context, query string, config *FactSearchConfig) ([]*Source, []float64, error) {
	if config == nil {
		config = &FactSearchConfig{Limit: 10}
	}
	if config.Limit <= 0 {
		config.Limit = 10
	}

	if query == "" {
		return []*Source{}, []float64{}, nil
	}

	// Build query with filters
	sqlQuery := `
		SELECT id, name, content, group_id, metadata, created_at,
			   ts_rank(to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(content, '')), 
			          plainto_tsquery('english', $1)) AS score
		FROM sources
		WHERE to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(content, '')) 
			  @@ plainto_tsquery('english', $1)`

	args := []interface{}{query}
	argIdx := 2

	if config.GroupID != "" {
		sqlQuery += fmt.Sprintf(" AND group_id = $%d", argIdx)
		args = append(args, config.GroupID)
		argIdx++
	}

	if config.TimeRange != nil {
		if !config.TimeRange.Start.IsZero() {
			sqlQuery += fmt.Sprintf(" AND created_at >= $%d", argIdx)
			args = append(args, config.TimeRange.Start)
			argIdx++
		}
		if !config.TimeRange.End.IsZero() {
			sqlQuery += fmt.Sprintf(" AND created_at <= $%d", argIdx)
			args = append(args, config.TimeRange.End)
		}
	}

	// Add limit to prevent loading too many rows into memory
	sqlQuery += fmt.Sprintf(" LIMIT %d", MaxInMemorySearchResults)

	rows, err := p.db.QueryContext(ctx, sqlQuery, args...)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to search sources: %w", err)
	}
	defer rows.Close()

	var sources []*Source
	var scores []float64

	for rows.Next() {
		var s Source
		var metadataBytes []byte
		var score float64

		if err := rows.Scan(&s.ID, &s.Name, &s.Content, &s.GroupID, &metadataBytes, &s.CreatedAt, &score); err != nil {
			return nil, nil, err
		}

		if score < config.MinScore {
			continue
		}

		if len(metadataBytes) > 0 {
			if err := json.Unmarshal(metadataBytes, &s.Metadata); err != nil {
				return nil, nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
			}
		}

		sources = append(sources, &s)
		scores = append(scores, score)
	}

	return sources, scores, nil
}

func (p *PostgresDB) HybridSearch(ctx context.Context, query string, embedding []float32, config *FactSearchConfig) (*FactSearchResults, error) {
	if config == nil {
		config = &FactSearchConfig{Limit: 10}
	}

	// Search nodes
	nodes, nodeScores, err := p.SearchNodes(ctx, query, embedding, config)
	if err != nil {
		return nil, fmt.Errorf("node search failed: %w", err)
	}

	// Search edges
	edges, edgeScores, err := p.SearchEdges(ctx, query, embedding, config)
	if err != nil {
		return nil, fmt.Errorf("edge search failed: %w", err)
	}

	return &FactSearchResults{
		Nodes:      nodes,
		Edges:      edges,
		NodeScores: nodeScores,
		EdgeScores: edgeScores,
		Query:      query,
		Total:      len(nodes) + len(edges),
	}, nil
}

// --- Internal search methods ---

func (p *PostgresDB) vectorSearchNodes(ctx context.Context, embedding []float32, config *FactSearchConfig) ([]*ExtractedNode, []float64, error) {
	// For DoltGres (no VectorChord), use in-memory cosine similarity
	if !p.usePgVector {
		return p.inMemoryVectorSearchNodes(ctx, embedding, config)
	}

	embeddingStr := p.embeddingToString(embedding)

	// Build query with filters (VectorChord mode)
	sqlQuery := `
		SELECT id, source_id, group_id, name, normalized_name, type, description, embedding, chunk_index, created_at,
			   1 - (embedding <=> $1::vector) AS score
		FROM extracted_nodes
		WHERE embedding IS NOT NULL`

	args := []interface{}{embeddingStr}
	argIdx := 2

	if config.GroupID != "" {
		sqlQuery += fmt.Sprintf(" AND group_id = $%d", argIdx)
		args = append(args, config.GroupID)
		argIdx++
	}

	if len(config.NodeTypes) > 0 {
		placeholders := make([]string, len(config.NodeTypes))
		for i, t := range config.NodeTypes {
			placeholders[i] = fmt.Sprintf("$%d", argIdx)
			args = append(args, t)
			argIdx++
		}
		sqlQuery += fmt.Sprintf(" AND type IN (%s)", strings.Join(placeholders, ", "))
	}

	if config.TimeRange != nil {
		if !config.TimeRange.Start.IsZero() {
			sqlQuery += fmt.Sprintf(" AND created_at >= $%d", argIdx)
			args = append(args, config.TimeRange.Start)
			argIdx++
		}
		if !config.TimeRange.End.IsZero() {
			sqlQuery += fmt.Sprintf(" AND created_at <= $%d", argIdx)
			args = append(args, config.TimeRange.End)
			argIdx++
		}
	}

	sqlQuery += " ORDER BY embedding <=> $1::vector"
	sqlQuery += fmt.Sprintf(" LIMIT $%d", argIdx)
	args = append(args, config.Limit*2) // Fetch more for filtering

	rows, err := p.db.QueryContext(ctx, sqlQuery, args...)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to execute vector search: %w", err)
	}
	defer rows.Close()

	var nodes []*ExtractedNode
	var scores []float64

	for rows.Next() {
		var n ExtractedNode
		var embeddingStr sql.NullString
		var normalizedName sql.NullString
		var score float64

		if err := rows.Scan(&n.ID, &n.SourceID, &n.GroupID, &n.Name, &normalizedName, &n.Type, &n.Description,
			&embeddingStr, &n.ChunkIndex, &n.CreatedAt, &score); err != nil {
			return nil, nil, err
		}

		if normalizedName.Valid {
			n.NormalizedName = normalizedName.String
		}

		if score < config.MinScore {
			continue
		}

		if embeddingStr.Valid {
			n.Embedding = p.parseEmbedding(embeddingStr.String)
		}

		nodes = append(nodes, &n)
		scores = append(scores, score)

		if len(nodes) >= config.Limit {
			break
		}
	}

	return nodes, scores, nil
}

// MaxInMemorySearchResults is the maximum number of results to process in-memory
// to prevent excessive memory usage on large datasets.
const MaxInMemorySearchResults = 10000

// inMemoryVectorSearchNodes performs vector search by loading embeddings and computing
// cosine similarity in Go. Used for DoltGres which doesn't support VectorChord.
func (p *PostgresDB) inMemoryVectorSearchNodes(ctx context.Context, embedding []float32, config *FactSearchConfig) ([]*ExtractedNode, []float64, error) {
	// Build query to fetch all nodes with embeddings
	// Limit to MaxInMemorySearchResults to prevent excessive memory usage
	sqlQuery := `
		SELECT id, source_id, group_id, name, normalized_name, type, description, embedding, chunk_index, created_at
		FROM extracted_nodes
		WHERE embedding IS NOT NULL`

	args := []interface{}{}
	argIdx := 1

	if config.GroupID != "" {
		sqlQuery += fmt.Sprintf(" AND group_id = $%d", argIdx)
		args = append(args, config.GroupID)
		argIdx++
	}

	if len(config.NodeTypes) > 0 {
		placeholders := make([]string, len(config.NodeTypes))
		for i, t := range config.NodeTypes {
			placeholders[i] = fmt.Sprintf("$%d", argIdx)
			args = append(args, t)
			argIdx++
		}
		sqlQuery += fmt.Sprintf(" AND type IN (%s)", strings.Join(placeholders, ", "))
	}

	if config.TimeRange != nil {
		if !config.TimeRange.Start.IsZero() {
			sqlQuery += fmt.Sprintf(" AND created_at >= $%d", argIdx)
			args = append(args, config.TimeRange.Start)
			argIdx++
		}
		if !config.TimeRange.End.IsZero() {
			sqlQuery += fmt.Sprintf(" AND created_at <= $%d", argIdx)
			args = append(args, config.TimeRange.End)
		}
	}

	rows, err := p.db.QueryContext(ctx, sqlQuery, args...)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to query nodes: %w", err)
	}
	defer rows.Close()

	// Collect all nodes with their embeddings
	type nodeWithScore struct {
		node  *ExtractedNode
		score float64
	}
	var candidates []nodeWithScore

	for rows.Next() {
		var n ExtractedNode
		var embeddingJSON sql.NullString
		var normalizedName sql.NullString

		if err := rows.Scan(&n.ID, &n.SourceID, &n.GroupID, &n.Name, &normalizedName, &n.Type, &n.Description,
			&embeddingJSON, &n.ChunkIndex, &n.CreatedAt); err != nil {
			return nil, nil, err
		}

		if normalizedName.Valid {
			n.NormalizedName = normalizedName.String
		}

		if !embeddingJSON.Valid || embeddingJSON.String == "" {
			continue
		}

		// Parse embedding from JSONB (stored as JSON array for DoltGres)
		nodeEmbedding := p.parseEmbeddingJSON(embeddingJSON.String)
		if len(nodeEmbedding) == 0 {
			continue
		}

		// Compute cosine similarity
		score := cosineSimilarity(embedding, nodeEmbedding)
		if score >= config.MinScore {
			n.Embedding = nodeEmbedding
			candidates = append(candidates, nodeWithScore{node: &n, score: score})
		}
	}

	// Log warning if we hit the limit
	if len(candidates) >= MaxInMemorySearchResults {
		log.Printf("WARNING: In-memory vector search hit limit of %d results. Consider using VectorChord for better performance.", MaxInMemorySearchResults)
	}

	// Sort by score descending using O(n log n) sort
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	// Extract top results
	var nodes []*ExtractedNode
	var scores []float64
	for i := 0; i < len(candidates) && i < config.Limit; i++ {
		nodes = append(nodes, candidates[i].node)
		scores = append(scores, candidates[i].score)
	}

	return nodes, scores, nil
}

func (p *PostgresDB) keywordSearchNodes(ctx context.Context, query string, config *FactSearchConfig) ([]*ExtractedNode, []float64, error) {
	sqlQuery := `
		SELECT id, source_id, group_id, name, normalized_name, type, description, embedding, chunk_index, created_at,
			   ts_rank(to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(description, '')),
			          plainto_tsquery('english', $1)) AS score
		FROM extracted_nodes
		WHERE to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(description, ''))
			  @@ plainto_tsquery('english', $1)`

	args := []interface{}{query}
	argIdx := 2

	if config.GroupID != "" {
		sqlQuery += fmt.Sprintf(" AND group_id = $%d", argIdx)
		args = append(args, config.GroupID)
		argIdx++
	}

	if len(config.NodeTypes) > 0 {
		placeholders := make([]string, len(config.NodeTypes))
		for i, t := range config.NodeTypes {
			placeholders[i] = fmt.Sprintf("$%d", argIdx)
			args = append(args, t)
			argIdx++
		}
		sqlQuery += fmt.Sprintf(" AND type IN (%s)", strings.Join(placeholders, ", "))
	}

	if config.TimeRange != nil {
		if !config.TimeRange.Start.IsZero() {
			sqlQuery += fmt.Sprintf(" AND created_at >= $%d", argIdx)
			args = append(args, config.TimeRange.Start)
			argIdx++
		}
		if !config.TimeRange.End.IsZero() {
			sqlQuery += fmt.Sprintf(" AND created_at <= $%d", argIdx)
			args = append(args, config.TimeRange.End)
			argIdx++
		}
	}

	sqlQuery += " ORDER BY score DESC"
	sqlQuery += fmt.Sprintf(" LIMIT $%d", argIdx)
	args = append(args, config.Limit*2)

	rows, err := p.db.QueryContext(ctx, sqlQuery, args...)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to execute keyword search: %w", err)
	}
	defer rows.Close()

	var nodes []*ExtractedNode
	var scores []float64

	for rows.Next() {
		var n ExtractedNode
		var embeddingStr sql.NullString
		var normalizedName sql.NullString
		var score float64

		if err := rows.Scan(&n.ID, &n.SourceID, &n.GroupID, &n.Name, &normalizedName, &n.Type, &n.Description,
			&embeddingStr, &n.ChunkIndex, &n.CreatedAt, &score); err != nil {
			return nil, nil, err
		}

		if normalizedName.Valid {
			n.NormalizedName = normalizedName.String
		}

		if score < config.MinScore {
			continue
		}

		if embeddingStr.Valid {
			n.Embedding = p.parseEmbedding(embeddingStr.String)
		}

		nodes = append(nodes, &n)
		scores = append(scores, score)

		if len(nodes) >= config.Limit {
			break
		}
	}

	return nodes, scores, nil
}

func (p *PostgresDB) vectorSearchEdges(ctx context.Context, embedding []float32, config *FactSearchConfig) ([]*ExtractedEdge, []float64, error) {
	// For DoltGres (no VectorChord), use in-memory cosine similarity
	if !p.usePgVector {
		return p.inMemoryVectorSearchEdges(ctx, embedding, config)
	}

	embeddingStr := p.embeddingToString(embedding)

	sqlQuery := `
		SELECT id, source_id, group_id, source_node_name, source_node_type, target_node_name, target_node_type, relation, description, embedding, weight, chunk_index, created_at,
			   1 - (embedding <=> $1::vector) AS score
		FROM extracted_edges
		WHERE embedding IS NOT NULL`

	args := []interface{}{embeddingStr}
	argIdx := 2

	if config.GroupID != "" {
		sqlQuery += fmt.Sprintf(" AND group_id = $%d", argIdx)
		args = append(args, config.GroupID)
		argIdx++
	}

	if config.TimeRange != nil {
		if !config.TimeRange.Start.IsZero() {
			sqlQuery += fmt.Sprintf(" AND created_at >= $%d", argIdx)
			args = append(args, config.TimeRange.Start)
			argIdx++
		}
		if !config.TimeRange.End.IsZero() {
			sqlQuery += fmt.Sprintf(" AND created_at <= $%d", argIdx)
			args = append(args, config.TimeRange.End)
			argIdx++
		}
	}

	sqlQuery += " ORDER BY embedding <=> $1::vector"
	sqlQuery += fmt.Sprintf(" LIMIT $%d", argIdx)
	args = append(args, config.Limit*2)

	rows, err := p.db.QueryContext(ctx, sqlQuery, args...)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to execute edge vector search: %w", err)
	}
	defer rows.Close()

	var edges []*ExtractedEdge
	var scores []float64

	for rows.Next() {
		var e ExtractedEdge
		var embeddingStr sql.NullString
		var score float64

		if err := rows.Scan(&e.ID, &e.SourceID, &e.GroupID, &e.SourceNodeName, &e.SourceNodeType, &e.TargetNodeName, &e.TargetNodeType,
			&e.Relation, &e.Description, &embeddingStr, &e.Weight, &e.ChunkIndex, &e.CreatedAt, &score); err != nil {
			return nil, nil, err
		}

		if score < config.MinScore {
			continue
		}

		if embeddingStr.Valid {
			e.Embedding = p.parseEmbedding(embeddingStr.String)
		}

		edges = append(edges, &e)
		scores = append(scores, score)

		if len(edges) >= config.Limit {
			break
		}
	}

	return edges, scores, nil
}

// inMemoryVectorSearchEdges performs vector search on edges by loading embeddings
// and computing cosine similarity in Go. Used for DoltGres.
func (p *PostgresDB) inMemoryVectorSearchEdges(ctx context.Context, embedding []float32, config *FactSearchConfig) ([]*ExtractedEdge, []float64, error) {
	sqlQuery := `
		SELECT id, source_id, group_id, source_node_name, source_node_type, target_node_name, target_node_type, relation, description, embedding, weight, chunk_index, created_at
		FROM extracted_edges
		WHERE embedding IS NOT NULL`

	args := []interface{}{}
	argIdx := 1

	if config.GroupID != "" {
		sqlQuery += fmt.Sprintf(" AND group_id = $%d", argIdx)
		args = append(args, config.GroupID)
		argIdx++
	}

	if config.TimeRange != nil {
		if !config.TimeRange.Start.IsZero() {
			sqlQuery += fmt.Sprintf(" AND created_at >= $%d", argIdx)
			args = append(args, config.TimeRange.Start)
			argIdx++
		}
		if !config.TimeRange.End.IsZero() {
			sqlQuery += fmt.Sprintf(" AND created_at <= $%d", argIdx)
			args = append(args, config.TimeRange.End)
		}
	}

	rows, err := p.db.QueryContext(ctx, sqlQuery, args...)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to query edges: %w", err)
	}
	defer rows.Close()

	type edgeWithScore struct {
		edge  *ExtractedEdge
		score float64
	}
	var candidates []edgeWithScore

	for rows.Next() {
		var e ExtractedEdge
		var embeddingJSON sql.NullString

		if err := rows.Scan(&e.ID, &e.SourceID, &e.GroupID, &e.SourceNodeName, &e.SourceNodeType, &e.TargetNodeName, &e.TargetNodeType,
			&e.Relation, &e.Description, &embeddingJSON, &e.Weight, &e.ChunkIndex, &e.CreatedAt); err != nil {
			return nil, nil, err
		}

		if !embeddingJSON.Valid || embeddingJSON.String == "" {
			continue
		}

		edgeEmbedding := p.parseEmbeddingJSON(embeddingJSON.String)
		if len(edgeEmbedding) == 0 {
			continue
		}

		score := cosineSimilarity(embedding, edgeEmbedding)
		if score >= config.MinScore {
			e.Embedding = edgeEmbedding
			candidates = append(candidates, edgeWithScore{edge: &e, score: score})
		}
	}

	// Sort by score descending
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	var edges []*ExtractedEdge
	var scores []float64
	for i := 0; i < len(candidates) && i < config.Limit; i++ {
		edges = append(edges, candidates[i].edge)
		scores = append(scores, candidates[i].score)
	}

	return edges, scores, nil
}

func (p *PostgresDB) keywordSearchEdges(ctx context.Context, query string, config *FactSearchConfig) ([]*ExtractedEdge, []float64, error) {
	sqlQuery := `
		SELECT id, source_id, group_id, source_node_name, source_node_type, target_node_name, target_node_type, relation, description, embedding, weight, chunk_index, created_at,
			   ts_rank(to_tsvector('english', COALESCE(relation, '') || ' ' || COALESCE(description, '')), 
			          plainto_tsquery('english', $1)) AS score
		FROM extracted_edges
		WHERE to_tsvector('english', COALESCE(relation, '') || ' ' || COALESCE(description, '')) 
			  @@ plainto_tsquery('english', $1)`

	args := []interface{}{query}
	argIdx := 2

	if config.GroupID != "" {
		sqlQuery += fmt.Sprintf(" AND group_id = $%d", argIdx)
		args = append(args, config.GroupID)
		argIdx++
	}

	if config.TimeRange != nil {
		if !config.TimeRange.Start.IsZero() {
			sqlQuery += fmt.Sprintf(" AND created_at >= $%d", argIdx)
			args = append(args, config.TimeRange.Start)
			argIdx++
		}
		if !config.TimeRange.End.IsZero() {
			sqlQuery += fmt.Sprintf(" AND created_at <= $%d", argIdx)
			args = append(args, config.TimeRange.End)
			argIdx++
		}
	}

	sqlQuery += " ORDER BY score DESC"
	sqlQuery += fmt.Sprintf(" LIMIT $%d", argIdx)
	args = append(args, config.Limit*2)

	rows, err := p.db.QueryContext(ctx, sqlQuery, args...)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to execute edge keyword search: %w", err)
	}
	defer rows.Close()

	var edges []*ExtractedEdge
	var scores []float64

	for rows.Next() {
		var e ExtractedEdge
		var embeddingStr sql.NullString
		var score float64

		if err := rows.Scan(&e.ID, &e.SourceID, &e.GroupID, &e.SourceNodeName, &e.SourceNodeType, &e.TargetNodeName, &e.TargetNodeType,
			&e.Relation, &e.Description, &embeddingStr, &e.Weight, &e.ChunkIndex, &e.CreatedAt, &score); err != nil {
			return nil, nil, err
		}

		if score < config.MinScore {
			continue
		}

		if embeddingStr.Valid {
			e.Embedding = p.parseEmbedding(embeddingStr.String)
		}

		edges = append(edges, &e)
		scores = append(scores, score)

		if len(edges) >= config.Limit {
			break
		}
	}

	return edges, scores, nil
}

// --- RRF Merge ---

func (p *PostgresDB) rrfMergeNodes(vectorNodes []*ExtractedNode, vectorScores []float64,
	keywordNodes []*ExtractedNode, keywordScores []float64,
	limit int, minScore float64) ([]*ExtractedNode, []float64, error) {

	const k = 60 // Standard RRF parameter

	// Build RRF score map
	rrfScores := make(map[string]float64)
	nodeMap := make(map[string]*ExtractedNode)

	// Add vector results
	for i, node := range vectorNodes {
		rrfScores[node.ID] += 1.0 / float64(k+i+1)
		nodeMap[node.ID] = node
	}

	// Add keyword results
	for i, node := range keywordNodes {
		rrfScores[node.ID] += 1.0 / float64(k+i+1)
		nodeMap[node.ID] = node
	}

	// Sort by RRF score
	type scoredNode struct {
		node  *ExtractedNode
		score float64
	}

	var scored []scoredNode
	for id, score := range rrfScores {
		if score >= minScore {
			scored = append(scored, scoredNode{node: nodeMap[id], score: score})
		}
	}

	// Sort descending
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Extract results
	var nodes []*ExtractedNode
	var scores []float64

	for i := 0; i < len(scored) && i < limit; i++ {
		nodes = append(nodes, scored[i].node)
		scores = append(scores, scored[i].score)
	}

	return nodes, scores, nil
}

func (p *PostgresDB) rrfMergeEdges(vectorEdges []*ExtractedEdge, vectorScores []float64,
	keywordEdges []*ExtractedEdge, keywordScores []float64,
	limit int, minScore float64) ([]*ExtractedEdge, []float64, error) {

	const k = 60

	rrfScores := make(map[string]float64)
	edgeMap := make(map[string]*ExtractedEdge)

	for i, edge := range vectorEdges {
		rrfScores[edge.ID] += 1.0 / float64(k+i+1)
		edgeMap[edge.ID] = edge
	}

	for i, edge := range keywordEdges {
		rrfScores[edge.ID] += 1.0 / float64(k+i+1)
		edgeMap[edge.ID] = edge
	}

	type scoredEdge struct {
		edge  *ExtractedEdge
		score float64
	}

	var scored []scoredEdge
	for id, score := range rrfScores {
		if score >= minScore {
			scored = append(scored, scoredEdge{edge: edgeMap[id], score: score})
		}
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	var edges []*ExtractedEdge
	var scores []float64

	for i := 0; i < len(scored) && i < limit; i++ {
		edges = append(edges, scored[i].edge)
		scores = append(scores, scored[i].score)
	}

	return edges, scores, nil
}

// --- Helper methods ---

func (p *PostgresDB) embeddingToString(embedding []float32) string {
	if len(embedding) == 0 {
		return ""
	}
	// Format as vector string: [1.0,2.0,3.0]
	parts := make([]string, len(embedding))
	for i, v := range embedding {
		parts[i] = fmt.Sprintf("%f", v)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

func (p *PostgresDB) parseEmbedding(s string) []float32 {
	if s == "" {
		return nil
	}
	// Remove brackets
	s = strings.TrimPrefix(s, "[")
	s = strings.TrimSuffix(s, "]")

	parts := strings.Split(s, ",")
	embedding := make([]float32, len(parts))

	for i, part := range parts {
		v, _ := strconv.ParseFloat(strings.TrimSpace(part), 64)
		embedding[i] = float32(v)
	}

	return embedding
}

// parseEmbeddingJSON parses an embedding stored as JSONB array in DoltGres.
// Format: [1.0, 2.0, 3.0, ...] (JSON array)
func (p *PostgresDB) parseEmbeddingJSON(s string) []float32 {
	if s == "" {
		return nil
	}

	// Try parsing as JSON array first
	var floats []float64
	if err := json.Unmarshal([]byte(s), &floats); err == nil {
		embedding := make([]float32, len(floats))
		for i, v := range floats {
			embedding[i] = float32(v)
		}
		return embedding
	}

	// Fall back to vector format parsing
	return p.parseEmbedding(s)
}

func (p *PostgresDB) scanNodes(rows *sql.Rows) ([]*ExtractedNode, error) {
	var nodes []*ExtractedNode

	for rows.Next() {
		var n ExtractedNode
		var embeddingStr sql.NullString
		var normalizedName sql.NullString

		if err := rows.Scan(&n.ID, &n.SourceID, &n.GroupID, &n.Name, &normalizedName, &n.Type, &n.Description,
			&embeddingStr, &n.ChunkIndex, &n.CreatedAt); err != nil {
			return nil, err
		}

		if normalizedName.Valid {
			n.NormalizedName = normalizedName.String
		}

		if embeddingStr.Valid {
			n.Embedding = p.parseEmbedding(embeddingStr.String)
		}

		nodes = append(nodes, &n)
	}

	return nodes, nil
}

func (p *PostgresDB) scanEdges(rows *sql.Rows) ([]*ExtractedEdge, error) {
	var edges []*ExtractedEdge

	for rows.Next() {
		var e ExtractedEdge
		var embeddingStr sql.NullString

		if err := rows.Scan(&e.ID, &e.SourceID, &e.GroupID, &e.SourceNodeName, &e.SourceNodeType, &e.TargetNodeName, &e.TargetNodeType,
			&e.Relation, &e.Description, &embeddingStr, &e.Weight, &e.ChunkIndex, &e.CreatedAt); err != nil {
			return nil, err
		}

		if embeddingStr.Valid {
			e.Embedding = p.parseEmbedding(embeddingStr.String)
		}

		edges = append(edges, &e)
	}

	return edges, nil
}
