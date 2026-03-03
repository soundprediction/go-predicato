package factstore

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	_ "github.com/lib/pq"
	"github.com/soundprediction/predicato/pkg/utils"
)

// PostgresDB implements FactsDB using PostgreSQL with VectorChord extension.
type PostgresDB struct {
	db                  *sql.DB
	embeddingDimensions int
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
	return NewPostgresDBWithConfig(connectionString, embeddingDimensions, nil)
}

// NewPostgresDBWithConfig creates a new PostgresDB instance with custom configuration.
// If config is nil, default configuration values are used.
func NewPostgresDBWithConfig(connectionString string, embeddingDimensions int, config *PostgresDBConfig) (*PostgresDB, error) {
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
	}
}

func (p *PostgresDB) Initialize(ctx context.Context) error {
	// Enable VectorChord extension
	if _, err := p.db.ExecContext(ctx, "CREATE EXTENSION IF NOT EXISTS vector"); err != nil {
		return fmt.Errorf("failed to create vector extension: %w", err)
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
	nodesTable := fmt.Sprintf(`
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
	if _, err := p.db.ExecContext(ctx, nodesTable); err != nil {
		return fmt.Errorf("failed to create extracted_nodes table: %w", err)
	}

	// Create extracted_triples table (unified: replaces former extracted_edges + extracted_triples)
	triplesTable := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS extracted_triples (
			id VARCHAR(255) PRIMARY KEY,
			source_id VARCHAR(255) REFERENCES sources(id),
			group_id VARCHAR(255),
			subject TEXT,
			subject_type VARCHAR(50),
			predicate TEXT,
			object TEXT,
			object_type VARCHAR(50),
			description TEXT,
			condition TEXT,
			temporal TEXT,
			location TEXT,
			certainty TEXT,
			scope TEXT,
			source_attribution TEXT,
			confidence FLOAT,
			embedding vector(%d),
			chunk_index INT,
			model VARCHAR(255),
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		)`, p.embeddingDimensions)
	if _, err := p.db.ExecContext(ctx, triplesTable); err != nil {
		return fmt.Errorf("failed to create extracted_triples table: %w", err)
	}

	// Create indices for better query performance
	indices := []string{
		"CREATE INDEX IF NOT EXISTS idx_sources_group ON sources(group_id)",
		"CREATE INDEX IF NOT EXISTS idx_nodes_source ON extracted_nodes(source_id)",
		"CREATE INDEX IF NOT EXISTS idx_nodes_group ON extracted_nodes(group_id)",
		"CREATE INDEX IF NOT EXISTS idx_nodes_type ON extracted_nodes(type)",
		"CREATE INDEX IF NOT EXISTS idx_triples_source ON extracted_triples(source_id)",
		"CREATE INDEX IF NOT EXISTS idx_triples_group ON extracted_triples(group_id)",
		"CREATE INDEX IF NOT EXISTS idx_rules_source ON extracted_rules(source_id)",
	}

	for _, idx := range indices {
		if _, err := p.db.ExecContext(ctx, idx); err != nil {
			// Log warning but don't fail - indices are optional
			fmt.Printf("Warning: failed to create index: %v\n", err)
		}
	}

	// Create GIN indices for full-text search (keyword search performance)
	ginIndices := []string{
		`CREATE INDEX IF NOT EXISTS idx_nodes_fts ON extracted_nodes
		 USING GIN (to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(description, '')))`,
		`CREATE INDEX IF NOT EXISTS idx_triples_fts ON extracted_triples
		 USING GIN (to_tsvector('english', COALESCE(subject, '') || ' ' || COALESCE(predicate, '') || ' ' || COALESCE(object, '') || ' ' || COALESCE(description, '')))`,
		`CREATE INDEX IF NOT EXISTS idx_sources_fts ON sources
		 USING GIN (to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(content, '')))`,
	}

	for _, idx := range ginIndices {
		if _, err := p.db.ExecContext(ctx, idx); err != nil {
			fmt.Printf("Warning: failed to create GIN index (keyword search may be slower): %v\n", err)
		}
	}

	// Create extracted_rules table
	rulesTable := fmt.Sprintf(`
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

	// --- Model provenance migration ---
	// Track which LLM model produced each extracted item
	modelMigrations := []string{
		`ALTER TABLE extracted_nodes ADD COLUMN IF NOT EXISTS model VARCHAR(255)`,
		`ALTER TABLE extracted_rules ADD COLUMN IF NOT EXISTS model VARCHAR(255)`,
	}
	for _, migration := range modelMigrations {
		if _, err := p.db.ExecContext(ctx, migration); err != nil {
			fmt.Printf("Warning: failed to add model column: %v\n", err)
		}
	}

	// --- Consolidation migration: add new triple columns if missing ---
	tripleConsolidation := []string{
		`ALTER TABLE extracted_triples ADD COLUMN IF NOT EXISTS group_id VARCHAR(255)`,
		`ALTER TABLE extracted_triples ADD COLUMN IF NOT EXISTS subject_type VARCHAR(50)`,
		`ALTER TABLE extracted_triples ADD COLUMN IF NOT EXISTS object_type VARCHAR(50)`,
		`ALTER TABLE extracted_triples ADD COLUMN IF NOT EXISTS description TEXT`,
		`ALTER TABLE extracted_triples ADD COLUMN IF NOT EXISTS model VARCHAR(255)`,
	}
	for _, migration := range tripleConsolidation {
		if _, err := p.db.ExecContext(ctx, migration); err != nil {
			fmt.Printf("Warning: failed to add triple consolidation column: %v\n", err)
		}
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

	tripleIdx := fmt.Sprintf(`
		CREATE INDEX IF NOT EXISTS idx_triples_embedding
		ON extracted_triples USING ivfflat (embedding vector_cosine_ops)
		WITH (lists = %d)`, lists)
	if _, err := p.db.ExecContext(ctx, tripleIdx); err != nil {
		return fmt.Errorf("failed to create triple vector index: %w", err)
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

func (p *PostgresDB) SaveExtractedKnowledge(ctx context.Context, sourceID string, nodes []*ExtractedNode, triples []*ExtractedTriple) error {
	tx, err := p.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	// Get source's group_id for nodes/triples
	var groupID string
	err = tx.QueryRowContext(ctx, "SELECT group_id FROM sources WHERE id = $1", sourceID).Scan(&groupID)
	if err != nil && err != sql.ErrNoRows {
		return fmt.Errorf("failed to get source group_id: %w", err)
	}

	// Track original node ID -> canonical (deduplicated) node ID
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
				INSERT INTO extracted_nodes (id, source_id, group_id, name, normalized_name, type, description, embedding, chunk_index, model, created_at)
				VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
				ON CONFLICT (id) DO UPDATE SET
					name = EXCLUDED.name,
					normalized_name = EXCLUDED.normalized_name,
					type = EXCLUDED.type,
					description = EXCLUDED.description,
					embedding = EXCLUDED.embedding,
					chunk_index = EXCLUDED.chunk_index,
					model = EXCLUDED.model`,
				canonicalID, sourceID, nodeGroupID, node.Name, normalizedName, node.Type, node.Description,
				embeddingVal, node.ChunkIndex, node.Model, createdAt); err != nil {
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

	// Insert triples
	if len(triples) > 0 {
		tripleStmt, err := tx.PrepareContext(ctx, `
			INSERT INTO extracted_triples (id, source_id, group_id, subject, subject_type, predicate, object, object_type,
				description, condition, temporal, location, certainty, scope, source_attribution,
				confidence, embedding, chunk_index, model, created_at)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
			ON CONFLICT (id) DO UPDATE SET
				subject = EXCLUDED.subject,
				subject_type = EXCLUDED.subject_type,
				predicate = EXCLUDED.predicate,
				object = EXCLUDED.object,
				object_type = EXCLUDED.object_type,
				description = EXCLUDED.description,
				condition = EXCLUDED.condition,
				temporal = EXCLUDED.temporal,
				location = EXCLUDED.location,
				certainty = EXCLUDED.certainty,
				scope = EXCLUDED.scope,
				source_attribution = EXCLUDED.source_attribution,
				confidence = EXCLUDED.confidence,
				embedding = EXCLUDED.embedding,
				chunk_index = EXCLUDED.chunk_index,
				model = EXCLUDED.model`)
		if err != nil {
			return fmt.Errorf("failed to prepare triple statement: %w", err)
		}
		defer tripleStmt.Close()

		for _, t := range triples {
			var embeddingVal interface{}
			if len(t.Embedding) > 0 {
				embeddingVal = p.embeddingToString(t.Embedding)
			}
			tripleGroupID := t.GroupID
			if tripleGroupID == "" {
				tripleGroupID = groupID
			}
			createdAt := t.CreatedAt
			if createdAt.IsZero() {
				createdAt = time.Now()
			}

			if _, err := tripleStmt.ExecContext(ctx,
				t.ID, sourceID, tripleGroupID, t.Subject, t.SubjectType, t.Predicate, t.Object, t.ObjectType,
				t.Description, t.Condition, t.Temporal, t.Location, t.Certainty, t.Scope, t.SourceAttribution,
				t.Confidence, embeddingVal, t.ChunkIndex, t.Model, createdAt); err != nil {
				return fmt.Errorf("failed to insert triple %s: %w", t.ID, err)
			}
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
		       en.description, en.embedding, en.chunk_index, en.model, en.created_at
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

func (p *PostgresDB) GetExtractedTriples(ctx context.Context, sourceID string) ([]*ExtractedTriple, error) {
	rows, err := p.db.QueryContext(ctx, `
		SELECT id, source_id, group_id, subject, subject_type, predicate, object, object_type,
		       description, condition, temporal, location, certainty, scope, source_attribution,
		       confidence, embedding, chunk_index, model, created_at
		FROM extracted_triples WHERE source_id = $1`, sourceID)
	if err != nil {
		return nil, fmt.Errorf("failed to query extracted triples: %w", err)
	}
	defer rows.Close()

	return p.scanTriples(rows)
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
	query := "SELECT id, source_id, group_id, name, normalized_name, type, description, embedding, chunk_index, model, created_at FROM extracted_nodes"
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

func (p *PostgresDB) GetAllTriples(ctx context.Context, limit int) ([]*ExtractedTriple, error) {
	query := `SELECT id, source_id, group_id, subject, subject_type, predicate, object, object_type,
	                 description, condition, temporal, location, certainty, scope, source_attribution,
	                 confidence, embedding, chunk_index, model, created_at
	          FROM extracted_triples`
	if limit > 0 {
		query += fmt.Sprintf(" LIMIT %d", limit)
	}

	rows, err := p.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to query triples: %w", err)
	}
	defer rows.Close()

	return p.scanTriples(rows)
}

func (p *PostgresDB) GetStats(ctx context.Context) (*Stats, error) {
	stats := &Stats{}

	if err := p.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM sources").Scan(&stats.SourceCount); err != nil {
		return nil, fmt.Errorf("failed to count sources: %w", err)
	}
	if err := p.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM extracted_nodes").Scan(&stats.NodeCount); err != nil {
		return nil, fmt.Errorf("failed to count nodes: %w", err)
	}
	if err := p.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM extracted_triples").Scan(&stats.TripleCount); err != nil {
		return nil, fmt.Errorf("failed to count triples: %w", err)
	}
	_ = p.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM extracted_rules").Scan(&stats.RuleCount)

	return stats, nil
}

func (p *PostgresDB) Close() error {
	return p.db.Close()
}

// --- Extended Extraction Methods ---

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
		INSERT INTO extracted_rules (id, source_id, antecedent, consequent, exception, rule_type, scope, source_attribution, confidence, embedding, chunk_index, model, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
		ON CONFLICT (id) DO UPDATE SET
			antecedent = EXCLUDED.antecedent,
			consequent = EXCLUDED.consequent,
			exception = EXCLUDED.exception,
			rule_type = EXCLUDED.rule_type,
			scope = EXCLUDED.scope,
			source_attribution = EXCLUDED.source_attribution,
			confidence = EXCLUDED.confidence,
			embedding = EXCLUDED.embedding,
			chunk_index = EXCLUDED.chunk_index,
			model = EXCLUDED.model`)
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
			r.Confidence, embeddingVal, r.ChunkIndex, r.Model, createdAt); err != nil {
			return fmt.Errorf("failed to insert rule %s: %w", r.ID, err)
		}
	}

	return tx.Commit()
}

func (p *PostgresDB) GetExtractedRules(ctx context.Context, sourceID string) ([]*ExtractedRule, error) {
	rows, err := p.db.QueryContext(ctx, `
		SELECT id, source_id, antecedent, consequent, exception, rule_type, scope,
		       source_attribution, confidence, embedding, chunk_index, model, created_at
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

		var model sql.NullString
		if err := rows.Scan(&r.ID, &r.SourceID, &r.Antecedent, &r.Consequent,
			&exception, &ruleType, &scope, &sourceAttr,
			&confidence, &embeddingStr, &r.ChunkIndex, &model, &r.CreatedAt); err != nil {
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
		if model.Valid {
			r.Model = model.String
		}

		rules = append(rules, &r)
	}
	return rules, nil
}

// --- Paginated Methods ---

func (p *PostgresDB) GetExtractedNodesPaginated(ctx context.Context, sourceID string, offset, limit int) ([]*ExtractedNode, error) {
	rows, err := p.db.QueryContext(ctx, `
		SELECT en.id, en.source_id, en.group_id, en.name, en.normalized_name, en.type,
		       en.description, en.embedding, en.chunk_index, en.model, en.created_at
		FROM extracted_nodes en
		JOIN node_sources ns ON ns.node_id = en.id
		WHERE ns.source_id = $1
		ORDER BY en.id
		LIMIT $2 OFFSET $3`,
		sourceID, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to query extracted nodes (paginated): %w", err)
	}
	defer rows.Close()

	return p.scanNodes(rows)
}

func (p *PostgresDB) GetExtractedTriplesPaginated(ctx context.Context, sourceID string, offset, limit int) ([]*ExtractedTriple, error) {
	rows, err := p.db.QueryContext(ctx, `
		SELECT id, source_id, group_id, subject, subject_type, predicate, object, object_type,
		       description, condition, temporal, location, certainty, scope, source_attribution,
		       confidence, embedding, chunk_index, model, created_at
		FROM extracted_triples WHERE source_id = $1
		ORDER BY id
		LIMIT $2 OFFSET $3`, sourceID, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to query extracted triples (paginated): %w", err)
	}
	defer rows.Close()

	return p.scanTriples(rows)
}

func (p *PostgresDB) GetExtractedRulesPaginated(ctx context.Context, sourceID string, offset, limit int) ([]*ExtractedRule, error) {
	rows, err := p.db.QueryContext(ctx, `
		SELECT id, source_id, antecedent, consequent, exception, rule_type, scope,
		       source_attribution, confidence, embedding, chunk_index, model, created_at
		FROM extracted_rules WHERE source_id = $1
		ORDER BY id
		LIMIT $2 OFFSET $3`, sourceID, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to query extracted rules (paginated): %w", err)
	}
	defer rows.Close()

	var rules []*ExtractedRule
	for rows.Next() {
		var r ExtractedRule
		var embeddingStr sql.NullString
		var exception, ruleType, scope, sourceAttr sql.NullString
		var confidence sql.NullFloat64
		var model sql.NullString
		if err := rows.Scan(&r.ID, &r.SourceID, &r.Antecedent, &r.Consequent,
			&exception, &ruleType, &scope, &sourceAttr,
			&confidence, &embeddingStr, &r.ChunkIndex, &model, &r.CreatedAt); err != nil {
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
		if model.Valid {
			r.Model = model.String
		}
		rules = append(rules, &r)
	}
	return rules, nil
}

func (p *PostgresDB) CountExtractedNodes(ctx context.Context, sourceID string) (int64, error) {
	var count int64
	err := p.db.QueryRowContext(ctx, `
		SELECT COUNT(*) FROM node_sources WHERE source_id = $1`, sourceID).Scan(&count)
	return count, err
}

func (p *PostgresDB) CountExtractedTriples(ctx context.Context, sourceID string) (int64, error) {
	var count int64
	err := p.db.QueryRowContext(ctx, `
		SELECT COUNT(*) FROM extracted_triples WHERE source_id = $1`, sourceID).Scan(&count)
	return count, err
}

func (p *PostgresDB) CountExtractedRules(ctx context.Context, sourceID string) (int64, error) {
	var count int64
	err := p.db.QueryRowContext(ctx, `
		SELECT COUNT(*) FROM extracted_rules WHERE source_id = $1`, sourceID).Scan(&count)
	return count, err
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

func (p *PostgresDB) SearchTriples(ctx context.Context, query string, embedding []float32, config *FactSearchConfig) ([]*ExtractedTriple, []float64, error) {
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
		return []*ExtractedTriple{}, []float64{}, nil
	}

	if useVector && useKeyword {
		vectorTriples, vectorScores, err := p.vectorSearchTriples(ctx, embedding, config)
		if err != nil {
			return nil, nil, err
		}
		keywordTriples, keywordScores, err := p.keywordSearchTriples(ctx, query, config)
		if err != nil {
			return nil, nil, err
		}
		return p.rrfMergeTriples(vectorTriples, vectorScores, keywordTriples, keywordScores, config.Limit, config.MinScore)
	}

	if useVector {
		return p.vectorSearchTriples(ctx, embedding, config)
	}

	return p.keywordSearchTriples(ctx, query, config)
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

	// Search triples
	triples, tripleScores, err := p.SearchTriples(ctx, query, embedding, config)
	if err != nil {
		return nil, fmt.Errorf("triple search failed: %w", err)
	}

	return &FactSearchResults{
		Nodes:        nodes,
		Triples:      triples,
		NodeScores:   nodeScores,
		TripleScores: tripleScores,
		Query:        query,
		Total:        len(nodes) + len(triples),
	}, nil
}

// --- Internal search methods ---

func (p *PostgresDB) vectorSearchNodes(ctx context.Context, embedding []float32, config *FactSearchConfig) ([]*ExtractedNode, []float64, error) {
	embeddingStr := p.embeddingToString(embedding)

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

func (p *PostgresDB) vectorSearchTriples(ctx context.Context, embedding []float32, config *FactSearchConfig) ([]*ExtractedTriple, []float64, error) {
	embeddingStr := p.embeddingToString(embedding)

	sqlQuery := `
		SELECT id, source_id, group_id, subject, subject_type, predicate, object, object_type,
		       description, condition, temporal, location, certainty, scope, source_attribution,
		       confidence, embedding, chunk_index, model, created_at,
		       1 - (embedding <=> $1::vector) AS score
		FROM extracted_triples
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
		return nil, nil, fmt.Errorf("failed to execute triple vector search: %w", err)
	}
	defer rows.Close()

	var triples []*ExtractedTriple
	var scores []float64

	for rows.Next() {
		var t ExtractedTriple
		var embStr sql.NullString
		var subjectType, objectType, description, condition, temporal, location, certainty, scope, sourceAttr, model sql.NullString
		var confidence sql.NullFloat64
		var score float64

		if err := rows.Scan(&t.ID, &t.SourceID, &t.GroupID, &t.Subject, &subjectType, &t.Predicate, &t.Object, &objectType,
			&description, &condition, &temporal, &location, &certainty, &scope, &sourceAttr,
			&confidence, &embStr, &t.ChunkIndex, &model, &t.CreatedAt, &score); err != nil {
			return nil, nil, err
		}

		if score < config.MinScore {
			continue
		}

		if subjectType.Valid {
			t.SubjectType = subjectType.String
		}
		if objectType.Valid {
			t.ObjectType = objectType.String
		}
		if description.Valid {
			t.Description = description.String
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
		if embStr.Valid {
			t.Embedding = p.parseEmbedding(embStr.String)
		}
		if model.Valid {
			t.Model = model.String
		}

		triples = append(triples, &t)
		scores = append(scores, score)

		if len(triples) >= config.Limit {
			break
		}
	}

	return triples, scores, nil
}

func (p *PostgresDB) keywordSearchTriples(ctx context.Context, query string, config *FactSearchConfig) ([]*ExtractedTriple, []float64, error) {
	sqlQuery := `
		SELECT id, source_id, group_id, subject, subject_type, predicate, object, object_type,
		       description, condition, temporal, location, certainty, scope, source_attribution,
		       confidence, embedding, chunk_index, model, created_at,
		       ts_rank(to_tsvector('english', COALESCE(subject, '') || ' ' || COALESCE(predicate, '') || ' ' || COALESCE(object, '') || ' ' || COALESCE(description, '')),
		              plainto_tsquery('english', $1)) AS score
		FROM extracted_triples
		WHERE to_tsvector('english', COALESCE(subject, '') || ' ' || COALESCE(predicate, '') || ' ' || COALESCE(object, '') || ' ' || COALESCE(description, ''))
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
		return nil, nil, fmt.Errorf("failed to execute triple keyword search: %w", err)
	}
	defer rows.Close()

	var triples []*ExtractedTriple
	var scores []float64

	for rows.Next() {
		var t ExtractedTriple
		var embStr sql.NullString
		var subjectType, objectType, description, condition, temporal, location, certainty, scope, sourceAttr, model sql.NullString
		var confidence sql.NullFloat64
		var score float64

		if err := rows.Scan(&t.ID, &t.SourceID, &t.GroupID, &t.Subject, &subjectType, &t.Predicate, &t.Object, &objectType,
			&description, &condition, &temporal, &location, &certainty, &scope, &sourceAttr,
			&confidence, &embStr, &t.ChunkIndex, &model, &t.CreatedAt, &score); err != nil {
			return nil, nil, err
		}

		if score < config.MinScore {
			continue
		}

		if subjectType.Valid {
			t.SubjectType = subjectType.String
		}
		if objectType.Valid {
			t.ObjectType = objectType.String
		}
		if description.Valid {
			t.Description = description.String
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
		if embStr.Valid {
			t.Embedding = p.parseEmbedding(embStr.String)
		}
		if model.Valid {
			t.Model = model.String
		}

		triples = append(triples, &t)
		scores = append(scores, score)

		if len(triples) >= config.Limit {
			break
		}
	}

	return triples, scores, nil
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

func (p *PostgresDB) rrfMergeTriples(vectorTriples []*ExtractedTriple, vectorScores []float64,
	keywordTriples []*ExtractedTriple, keywordScores []float64,
	limit int, minScore float64) ([]*ExtractedTriple, []float64, error) {

	const k = 60

	rrfScores := make(map[string]float64)
	tripleMap := make(map[string]*ExtractedTriple)

	for i, t := range vectorTriples {
		rrfScores[t.ID] += 1.0 / float64(k+i+1)
		tripleMap[t.ID] = t
	}

	for i, t := range keywordTriples {
		rrfScores[t.ID] += 1.0 / float64(k+i+1)
		tripleMap[t.ID] = t
	}

	type scoredTriple struct {
		triple *ExtractedTriple
		score  float64
	}

	var scored []scoredTriple
	for id, score := range rrfScores {
		if score >= minScore {
			scored = append(scored, scoredTriple{triple: tripleMap[id], score: score})
		}
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	var triples []*ExtractedTriple
	var scores []float64

	for i := 0; i < len(scored) && i < limit; i++ {
		triples = append(triples, scored[i].triple)
		scores = append(scores, scored[i].score)
	}

	return triples, scores, nil
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

func (p *PostgresDB) scanNodes(rows *sql.Rows) ([]*ExtractedNode, error) {
	var nodes []*ExtractedNode

	for rows.Next() {
		var n ExtractedNode
		var embeddingStr sql.NullString
		var normalizedName sql.NullString

		var model sql.NullString
		if err := rows.Scan(&n.ID, &n.SourceID, &n.GroupID, &n.Name, &normalizedName, &n.Type, &n.Description,
			&embeddingStr, &n.ChunkIndex, &model, &n.CreatedAt); err != nil {
			return nil, err
		}

		if normalizedName.Valid {
			n.NormalizedName = normalizedName.String
		}
		if model.Valid {
			n.Model = model.String
		}

		if embeddingStr.Valid {
			n.Embedding = p.parseEmbedding(embeddingStr.String)
		}

		nodes = append(nodes, &n)
	}

	return nodes, nil
}

func (p *PostgresDB) scanTriples(rows *sql.Rows) ([]*ExtractedTriple, error) {
	var triples []*ExtractedTriple

	for rows.Next() {
		var t ExtractedTriple
		var embStr sql.NullString
		var subjectType, objectType, description, condition, temporal, location, certainty, scope, sourceAttr, model sql.NullString
		var confidence sql.NullFloat64

		if err := rows.Scan(&t.ID, &t.SourceID, &t.GroupID, &t.Subject, &subjectType, &t.Predicate, &t.Object, &objectType,
			&description, &condition, &temporal, &location, &certainty, &scope, &sourceAttr,
			&confidence, &embStr, &t.ChunkIndex, &model, &t.CreatedAt); err != nil {
			return nil, err
		}

		if subjectType.Valid {
			t.SubjectType = subjectType.String
		}
		if objectType.Valid {
			t.ObjectType = objectType.String
		}
		if description.Valid {
			t.Description = description.String
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
		if embStr.Valid {
			t.Embedding = p.parseEmbedding(embStr.String)
		}
		if model.Valid {
			t.Model = model.String
		}

		triples = append(triples, &t)
	}

	return triples, nil
}

