package factstore

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	_ "github.com/dolthub/driver"
)

// DoltDB implements FactsDB using a Dolt SQL database.
// Deprecated: Use PostgresDB with DoltGres for better search capabilities including VectorChord.
// DoltDB is maintained for backward compatibility but uses in-memory vector search.
type DoltDB struct {
	db *sql.DB
}

// DoltDBConfig holds configuration options for DoltDB connection pool.
type DoltDBConfig struct {
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

// DefaultDoltDBConfig returns the default DoltDB configuration.
func DefaultDoltDBConfig() *DoltDBConfig {
	return &DoltDBConfig{
		MaxOpenConns:    25,
		MaxIdleConns:    5,
		ConnMaxLifetime: 5 * time.Minute,
	}
}

// NewDoltDB creates a new DoltDB instance.
// Deprecated: Use NewPostgresDB with DoltGres for better search capabilities.
// connectionString should be a valid Dolt DSN, e.g., "file:///path/to/databases?commitname=User&commitemail=user@example.com&database=mydb"
func NewDoltDB(connectionString string) (*DoltDB, error) {
	return NewDoltDBWithConfig(connectionString, nil)
}

// NewDoltDBWithConfig creates a new DoltDB instance with custom configuration.
// Deprecated: Use NewPostgresDB with DoltGres for better search capabilities.
// If config is nil, default configuration values are used.
func NewDoltDBWithConfig(connectionString string, config *DoltDBConfig) (*DoltDB, error) {
	fmt.Println("Warning: DoltDB is deprecated. Consider using PostgresDB with DoltGres for VectorChord support.")

	if config == nil {
		config = DefaultDoltDBConfig()
	}

	db, err := sql.Open("dolt", connectionString)
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
		`CREATE TABLE IF NOT EXISTS extracted_nodes (
			id VARCHAR(255) PRIMARY KEY,
			source_id VARCHAR(255),
			group_id VARCHAR(255),
			name TEXT,
			type VARCHAR(50),
			description TEXT,
			embedding JSON,
			chunk_index INT,
			created_at TIMESTAMP,
			FOREIGN KEY (source_id) REFERENCES sources(id)
		)`,
		`CREATE TABLE IF NOT EXISTS extracted_edges (
			id VARCHAR(255) PRIMARY KEY,
			source_id VARCHAR(255),
			group_id VARCHAR(255),
			source_node_name TEXT,
			target_node_name TEXT,
			relation TEXT,
			description TEXT,
			embedding JSON,
			weight FLOAT,
			chunk_index INT,
			created_at TIMESTAMP,
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
	defer func() { _ = tx.Rollback() }()

	// Get source's group_id for nodes/edges
	var groupID string
	err = tx.QueryRowContext(ctx, "SELECT group_id FROM sources WHERE id = ?", sourceID).Scan(&groupID)
	if err != nil && err != sql.ErrNoRows {
		return fmt.Errorf("failed to get source group_id: %w", err)
	}

	nodeStmt, err := tx.PrepareContext(ctx, `INSERT INTO extracted_nodes (id, source_id, group_id, name, type, description, embedding, chunk_index, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return fmt.Errorf("failed to prepare node statement: %w", err)
	}
	defer nodeStmt.Close()

	for _, node := range nodes {
		embeddingJSON, err := json.Marshal(node.Embedding)
		if err != nil {
			return fmt.Errorf("failed to marshal embedding for node %s: %w", node.ID, err)
		}
		nodeGroupID := node.GroupID
		if nodeGroupID == "" {
			nodeGroupID = groupID
		}
		createdAt := node.CreatedAt
		if createdAt.IsZero() {
			createdAt = time.Now()
		}
		if _, err := nodeStmt.ExecContext(ctx, node.ID, sourceID, nodeGroupID, node.Name, node.Type, node.Description, embeddingJSON, node.ChunkIndex, createdAt); err != nil {
			return fmt.Errorf("failed to insert node %s: %w", node.ID, err)
		}
	}

	edgeStmt, err := tx.PrepareContext(ctx, `INSERT INTO extracted_edges (id, source_id, group_id, source_node_name, target_node_name, relation, description, embedding, weight, chunk_index, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return fmt.Errorf("failed to prepare edge statement: %w", err)
	}
	defer edgeStmt.Close()

	for _, edge := range edges {
		embeddingJSON, err := json.Marshal(edge.Embedding)
		if err != nil {
			return fmt.Errorf("failed to marshal embedding for edge %s: %w", edge.ID, err)
		}
		edgeGroupID := edge.GroupID
		if edgeGroupID == "" {
			edgeGroupID = groupID
		}
		createdAt := edge.CreatedAt
		if createdAt.IsZero() {
			createdAt = time.Now()
		}
		if _, err := edgeStmt.ExecContext(ctx, edge.ID, sourceID, edgeGroupID, edge.SourceNodeName, edge.TargetNodeName, edge.Relation, edge.Description, embeddingJSON, edge.Weight, edge.ChunkIndex, createdAt); err != nil {
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
	rows, err := d.db.QueryContext(ctx, "SELECT id, source_id, group_id, name, type, description, embedding, chunk_index, created_at FROM extracted_nodes WHERE source_id = ?", sourceID)
	if err != nil {
		return nil, fmt.Errorf("failed to query extracted nodes: %w", err)
	}
	defer rows.Close()

	var nodes []*ExtractedNode
	for rows.Next() {
		var n ExtractedNode
		var embeddingBytes []byte
		var groupID sql.NullString
		var createdAt sql.NullTime
		if err := rows.Scan(&n.ID, &n.SourceID, &groupID, &n.Name, &n.Type, &n.Description, &embeddingBytes, &n.ChunkIndex, &createdAt); err != nil {
			return nil, err
		}
		if groupID.Valid {
			n.GroupID = groupID.String
		}
		if createdAt.Valid {
			n.CreatedAt = createdAt.Time
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
	rows, err := d.db.QueryContext(ctx, "SELECT id, source_id, group_id, source_node_name, target_node_name, relation, description, embedding, weight, chunk_index, created_at FROM extracted_edges WHERE source_id = ?", sourceID)
	if err != nil {
		return nil, fmt.Errorf("failed to query extracted edges: %w", err)
	}
	defer rows.Close()

	var edges []*ExtractedEdge
	for rows.Next() {
		var e ExtractedEdge
		var embeddingBytes []byte
		var groupID sql.NullString
		var createdAt sql.NullTime
		if err := rows.Scan(&e.ID, &e.SourceID, &groupID, &e.SourceNodeName, &e.TargetNodeName, &e.Relation, &e.Description, &embeddingBytes, &e.Weight, &e.ChunkIndex, &createdAt); err != nil {
			return nil, err
		}
		if groupID.Valid {
			e.GroupID = groupID.String
		}
		if createdAt.Valid {
			e.CreatedAt = createdAt.Time
		}
		if len(embeddingBytes) > 0 {
			if err := json.Unmarshal(embeddingBytes, &e.Embedding); err != nil {
				return nil, fmt.Errorf("failed to unmarshal embedding: %w", err)
			}
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
	query := "SELECT id, source_id, group_id, name, type, description, embedding, chunk_index, created_at FROM extracted_nodes"
	var rows *sql.Rows
	var err error
	if limit > 0 {
		query += " LIMIT ?"
		rows, err = d.db.QueryContext(ctx, query, limit)
	} else {
		rows, err = d.db.QueryContext(ctx, query)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to query nodes: %w", err)
	}
	defer rows.Close()

	var nodes []*ExtractedNode
	for rows.Next() {
		var n ExtractedNode
		var embeddingBytes []byte
		var groupID sql.NullString
		var createdAt sql.NullTime
		if err := rows.Scan(&n.ID, &n.SourceID, &groupID, &n.Name, &n.Type, &n.Description, &embeddingBytes, &n.ChunkIndex, &createdAt); err != nil {
			return nil, err
		}
		if groupID.Valid {
			n.GroupID = groupID.String
		}
		if createdAt.Valid {
			n.CreatedAt = createdAt.Time
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
	query := "SELECT id, source_id, group_id, source_node_name, target_node_name, relation, description, embedding, weight, chunk_index, created_at FROM extracted_edges"
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
		var embeddingBytes []byte
		var groupID sql.NullString
		var createdAt sql.NullTime
		if err := rows.Scan(&e.ID, &e.SourceID, &groupID, &e.SourceNodeName, &e.TargetNodeName, &e.Relation, &e.Description, &embeddingBytes, &e.Weight, &e.ChunkIndex, &createdAt); err != nil {
			return nil, err
		}
		if groupID.Valid {
			e.GroupID = groupID.String
		}
		if createdAt.Valid {
			e.CreatedAt = createdAt.Time
		}
		if len(embeddingBytes) > 0 {
			if err := json.Unmarshal(embeddingBytes, &e.Embedding); err != nil {
				return nil, fmt.Errorf("failed to unmarshal embedding: %w", err)
			}
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
	if err := d.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM extracted_nodes").Scan(&stats.NodeCount); err != nil {
		return nil, fmt.Errorf("failed to count nodes: %w", err)
	}
	if err := d.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM extracted_edges").Scan(&stats.EdgeCount); err != nil {
		return nil, fmt.Errorf("failed to count edges: %w", err)
	}

	return stats, nil
}

func (d *DoltDB) Close() error {
	return d.db.Close()
}

// --- Search Methods (In-memory implementation for DoltDB) ---
// Note: These methods use in-memory search since Dolt doesn't have VectorChord.
// For production use with large datasets, migrate to PostgresDB/DoltGres with VectorChord.

func (d *DoltDB) SearchNodes(ctx context.Context, query string, embedding []float32, config *FactSearchConfig) ([]*ExtractedNode, []float64, error) {
	if config == nil {
		config = &FactSearchConfig{Limit: 10}
	}
	if config.Limit <= 0 {
		config.Limit = 10
	}

	// Load all nodes for in-memory search
	allNodes, err := d.GetAllNodes(ctx, 0)
	if err != nil {
		return nil, nil, err
	}

	// Filter and score nodes
	type scoredNode struct {
		node  *ExtractedNode
		score float64
	}

	var scored []scoredNode

	for _, node := range allNodes {
		// Apply filters
		if config.GroupID != "" && node.GroupID != config.GroupID {
			continue
		}
		if len(config.NodeTypes) > 0 {
			found := false
			for _, t := range config.NodeTypes {
				if node.Type == t {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}
		if config.TimeRange != nil {
			if !config.TimeRange.Start.IsZero() && node.CreatedAt.Before(config.TimeRange.Start) {
				continue
			}
			if !config.TimeRange.End.IsZero() && node.CreatedAt.After(config.TimeRange.End) {
				continue
			}
		}

		var score float64 = 0

		// Vector similarity
		if len(embedding) > 0 && len(node.Embedding) > 0 {
			score = cosineSimilarity(embedding, node.Embedding)
		}

		// Keyword matching (simple approach)
		if query != "" {
			queryLower := strings.ToLower(query)
			textLower := strings.ToLower(node.Name + " " + node.Description)
			if strings.Contains(textLower, queryLower) {
				score += 0.5 // Boost for keyword match
			}
		}

		if score >= config.MinScore {
			scored = append(scored, scoredNode{node: node, score: score})
		}
	}

	// Sort by score descending using O(n log n) sort
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Extract results
	var nodes []*ExtractedNode
	var scores []float64

	for i := 0; i < len(scored) && i < config.Limit; i++ {
		nodes = append(nodes, scored[i].node)
		scores = append(scores, scored[i].score)
	}

	return nodes, scores, nil
}

func (d *DoltDB) SearchEdges(ctx context.Context, query string, embedding []float32, config *FactSearchConfig) ([]*ExtractedEdge, []float64, error) {
	if config == nil {
		config = &FactSearchConfig{Limit: 10}
	}
	if config.Limit <= 0 {
		config.Limit = 10
	}

	allEdges, err := d.GetAllEdges(ctx, 0)
	if err != nil {
		return nil, nil, err
	}

	type scoredEdge struct {
		edge  *ExtractedEdge
		score float64
	}

	var scored []scoredEdge

	for _, edge := range allEdges {
		if config.GroupID != "" && edge.GroupID != config.GroupID {
			continue
		}
		if config.TimeRange != nil {
			if !config.TimeRange.Start.IsZero() && edge.CreatedAt.Before(config.TimeRange.Start) {
				continue
			}
			if !config.TimeRange.End.IsZero() && edge.CreatedAt.After(config.TimeRange.End) {
				continue
			}
		}

		var score float64 = 0

		// Vector similarity
		if len(embedding) > 0 && len(edge.Embedding) > 0 {
			score = cosineSimilarity(embedding, edge.Embedding)
		}

		// Keyword matching
		if query != "" {
			queryLower := strings.ToLower(query)
			textLower := strings.ToLower(edge.Relation + " " + edge.Description)
			if strings.Contains(textLower, queryLower) {
				score += 0.5
			}
		}

		if score >= config.MinScore {
			scored = append(scored, scoredEdge{edge: edge, score: score})
		}
	}

	// Sort using O(n log n) sort
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	var edges []*ExtractedEdge
	var scores []float64

	for i := 0; i < len(scored) && i < config.Limit; i++ {
		edges = append(edges, scored[i].edge)
		scores = append(scores, scored[i].score)
	}

	return edges, scores, nil
}

func (d *DoltDB) SearchSources(ctx context.Context, query string, config *FactSearchConfig) ([]*Source, []float64, error) {
	if config == nil {
		config = &FactSearchConfig{Limit: 10}
	}
	if query == "" {
		return []*Source{}, []float64{}, nil
	}

	allSources, err := d.GetAllSources(ctx, 0)
	if err != nil {
		return nil, nil, err
	}

	type scoredSource struct {
		source *Source
		score  float64
	}

	var scored []scoredSource
	queryLower := strings.ToLower(query)

	for _, source := range allSources {
		if config.GroupID != "" && source.GroupID != config.GroupID {
			continue
		}
		if config.TimeRange != nil {
			if !config.TimeRange.Start.IsZero() && source.CreatedAt.Before(config.TimeRange.Start) {
				continue
			}
			if !config.TimeRange.End.IsZero() && source.CreatedAt.After(config.TimeRange.End) {
				continue
			}
		}

		textLower := strings.ToLower(source.Name + " " + source.Content)
		if strings.Contains(textLower, queryLower) {
			// Simple scoring based on match count
			score := float64(strings.Count(textLower, queryLower)) * 0.1
			if score < 0.1 {
				score = 0.1
			}
			if score >= config.MinScore {
				scored = append(scored, scoredSource{source: source, score: score})
			}
		}
	}

	// Sort
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	var sources []*Source
	var scores []float64

	for i := 0; i < len(scored) && i < config.Limit; i++ {
		sources = append(sources, scored[i].source)
		scores = append(scores, scored[i].score)
	}

	return sources, scores, nil
}

func (d *DoltDB) HybridSearch(ctx context.Context, query string, embedding []float32, config *FactSearchConfig) (*FactSearchResults, error) {
	if config == nil {
		config = &FactSearchConfig{Limit: 10}
	}

	nodes, nodeScores, err := d.SearchNodes(ctx, query, embedding, config)
	if err != nil {
		return nil, fmt.Errorf("node search failed: %w", err)
	}

	edges, edgeScores, err := d.SearchEdges(ctx, query, embedding, config)
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

// cosineSimilarity calculates the cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dotProduct, normA, normB float64

	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
