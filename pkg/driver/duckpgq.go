//go:build system_duckpgq

package driver

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	_ "github.com/marcboeker/go-duckdb"
	"github.com/soundprediction/predicato/pkg/types"
)

// DuckPGQDriver implements GraphDriver using DuckDB with the DuckPGQ extension.
type DuckPGQDriver struct {
	db           *sql.DB
	embeddingDim int
}

// DuckPGQDriverSession is a minimal session implementation.
type DuckPGQDriverSession struct {
	driver *DuckPGQDriver
}

func (s *DuckPGQDriverSession) Enter(ctx context.Context) (GraphDriverSession, error) { return s, nil }
func (s *DuckPGQDriverSession) Exit(ctx context.Context, excType, excVal, excTb interface{}) error {
	return nil
}
func (s *DuckPGQDriverSession) Close() error { return nil }
func (s *DuckPGQDriverSession) Run(ctx context.Context, query interface{}, kwargs map[string]interface{}) error {
	q, ok := query.(string)
	if !ok {
		return fmt.Errorf("query must be a string")
	}
	_, _, _, err := s.driver.ExecuteQuery(ctx, q, kwargs)
	return err
}
func (s *DuckPGQDriverSession) ExecuteWrite(ctx context.Context, fn func(context.Context, GraphDriverSession, ...interface{}) (interface{}, error), args ...interface{}) (interface{}, error) {
	return fn(ctx, s, args...)
}
func (s *DuckPGQDriverSession) Provider() GraphProvider { return GraphProviderDuckPGQ }

// NewDuckPGQDriver creates a new DuckDB+DuckPGQ-backed GraphDriver.
func NewDuckPGQDriver(uri string, embeddingDim int) (*DuckPGQDriver, error) {
	dsn := uri
	if dsn == ":memory:" {
		dsn = ""
	}

	db, err := sql.Open("duckdb", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open DuckDB: %w", err)
	}

	// Load extensions: duckpgq (graph queries), vss (vector similarity), fts (full-text search)
	for _, ext := range []struct{ name, source string }{
		{"duckpgq", "community"},
		{"vss", ""},
		{"fts", ""},
	} {
		if ext.source != "" {
			_, _ = db.Exec(fmt.Sprintf("INSTALL %s FROM %s", ext.name, ext.source))
		} else {
			_, _ = db.Exec(fmt.Sprintf("INSTALL %s", ext.name))
		}
		_, _ = db.Exec(fmt.Sprintf("LOAD %s", ext.name))
	}

	d := &DuckPGQDriver{db: db, embeddingDim: embeddingDim}
	if err := d.initSchema(); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to init schema: %w", err)
	}

	return d, nil
}

func (d *DuckPGQDriver) initSchema() error {
	floatN := fmt.Sprintf("FLOAT[%d]", d.embeddingDim)
	stmts := []string{
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS entities (
			uuid VARCHAR NOT NULL,
			group_id VARCHAR NOT NULL,
			type VARCHAR NOT NULL DEFAULT 'entity',
			name VARCHAR NOT NULL DEFAULT '',
			content VARCHAR DEFAULT '',
			summary VARCHAR DEFAULT '',
			entity_type VARCHAR DEFAULT '',
			episode_type VARCHAR DEFAULT '',
			embedding %s,
			name_embedding %s,
			metadata VARCHAR DEFAULT '{}',
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			valid_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			valid_to TIMESTAMP,
			source_ids VARCHAR DEFAULT '[]',
			entity_edges VARCHAR DEFAULT '[]',
			level INTEGER DEFAULT 0,
			PRIMARY KEY (uuid, group_id)
		)`, floatN, floatN),
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS edges (
			uuid VARCHAR NOT NULL,
			group_id VARCHAR NOT NULL,
			source_id VARCHAR NOT NULL DEFAULT '',
			target_id VARCHAR NOT NULL DEFAULT '',
			type VARCHAR NOT NULL DEFAULT 'entity',
			name VARCHAR NOT NULL DEFAULT '',
			fact VARCHAR DEFAULT '',
			fact_embedding %s,
			embedding %s,
			episodes VARCHAR DEFAULT '[]',
			attributes VARCHAR DEFAULT '{}',
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			valid_from TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			valid_to TIMESTAMP,
			expired_at TIMESTAMP,
			valid_at TIMESTAMP,
			invalid_at TIMESTAMP,
			source_ids VARCHAR DEFAULT '[]',
			strength DOUBLE DEFAULT 0.0,
			PRIMARY KEY (uuid, group_id)
		)`, floatN, floatN),
	}
	for _, s := range stmts {
		if _, err := d.db.Exec(s); err != nil {
			return err
		}
	}

	// Try to create property graph (may fail if duckpgq not loaded)
	pgStmt := `CREATE OR REPLACE PROPERTY GRAPH knowledge_graph
		VERTEX TABLES (entities LABEL entity)
		EDGE TABLES (
			edges SOURCE KEY (source_id, group_id) REFERENCES entities (uuid, group_id)
			      DESTINATION KEY (target_id, group_id) REFERENCES entities (uuid, group_id)
			      LABEL relates_to
		)`
	_, _ = d.db.Exec(pgStmt)

	return nil
}

func (d *DuckPGQDriver) Close() error {
	return d.db.Close()
}

func (d *DuckPGQDriver) Provider() GraphProvider    { return GraphProviderDuckPGQ }
func (d *DuckPGQDriver) GetAossClient() interface{} { return nil }

func (d *DuckPGQDriver) Session(database *string) GraphDriverSession {
	return &DuckPGQDriverSession{driver: d}
}

func (d *DuckPGQDriver) DeleteAllIndexes(database string) {
	// DuckDB indices are managed by the engine
}

func (d *DuckPGQDriver) ExecuteQuery(ctx context.Context, query string, kwargs map[string]interface{}) (interface{}, interface{}, interface{}, error) {
	if strings.Contains(strings.ToUpper(query), "MATCH") && strings.Contains(query, "->") && strings.Contains(query, "RETURN") {
		return nil, nil, nil, fmt.Errorf("raw Cypher queries are not supported by DuckPGQ driver; use typed methods or SQL instead")
	}
	rows, err := d.db.QueryContext(ctx, query)
	if err != nil {
		return nil, nil, nil, err
	}
	defer rows.Close()

	cols, _ := rows.Columns()
	var results []map[string]interface{}
	for rows.Next() {
		vals := make([]interface{}, len(cols))
		ptrs := make([]interface{}, len(cols))
		for i := range vals {
			ptrs[i] = &vals[i]
		}
		if err := rows.Scan(ptrs...); err != nil {
			return nil, nil, nil, err
		}
		m := make(map[string]interface{}, len(cols))
		for i, c := range cols {
			m[c] = vals[i]
		}
		results = append(results, m)
	}
	return results, nil, cols, nil
}

// --- Node Operations ---

func (d *DuckPGQDriver) UpsertNode(ctx context.Context, node *types.Node) error {
	if node == nil {
		return fmt.Errorf("node cannot be nil")
	}
	now := time.Now()
	if node.CreatedAt.IsZero() {
		node.CreatedAt = now
	}
	if node.UpdatedAt.IsZero() {
		node.UpdatedAt = now
	}

	metaJSON, _ := json.Marshal(node.Metadata)
	sourceJSON, _ := json.Marshal(node.SourceIDs)
	edgesJSON, _ := json.Marshal(node.EntityEdges)

	var validTo *time.Time
	if node.ValidTo != nil {
		validTo = node.ValidTo
	}

	// DuckDB doesn't support UPDATE on ARRAY columns, so upsert via DELETE+INSERT.
	d.db.ExecContext(ctx, `DELETE FROM entities WHERE uuid = ? AND group_id = ?`, node.Uuid, node.GroupID)

	floatCast := fmt.Sprintf("?::FLOAT[%d]", d.embeddingDim)
	query := fmt.Sprintf(`INSERT INTO entities (uuid, group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level) VALUES (?, ?, ?, ?, ?, ?, ?, ?, %s, %s, ?, ?, ?, ?, ?, ?, ?, ?)`, floatCast, floatCast)
	_, err := d.db.ExecContext(ctx, query,
		node.Uuid, node.GroupID, string(node.Type), node.Name, node.Content, node.Summary,
		node.EntityType, string(node.EpisodeType),
		float32SliceToString(node.Embedding), float32SliceToString(node.NameEmbedding),
		string(metaJSON),
		node.CreatedAt, node.UpdatedAt, node.ValidFrom, validTo,
		string(sourceJSON), string(edgesJSON), node.Level,
	)
	return err
}

func (d *DuckPGQDriver) UpsertNodes(ctx context.Context, nodes []*types.Node) error {
	for _, node := range nodes {
		if err := d.UpsertNode(ctx, node); err != nil {
			return err
		}
	}
	return nil
}

// BulkLoadFromPostgres uses DuckDB's postgres_scanner extension to bulk-load
// extracted facts directly from a PostgreSQL fact store into the graph tables.
// This is orders of magnitude faster than row-by-row insertion for large datasets.
// Returns (nodesLoaded, edgesLoaded, rulesLoaded, error).
func (d *DuckPGQDriver) BulkLoadFromPostgres(ctx context.Context, pgConnStr, sourceID, groupID string) (int64, int64, int64, error) {
	// Install and load postgres_scanner
	for _, stmt := range []string{
		"INSTALL postgres_scanner",
		"LOAD postgres_scanner",
	} {
		if _, err := d.db.ExecContext(ctx, stmt); err != nil {
			return 0, 0, 0, fmt.Errorf("failed to load postgres_scanner: %w", err)
		}
	}

	// Attach PostgreSQL database
	attachStmt := fmt.Sprintf("ATTACH '%s' AS pg (TYPE POSTGRES, READ_ONLY)", pgConnStr)
	if _, err := d.db.ExecContext(ctx, attachStmt); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to attach postgres: %w", err)
	}
	defer d.db.ExecContext(ctx, "DETACH pg")

	// Phase 1: Load nodes from extracted_nodes
	nodeQuery := fmt.Sprintf(`INSERT INTO entities (uuid, group_id, type, name, content, summary, entity_type, embedding, created_at, updated_at, valid_from, source_ids)
		SELECT
			n.id,
			'%s',
			COALESCE(NULLIF(n.type, ''), 'entity'),
			n.name,
			'',
			COALESCE(n.description, ''),
			COALESCE(n.type, ''),
			n.embedding::FLOAT[%d],
			n.created_at,
			n.created_at,
			n.created_at,
			'["%s"]'
		FROM pg.extracted_nodes n
		WHERE n.source_id = '%s'`,
		groupID, d.embeddingDim, sourceID, sourceID)

	res, err := d.db.ExecContext(ctx, nodeQuery)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("failed to bulk load nodes: %w", err)
	}
	nodesLoaded, _ := res.RowsAffected()

	// Phase 2: Load edges from extracted_triples (joining to get node UUIDs)
	edgeQuery := fmt.Sprintf(`INSERT INTO edges (uuid, group_id, source_id, target_id, type, name, fact, embedding, created_at, updated_at, valid_from, source_ids, strength)
		SELECT
			t.id,
			'%s',
			sn.id,
			tn.id,
			'entity',
			t.predicate,
			COALESCE(t.description, ''),
			t.embedding::FLOAT[%d],
			t.created_at,
			t.created_at,
			t.created_at,
			'["%s"]',
			COALESCE(t.confidence, 0.0)
		FROM pg.extracted_triples t
		JOIN pg.extracted_nodes sn ON sn.name = t.subject AND sn.source_id = '%s'
		JOIN pg.extracted_nodes tn ON tn.name = t.object AND tn.source_id = '%s'
		WHERE t.source_id = '%s'`,
		groupID, d.embeddingDim, sourceID, sourceID, sourceID, sourceID)

	res, err = d.db.ExecContext(ctx, edgeQuery)
	if err != nil {
		return nodesLoaded, 0, 0, fmt.Errorf("failed to bulk load edges: %w", err)
	}
	edgesLoaded, _ := res.RowsAffected()

	// Phase 3: Load rules as entity nodes
	ruleQuery := fmt.Sprintf(`INSERT INTO entities (uuid, group_id, type, name, content, summary, entity_type, embedding, created_at, updated_at, valid_from, source_ids, metadata)
		SELECT
			r.id,
			'%s',
			'rule',
			CASE WHEN r.rule_type != '' THEN r.rule_type || ': ' ELSE '' END || r.antecedent || ' → ' || r.consequent,
			'IF ' || r.antecedent || ' THEN ' || r.consequent,
			COALESCE(r.description, ''),
			COALESCE(r.rule_type, ''),
			r.embedding::FLOAT[%d],
			r.created_at,
			r.created_at,
			r.created_at,
			'["%s"]',
			json_object('scope', COALESCE(r.scope, ''), 'source_attribution', COALESCE(r.source_attribution, ''), 'confidence', COALESCE(r.confidence, 0.0))
		FROM pg.extracted_rules r
		WHERE r.source_id = '%s'`,
		groupID, d.embeddingDim, sourceID, sourceID)

	res, err = d.db.ExecContext(ctx, ruleQuery)
	if err != nil {
		return nodesLoaded, edgesLoaded, 0, fmt.Errorf("failed to bulk load rules: %w", err)
	}
	rulesLoaded, _ := res.RowsAffected()

	// Recreate property graph to pick up new data
	pgStmt := `CREATE OR REPLACE PROPERTY GRAPH knowledge_graph
		VERTEX TABLES (entities LABEL entity)
		EDGE TABLES (
			edges SOURCE KEY (source_id, group_id) REFERENCES entities (uuid, group_id)
			      DESTINATION KEY (target_id, group_id) REFERENCES entities (uuid, group_id)
			      LABEL relates_to
		)`
	_, _ = d.db.ExecContext(ctx, pgStmt)

	return nodesLoaded, edgesLoaded, rulesLoaded, nil
}

// BulkLoadFromParquet uses DuckDB's native read_parquet() to bulk-load
// predicato fact store parquet files directly into the graph tables.
// This is the fastest path for loading parquet files produced by the
// wikidata pipeline (duckdb_to_predicato.py / wikidata_to_predicato.py).
//
// inputDir should contain: nodes.parquet, extracted_triples.parquet.gz,
// and optionally extracted_rules.parquet.gz.
//
// Returns (nodesLoaded, edgesLoaded, rulesLoaded, error).
func (d *DuckPGQDriver) BulkLoadFromParquet(ctx context.Context, inputDir, groupID string) (int64, int64, int64, error) {
	// Resolve parquet file paths
	nodesPath := resolveParquetPath(inputDir, "nodes.parquet")
	if nodesPath == "" {
		return 0, 0, 0, fmt.Errorf("nodes.parquet not found in %s", inputDir)
	}
	triplesPath := resolveParquetPath(inputDir, "extracted_triples.parquet.gz")
	if triplesPath == "" {
		triplesPath = resolveParquetPath(inputDir, "extracted_triples.parquet")
	}
	if triplesPath == "" {
		return 0, 0, 0, fmt.Errorf("extracted_triples parquet not found in %s", inputDir)
	}

	// Phase 1: Load nodes → entities
	// Use CASE to skip embeddings with wrong length (e.g. 0 for nodes without embeddings)
	nodeQuery := fmt.Sprintf(`INSERT INTO entities (
			uuid, group_id, type, name, content, entity_type,
			embedding, name_embedding, source_ids,
			created_at, updated_at, valid_from
		)
		SELECT
			id,
			COALESCE(group_id, '%s'),
			'entity',
			COALESCE(name, ''),
			COALESCE(description, ''),
			COALESCE(type, ''),
			CASE WHEN embedding IS NOT NULL AND array_length(embedding) = %d THEN embedding ELSE NULL END,
			CASE WHEN embedding IS NOT NULL AND array_length(embedding) = %d THEN embedding ELSE NULL END,
			json_array(COALESCE(source_id, 'wikidata')),
			COALESCE(created_at, CURRENT_TIMESTAMP),
			CURRENT_TIMESTAMP,
			COALESCE(created_at, CURRENT_TIMESTAMP)
		FROM read_parquet('%s')`, groupID, d.embeddingDim, d.embeddingDim, nodesPath)

	res, err := d.db.ExecContext(ctx, nodeQuery)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("failed to bulk load nodes from parquet: %w", err)
	}
	nodesLoaded, _ := res.RowsAffected()

	// Phase 2: Load triples → edges (with entity name resolution via JOIN)
	edgeQuery := fmt.Sprintf(`INSERT INTO edges (
			uuid, group_id, source_id, target_id, type, name, fact,
			fact_embedding, embedding, attributes, source_ids,
			strength, created_at, updated_at, valid_from
		)
		SELECT
			t.id,
			COALESCE(t.group_id, '%s'),
			subj.uuid,
			obj.uuid,
			'entity',
			COALESCE(t.predicate, ''),
			COALESCE(t.description, ''),
			CASE WHEN t.embedding IS NOT NULL AND array_length(t.embedding) = %d THEN t.embedding ELSE NULL END,
			CASE WHEN t.embedding IS NOT NULL AND array_length(t.embedding) = %d THEN t.embedding ELSE NULL END,
			(SELECT json_group_object(k, v) FROM (
				VALUES
					('condition', t.condition),
					('temporal', t.temporal),
					('location', t.location),
					('certainty', t.certainty),
					('scope', t.scope),
					('source_attribution', t.source_attribution)
			) AS ctx(k, v) WHERE v IS NOT NULL AND v != ''),
			json_array(COALESCE(t.source_id, 'wikidata')),
			COALESCE(t.confidence, 0.0),
			COALESCE(t.created_at, CURRENT_TIMESTAMP),
			CURRENT_TIMESTAMP,
			COALESCE(t.created_at, CURRENT_TIMESTAMP)
		FROM read_parquet('%s') t
		JOIN entities subj
			ON LOWER(TRIM(t.subject)) = LOWER(TRIM(subj.name))
			AND subj.group_id = COALESCE(t.group_id, '%s')
		JOIN entities obj
			ON LOWER(TRIM(t.object)) = LOWER(TRIM(obj.name))
			AND obj.group_id = COALESCE(t.group_id, '%s')`,
		groupID, d.embeddingDim, d.embeddingDim, triplesPath, groupID, groupID)

	res, err = d.db.ExecContext(ctx, edgeQuery)
	if err != nil {
		return nodesLoaded, 0, 0, fmt.Errorf("failed to bulk load edges from parquet: %w", err)
	}
	edgesLoaded, _ := res.RowsAffected()

	// Phase 3: Load rules (optional)
	var rulesLoaded int64
	rulesPath := resolveParquetPath(inputDir, "extracted_rules.parquet.gz")
	if rulesPath == "" {
		rulesPath = resolveParquetPath(inputDir, "extracted_rules.parquet")
	}
	if rulesPath != "" {
		ruleQuery := fmt.Sprintf(`INSERT INTO edges (
				uuid, group_id, source_id, target_id, type, name, fact,
				fact_embedding, embedding, attributes, source_ids,
				strength, created_at, updated_at, valid_from
			)
			SELECT
				id,
				'%s',
				'',
				'',
				COALESCE(rule_type, 'rule'),
				'RULE',
				CASE
					WHEN COALESCE(exception, '') != ''
					THEN 'IF ' || antecedent || ' THEN ' || consequent || ' UNLESS ' || exception
					ELSE 'IF ' || antecedent || ' THEN ' || consequent
				END,
				CASE WHEN embedding IS NOT NULL AND array_length(embedding) = %d THEN embedding ELSE NULL END,
				CASE WHEN embedding IS NOT NULL AND array_length(embedding) = %d THEN embedding ELSE NULL END,
				json_object(
					'antecedent', antecedent,
					'consequent', consequent,
					'exception', COALESCE(exception, ''),
					'rule_type', COALESCE(rule_type, ''),
					'scope', COALESCE(scope, ''),
					'source_attribution', COALESCE(source_attribution, '')
				),
				json_array(COALESCE(source_id, 'wikidata')),
				COALESCE(confidence, 0.0),
				COALESCE(created_at, CURRENT_TIMESTAMP),
				CURRENT_TIMESTAMP,
				COALESCE(created_at, CURRENT_TIMESTAMP)
			FROM read_parquet('%s')`, groupID, d.embeddingDim, d.embeddingDim, rulesPath)

		res, err = d.db.ExecContext(ctx, ruleQuery)
		if err != nil {
			return nodesLoaded, edgesLoaded, 0, fmt.Errorf("failed to bulk load rules from parquet: %w", err)
		}
		rulesLoaded, _ = res.RowsAffected()
	}

	// Recreate property graph to pick up new data
	pgStmt := `CREATE OR REPLACE PROPERTY GRAPH knowledge_graph
		VERTEX TABLES (entities LABEL entity)
		EDGE TABLES (
			edges SOURCE KEY (source_id, group_id) REFERENCES entities (uuid, group_id)
			      DESTINATION KEY (target_id, group_id) REFERENCES entities (uuid, group_id)
			      LABEL relates_to
		)`
	_, _ = d.db.ExecContext(ctx, pgStmt)

	return nodesLoaded, edgesLoaded, rulesLoaded, nil
}

// resolveParquetPath finds a parquet file or partitioned directory in inputDir.
// Returns the path suitable for DuckDB's read_parquet(), or empty string if not found.
func resolveParquetPath(inputDir, name string) string {
	// Try exact file
	path := inputDir + "/" + name
	if fileExists(path) {
		return path
	}

	// Try with .gz suffix
	gzPath := path + ".gz"
	if fileExists(gzPath) {
		return gzPath
	}

	// Try as partitioned directory (glob pattern for DuckDB)
	if dirExists(path) {
		return path + "/*.parquet"
	}

	// Try stem as directory
	stem := strings.TrimSuffix(strings.TrimSuffix(name, ".gz"), ".parquet")
	dirPath := inputDir + "/" + stem
	if dirExists(dirPath) {
		return dirPath + "/*.parquet"
	}

	return ""
}

func fileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

func dirExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.IsDir()
}

func (d *DuckPGQDriver) GetNode(ctx context.Context, nodeID, groupID string) (*types.Node, error) {
	query := `SELECT uuid, group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level FROM entities WHERE uuid = ? AND group_id = ?`
	row := d.db.QueryRowContext(ctx, query, nodeID, groupID)

	var (
		uuid, gid, ntype, name, content, summary, entityType, episodeType string
		rawEmb, rawNameEmb                                                interface{}
		metaJSON, sourceJSON, edgesJSON                                   sql.NullString
		createdAt, updatedAt, validFrom                                   time.Time
		validTo                                                           sql.NullTime
		level                                                             int
	)

	err := row.Scan(&uuid, &gid, &ntype, &name, &content, &summary, &entityType, &episodeType,
		&rawEmb, &rawNameEmb, &metaJSON, &createdAt, &updatedAt, &validFrom, &validTo,
		&sourceJSON, &edgesJSON, &level)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("node %s not found: %w", nodeID, err)
	}

	node := &types.Node{
		Uuid:        uuid,
		GroupID:     gid,
		Type:        types.NodeType(ntype),
		Name:        name,
		Content:     content,
		Summary:     summary,
		EntityType:  entityType,
		EpisodeType: types.EpisodeType(episodeType),
		CreatedAt:   createdAt,
		UpdatedAt:   updatedAt,
		ValidFrom:   validFrom,
		Level:       level,
	}

	if validTo.Valid {
		node.ValidTo = &validTo.Time
	}
	node.Embedding = scanFloat32Array(rawEmb)
	node.NameEmbedding = scanFloat32Array(rawNameEmb)

	if metaJSON.Valid {
		var meta map[string]interface{}
		if err := json.Unmarshal([]byte(metaJSON.String), &meta); err == nil {
			node.Metadata = meta
		}
	}
	if sourceJSON.Valid {
		var sids []string
		if err := json.Unmarshal([]byte(sourceJSON.String), &sids); err == nil {
			node.SourceIDs = sids
		}
	}
	if edgesJSON.Valid {
		var ee []string
		if err := json.Unmarshal([]byte(edgesJSON.String), &ee); err == nil {
			node.EntityEdges = ee
		}
	}

	return node, nil
}

func (d *DuckPGQDriver) GetNodes(ctx context.Context, nodeIDs []string, groupID string) ([]*types.Node, error) {
	nodes := make([]*types.Node, 0, len(nodeIDs))
	for _, id := range nodeIDs {
		node, err := d.GetNode(ctx, id, groupID)
		if err != nil {
			continue
		}
		nodes = append(nodes, node)
	}
	return nodes, nil
}

func (d *DuckPGQDriver) DeleteNode(ctx context.Context, nodeID, groupID string) error {
	_, err := d.db.ExecContext(ctx, `DELETE FROM entities WHERE uuid = ? AND group_id = ?`, nodeID, groupID)
	return err
}

func (d *DuckPGQDriver) GetEntityNodesByGroup(ctx context.Context, groupID string) ([]*types.Node, error) {
	return d.queryNodes(ctx, `SELECT uuid, group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level FROM entities WHERE group_id = ? AND type = 'entity'`, groupID)
}

func (d *DuckPGQDriver) ParseNodesFromRecords(records any) ([]*types.Node, error) {
	rows, ok := records.([]map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("unsupported record type")
	}
	nodes := make([]*types.Node, 0, len(rows))
	for _, row := range rows {
		node := &types.Node{
			Uuid:    fmt.Sprintf("%v", row["uuid"]),
			GroupID: fmt.Sprintf("%v", row["group_id"]),
			Name:    fmt.Sprintf("%v", row["name"]),
		}
		nodes = append(nodes, node)
	}
	return nodes, nil
}

// --- Edge Operations ---

func (d *DuckPGQDriver) UpsertEdge(ctx context.Context, edge *types.Edge) error {
	if edge == nil {
		return fmt.Errorf("edge cannot be nil")
	}
	now := time.Now()
	if edge.CreatedAt.IsZero() {
		edge.CreatedAt = now
	}

	sourceID := edge.SourceID
	if sourceID == "" {
		sourceID = edge.SourceNodeID
	}
	targetID := edge.TargetID
	if targetID == "" {
		targetID = edge.TargetNodeID
	}

	attrJSON, _ := json.Marshal(edge.Attributes)
	episodesJSON, _ := json.Marshal(edge.Episodes)
	sourceIDsJSON, _ := json.Marshal(edge.SourceIDs)

	// DuckDB doesn't support UPDATE on ARRAY columns, so upsert via DELETE+INSERT.
	d.db.ExecContext(ctx, `DELETE FROM edges WHERE uuid = ? AND group_id = ?`, edge.Uuid, edge.GroupID)

	floatCast := fmt.Sprintf("?::FLOAT[%d]", d.embeddingDim)
	query := fmt.Sprintf(`INSERT INTO edges (uuid, group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength) VALUES (?, ?, ?, ?, ?, ?, ?, %s, %s, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`, floatCast, floatCast)
	_, err := d.db.ExecContext(ctx, query,
		edge.Uuid, edge.GroupID, sourceID, targetID,
		string(edge.Type), edge.Name, edge.Fact,
		float32SliceToString(edge.FactEmbedding), float32SliceToString(edge.Embedding),
		string(episodesJSON), string(attrJSON),
		edge.CreatedAt, edge.UpdatedAt, edge.ValidFrom,
		nilTime(edge.ValidTo), nilTime(edge.ExpiredAt), nilTime(edge.ValidAt), nilTime(edge.InvalidAt),
		string(sourceIDsJSON), edge.Strength,
	)
	return err
}

func (d *DuckPGQDriver) UpsertEdges(ctx context.Context, edges []*types.Edge) error {
	for _, edge := range edges {
		if err := d.UpsertEdge(ctx, edge); err != nil {
			return err
		}
	}
	return nil
}

func (d *DuckPGQDriver) GetEdge(ctx context.Context, edgeID, groupID string) (*types.Edge, error) {
	return d.querySingleEdge(ctx, `SELECT uuid, group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength FROM edges WHERE uuid = ? AND group_id = ?`, edgeID, groupID)
}

func (d *DuckPGQDriver) GetEdges(ctx context.Context, edgeIDs []string, groupID string) ([]*types.Edge, error) {
	edges := make([]*types.Edge, 0, len(edgeIDs))
	for _, id := range edgeIDs {
		edge, err := d.GetEdge(ctx, id, groupID)
		if err != nil {
			continue
		}
		edges = append(edges, edge)
	}
	return edges, nil
}

func (d *DuckPGQDriver) DeleteEdge(ctx context.Context, edgeID, groupID string) error {
	_, err := d.db.ExecContext(ctx, `DELETE FROM edges WHERE uuid = ? AND group_id = ?`, edgeID, groupID)
	return err
}

func (d *DuckPGQDriver) UpsertEpisodicEdge(ctx context.Context, episodeUUID, entityUUID, groupID string) error {
	edge := &types.Edge{
		Name: "MENTIONS",
		Type: types.EpisodicEdgeType,
	}
	edge.Uuid = fmt.Sprintf("%s-%s", episodeUUID, entityUUID)
	edge.GroupID = groupID
	edge.SourceID = episodeUUID
	edge.TargetID = entityUUID
	edge.SourceNodeID = episodeUUID
	edge.TargetNodeID = entityUUID
	edge.CreatedAt = time.Now()
	return d.UpsertEdge(ctx, edge)
}

func (d *DuckPGQDriver) UpsertCommunityEdge(ctx context.Context, communityUUID, nodeUUID, uuid, groupID string) error {
	edge := &types.Edge{
		Name: "HAS_MEMBER",
		Type: types.CommunityEdgeType,
	}
	edge.Uuid = uuid
	edge.GroupID = groupID
	edge.SourceID = communityUUID
	edge.TargetID = nodeUUID
	edge.SourceNodeID = communityUUID
	edge.TargetNodeID = nodeUUID
	edge.CreatedAt = time.Now()
	return d.UpsertEdge(ctx, edge)
}

func (d *DuckPGQDriver) GetBetweenNodes(ctx context.Context, sourceNodeID, targetNodeID string) ([]*types.Edge, error) {
	return d.queryEdges(ctx, `SELECT uuid, group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength FROM edges WHERE source_id = ? AND target_id = ?`, sourceNodeID, targetNodeID)
}

// --- Search Operations ---

func (d *DuckPGQDriver) SearchNodes(ctx context.Context, query, groupID string, options *SearchOptions) ([]*types.Node, error) {
	limit := 10
	var nodeTypes []string

	if options != nil {
		if options.Limit > 0 {
			limit = options.Limit
		}
		if len(options.NodeTypes) > 0 {
			for _, nt := range options.NodeTypes {
				nodeTypes = append(nodeTypes, string(nt))
			}
		}
	}

	whereClause := "group_id = ? AND (name ILIKE ? OR summary ILIKE ?)"
	args := []interface{}{groupID, "%" + query + "%", "%" + query + "%"}

	if len(nodeTypes) > 0 {
		placeholders := make([]string, len(nodeTypes))
		for i, nt := range nodeTypes {
			placeholders[i] = "?"
			args = append(args, nt)
		}
		whereClause += fmt.Sprintf(" AND type IN (%s)", strings.Join(placeholders, ","))
	}

	args = append(args, limit)
	sqlQuery := fmt.Sprintf(`SELECT uuid, group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level FROM entities WHERE %s LIMIT ?`, whereClause)

	return d.queryNodes(ctx, sqlQuery, args...)
}

func (d *DuckPGQDriver) SearchEdges(ctx context.Context, query, groupID string, options *SearchOptions) ([]*types.Edge, error) {
	limit := 10
	var edgeTypes []string

	if options != nil {
		if options.Limit > 0 {
			limit = options.Limit
		}
		if len(options.EdgeTypes) > 0 {
			for _, et := range options.EdgeTypes {
				edgeTypes = append(edgeTypes, string(et))
			}
		}
	}

	whereClause := "group_id = ? AND (name ILIKE ? OR fact ILIKE ?)"
	args := []interface{}{groupID, "%" + query + "%", "%" + query + "%"}

	if len(edgeTypes) > 0 {
		placeholders := make([]string, len(edgeTypes))
		for i, et := range edgeTypes {
			placeholders[i] = "?"
			args = append(args, et)
		}
		whereClause += fmt.Sprintf(" AND type IN (%s)", strings.Join(placeholders, ","))
	}

	args = append(args, limit)
	sqlQuery := fmt.Sprintf(`SELECT uuid, group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength FROM edges WHERE %s LIMIT ?`, whereClause)

	return d.queryEdges(ctx, sqlQuery, args...)
}

func (d *DuckPGQDriver) SearchNodesByEmbedding(ctx context.Context, embedding []float32, groupID string, limit int) ([]*types.Node, error) {
	return d.searchNodesByVec(ctx, embedding, groupID, limit)
}

func (d *DuckPGQDriver) SearchEdgesByEmbedding(ctx context.Context, embedding []float32, groupID string, limit int) ([]*types.Edge, error) {
	return d.searchEdgesByVec(ctx, embedding, groupID, limit)
}

func (d *DuckPGQDriver) SearchNodesByVector(ctx context.Context, vector []float32, groupID string, options *VectorSearchOptions) ([]*types.Node, error) {
	limit := 10
	if options != nil && options.Limit > 0 {
		limit = options.Limit
	}
	return d.searchNodesByVec(ctx, vector, groupID, limit)
}

func (d *DuckPGQDriver) SearchEdgesByVector(ctx context.Context, vector []float32, groupID string, options *VectorSearchOptions) ([]*types.Edge, error) {
	limit := 10
	if options != nil && options.Limit > 0 {
		limit = options.Limit
	}
	return d.searchEdgesByVec(ctx, vector, groupID, limit)
}

func (d *DuckPGQDriver) searchNodesByVec(ctx context.Context, vector []float32, groupID string, limit int) ([]*types.Node, error) {
	// SQL-based cosine similarity using native FLOAT[N] columns and vss extension.
	floatCast := fmt.Sprintf("?::FLOAT[%d]", d.embeddingDim)
	sqlQuery := fmt.Sprintf(`SELECT uuid, group_id, type, name, content, summary, entity_type, episode_type,
		embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to,
		source_ids, entity_edges, level
		FROM entities
		WHERE group_id = ? AND embedding IS NOT NULL
		ORDER BY array_cosine_similarity(embedding, %s) DESC
		LIMIT ?`, floatCast)
	nodes, err := d.queryNodes(ctx, sqlQuery, groupID, float32SliceToString(vector), limit)
	if err == nil && len(nodes) > 0 {
		return nodes, nil
	}

	// Fallback: fetch all nodes and compute cosine similarity in Go
	allNodes, err := d.queryNodes(ctx, `SELECT uuid, group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level FROM entities WHERE group_id = ?`, groupID)
	if err != nil {
		return nil, err
	}

	type scored struct {
		node  *types.Node
		score float64
	}
	var results []scored
	for _, n := range allNodes {
		emb := n.Embedding
		if len(emb) == 0 {
			emb = n.NameEmbedding
		}
		if len(emb) == 0 {
			continue
		}
		s := duckCosineSimilarity(vector, emb)
		if s > 0 {
			results = append(results, scored{n, s})
		}
	}

	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].score > results[i].score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	if limit > len(results) {
		limit = len(results)
	}
	out := make([]*types.Node, limit)
	for i := 0; i < limit; i++ {
		out[i] = results[i].node
	}
	return out, nil
}

func (d *DuckPGQDriver) searchEdgesByVec(ctx context.Context, vector []float32, groupID string, limit int) ([]*types.Edge, error) {
	// SQL-based cosine similarity using native FLOAT[N] columns and vss extension.
	floatCast := fmt.Sprintf("?::FLOAT[%d]", d.embeddingDim)
	sqlQuery := fmt.Sprintf(`SELECT uuid, group_id, source_id, target_id, type, name, fact, fact_embedding, embedding,
		episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at,
		source_ids, strength
		FROM edges
		WHERE group_id = ? AND embedding IS NOT NULL
		ORDER BY array_cosine_similarity(embedding, %s) DESC
		LIMIT ?`, floatCast)
	edges, err := d.queryEdges(ctx, sqlQuery, groupID, float32SliceToString(vector), limit)
	if err == nil && len(edges) > 0 {
		return edges, nil
	}

	// Fallback: fetch all edges and compute cosine similarity in Go
	edges, err = d.queryEdges(ctx, `SELECT uuid, group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength FROM edges WHERE group_id = ?`, groupID)
	if err != nil {
		return nil, err
	}

	type scored struct {
		edge  *types.Edge
		score float64
	}
	var results []scored
	for _, e := range edges {
		emb := e.Embedding
		if len(emb) == 0 {
			emb = e.FactEmbedding
		}
		if len(emb) == 0 {
			continue
		}
		s := duckCosineSimilarity(vector, emb)
		if s > 0 {
			results = append(results, scored{e, s})
		}
	}

	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].score > results[i].score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	if limit > len(results) {
		limit = len(results)
	}
	out := make([]*types.Edge, limit)
	for i := 0; i < limit; i++ {
		out[i] = results[i].edge
	}
	return out, nil
}

// --- Traversal Operations ---

func (d *DuckPGQDriver) GetNeighbors(ctx context.Context, nodeID, groupID string, maxDistance int) ([]*types.Node, error) {
	if maxDistance < 1 {
		maxDistance = 1
	}
	if maxDistance > 10 {
		maxDistance = 10
	}

	// BFS using recursive CTE
	visited := map[string]bool{nodeID: true}
	frontier := []string{nodeID}

	for depth := 0; depth < maxDistance && len(frontier) > 0; depth++ {
		if len(frontier) == 0 {
			break
		}
		placeholders := make([]string, len(frontier))
		args := make([]interface{}, 0, len(frontier)+1)
		for i, f := range frontier {
			placeholders[i] = "?"
			args = append(args, f)
		}
		args = append(args, groupID)
		inClause := strings.Join(placeholders, ",")

		// Outgoing
		q := fmt.Sprintf(`SELECT DISTINCT target_id FROM edges WHERE source_id IN (%s) AND group_id = ?`, inClause)
		rows, err := d.db.QueryContext(ctx, q, args...)
		if err != nil {
			break
		}
		var nextFrontier []string
		for rows.Next() {
			var tid string
			if err := rows.Scan(&tid); err == nil && !visited[tid] {
				visited[tid] = true
				nextFrontier = append(nextFrontier, tid)
			}
		}
		rows.Close()

		// Incoming
		args2 := make([]interface{}, 0, len(frontier)+1)
		for _, f := range frontier {
			args2 = append(args2, f)
		}
		args2 = append(args2, groupID)
		q2 := fmt.Sprintf(`SELECT DISTINCT source_id FROM edges WHERE target_id IN (%s) AND group_id = ?`, inClause)
		rows2, err := d.db.QueryContext(ctx, q2, args2...)
		if err != nil {
			break
		}
		for rows2.Next() {
			var sid string
			if err := rows2.Scan(&sid); err == nil && !visited[sid] {
				visited[sid] = true
				nextFrontier = append(nextFrontier, sid)
			}
		}
		rows2.Close()

		frontier = nextFrontier
	}

	var nodes []*types.Node
	for id := range visited {
		if id == nodeID {
			continue
		}
		node, err := d.GetNode(ctx, id, groupID)
		if err != nil {
			continue
		}
		nodes = append(nodes, node)
	}
	return nodes, nil
}

func (d *DuckPGQDriver) GetRelatedNodes(ctx context.Context, nodeID, groupID string, edgeTypes []types.EdgeType) ([]*types.Node, error) {
	query := `SELECT DISTINCT target_id FROM edges WHERE source_id = ? AND group_id = ?`
	args := []interface{}{nodeID, groupID}

	if len(edgeTypes) > 0 {
		placeholders := make([]string, len(edgeTypes))
		for i, et := range edgeTypes {
			placeholders[i] = "?"
			args = append(args, string(et))
		}
		query += fmt.Sprintf(` AND type IN (%s)`, strings.Join(placeholders, ","))
	}

	rows, err := d.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var nodeIDs []string
	for rows.Next() {
		var tid string
		if err := rows.Scan(&tid); err == nil {
			nodeIDs = append(nodeIDs, tid)
		}
	}
	return d.GetNodes(ctx, nodeIDs, groupID)
}

func (d *DuckPGQDriver) GetNodeNeighbors(ctx context.Context, nodeUUID, groupID string) ([]types.Neighbor, error) {
	neighborCounts := map[string]int{}

	rows, err := d.db.QueryContext(ctx, `SELECT target_id, COUNT(*) FROM edges WHERE source_id = ? AND group_id = ? GROUP BY target_id`, nodeUUID, groupID)
	if err != nil {
		return nil, err
	}
	for rows.Next() {
		var tid string
		var cnt int
		if err := rows.Scan(&tid, &cnt); err == nil {
			neighborCounts[tid] += cnt
		}
	}
	rows.Close()

	rows2, err := d.db.QueryContext(ctx, `SELECT source_id, COUNT(*) FROM edges WHERE target_id = ? AND group_id = ? GROUP BY source_id`, nodeUUID, groupID)
	if err != nil {
		return nil, err
	}
	for rows2.Next() {
		var sid string
		var cnt int
		if err := rows2.Scan(&sid, &cnt); err == nil {
			neighborCounts[sid] += cnt
		}
	}
	rows2.Close()

	neighbors := make([]types.Neighbor, 0, len(neighborCounts))
	for id, count := range neighborCounts {
		neighbors = append(neighbors, types.Neighbor{NodeUUID: id, EdgeCount: count})
	}
	return neighbors, nil
}

// --- Temporal Operations ---

func (d *DuckPGQDriver) GetNodesInTimeRange(ctx context.Context, start, end time.Time, groupID string) ([]*types.Node, error) {
	return d.queryNodes(ctx, `SELECT uuid, group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level FROM entities WHERE group_id = ? AND created_at >= ? AND created_at <= ?`, groupID, start, end)
}

func (d *DuckPGQDriver) GetEdgesInTimeRange(ctx context.Context, start, end time.Time, groupID string) ([]*types.Edge, error) {
	return d.queryEdges(ctx, `SELECT uuid, group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength FROM edges WHERE group_id = ? AND created_at >= ? AND created_at <= ?`, groupID, start, end)
}

func (d *DuckPGQDriver) RetrieveEpisodes(ctx context.Context, referenceTime time.Time, groupIDs []string, limit int, episodeType *types.EpisodeType) ([]*types.Node, error) {
	var allNodes []*types.Node
	for _, gid := range groupIDs {
		query := `SELECT uuid, group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level FROM entities WHERE group_id = ? AND type = 'episodic' AND created_at <= ? ORDER BY created_at DESC LIMIT ?`
		nodes, err := d.queryNodes(ctx, query, gid, referenceTime, limit)
		if err != nil {
			continue
		}
		for _, n := range nodes {
			if episodeType != nil && n.EpisodeType != *episodeType {
				continue
			}
			allNodes = append(allNodes, n)
		}
	}
	if limit > 0 && len(allNodes) > limit {
		allNodes = allNodes[:limit]
	}
	return allNodes, nil
}

// --- Community Operations ---

func (d *DuckPGQDriver) GetCommunities(ctx context.Context, groupID string, level int) ([]*types.Node, error) {
	return d.queryNodes(ctx, `SELECT uuid, group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level FROM entities WHERE group_id = ? AND type = 'community' AND level = ?`, groupID, level)
}

func (d *DuckPGQDriver) BuildCommunities(ctx context.Context, groupID string) error {
	return nil // Minimal implementation
}

func (d *DuckPGQDriver) GetExistingCommunity(ctx context.Context, entityUUID string) (*types.Node, error) {
	row := d.db.QueryRowContext(ctx, `SELECT e.source_id, e.group_id FROM edges e WHERE e.target_id = ? AND e.type = 'community' LIMIT 1`, entityUUID)
	var communityID, groupID string
	if err := row.Scan(&communityID, &groupID); err != nil {
		return nil, fmt.Errorf("no community found for entity %s: %w", entityUUID, err)
	}
	return d.GetNode(ctx, communityID, groupID)
}

func (d *DuckPGQDriver) FindModalCommunity(ctx context.Context, entityUUID string) (*types.Node, error) {
	return d.GetExistingCommunity(ctx, entityUUID)
}

func (d *DuckPGQDriver) RemoveCommunities(ctx context.Context) error {
	_, err := d.db.ExecContext(ctx, `DELETE FROM edges WHERE type = 'community'`)
	if err != nil {
		return err
	}
	_, err = d.db.ExecContext(ctx, `DELETE FROM entities WHERE type = 'community'`)
	return err
}

// --- Admin Operations ---

func (d *DuckPGQDriver) CreateIndices(ctx context.Context) error {
	// B-Tree indices for common lookups
	btreeIndices := []string{
		`CREATE INDEX IF NOT EXISTS idx_entities_group ON entities(group_id)`,
		`CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)`,
		`CREATE INDEX IF NOT EXISTS idx_edges_group ON edges(group_id)`,
		`CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)`,
		`CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)`,
		`CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type)`,
	}
	for _, idx := range btreeIndices {
		if _, err := d.db.ExecContext(ctx, idx); err != nil {
			continue
		}
	}

	// FTS indices for text search (requires fts extension)
	ftsIndices := []string{
		`PRAGMA create_fts_index('entities', 'uuid', 'name', 'summary', 'content', overwrite=1)`,
		`PRAGMA create_fts_index('edges', 'uuid', 'name', 'fact', overwrite=1)`,
	}
	for _, idx := range ftsIndices {
		_, _ = d.db.ExecContext(ctx, idx) // non-fatal if fts extension unavailable
	}

	// HNSW indices for vector similarity search (requires vss extension)
	// Embedding columns use native FLOAT[N] types, enabling direct HNSW indexing.
	hnswIndices := []string{
		`CREATE INDEX IF NOT EXISTS idx_entities_embedding_hnsw ON entities USING HNSW (embedding) WITH (metric = 'cosine')`,
		`CREATE INDEX IF NOT EXISTS idx_edges_embedding_hnsw ON edges USING HNSW (embedding) WITH (metric = 'cosine')`,
	}
	for _, idx := range hnswIndices {
		_, _ = d.db.ExecContext(ctx, idx) // non-fatal if vss extension unavailable
	}

	return nil
}

func (d *DuckPGQDriver) GetStats(ctx context.Context, groupID string) (*GraphStats, error) {
	stats := &GraphStats{
		LastUpdated: time.Now(),
		NodesByType: make(map[string]int64),
		EdgesByType: make(map[string]int64),
	}

	rows, err := d.db.QueryContext(ctx, `SELECT type, COUNT(*) FROM entities WHERE group_id = ? GROUP BY type`, groupID)
	if err != nil {
		return stats, nil
	}
	for rows.Next() {
		var t string
		var c int64
		if err := rows.Scan(&t, &c); err == nil {
			stats.NodesByType[t] = c
			stats.NodeCount += c
		}
	}
	rows.Close()

	rows2, err := d.db.QueryContext(ctx, `SELECT type, COUNT(*) FROM edges WHERE group_id = ? GROUP BY type`, groupID)
	if err != nil {
		return stats, nil
	}
	for rows2.Next() {
		var t string
		var c int64
		if err := rows2.Scan(&t, &c); err == nil {
			stats.EdgesByType[t] = c
			stats.EdgeCount += c
		}
	}
	rows2.Close()

	if cnt, ok := stats.NodesByType["community"]; ok {
		stats.CommunityCount = cnt
	}

	return stats, nil
}

func (d *DuckPGQDriver) GetAllGroupIDs(ctx context.Context) ([]string, error) {
	rows, err := d.db.QueryContext(ctx, `SELECT DISTINCT group_id FROM entities`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var ids []string
	for rows.Next() {
		var gid string
		if err := rows.Scan(&gid); err == nil {
			ids = append(ids, gid)
		}
	}
	return ids, nil
}

// --- GraphExporter Implementation ---

func (d *DuckPGQDriver) IterateNodes(ctx context.Context, groupID string, fn func(*types.Node) error) error {
	nodes, err := d.queryNodes(ctx, `SELECT uuid, group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level FROM entities WHERE group_id = ?`, groupID)
	if err != nil {
		return err
	}
	for _, n := range nodes {
		if err := fn(n); err != nil {
			return err
		}
	}
	return nil
}

func (d *DuckPGQDriver) IterateEdges(ctx context.Context, groupID string, fn func(*types.Edge) error) error {
	edges, err := d.queryEdges(ctx, `SELECT uuid, group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength FROM edges WHERE group_id = ?`, groupID)
	if err != nil {
		return err
	}
	for _, e := range edges {
		if err := fn(e); err != nil {
			return err
		}
	}
	return nil
}

// --- Helper functions ---

func (d *DuckPGQDriver) queryNodes(ctx context.Context, query string, args ...interface{}) ([]*types.Node, error) {
	rows, err := d.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var nodes []*types.Node
	for rows.Next() {
		var (
			uuid, gid, ntype, name, content, summary, entityType, episodeType string
			rawEmb, rawNameEmb                                                interface{}
			metaJSON, sourceJSON, edgesJSON                                   sql.NullString
			createdAt, updatedAt, validFrom                                   time.Time
			validTo                                                           sql.NullTime
			level                                                             int
		)
		if err := rows.Scan(&uuid, &gid, &ntype, &name, &content, &summary, &entityType, &episodeType,
			&rawEmb, &rawNameEmb, &metaJSON, &createdAt, &updatedAt, &validFrom, &validTo,
			&sourceJSON, &edgesJSON, &level); err != nil {
			continue
		}

		node := &types.Node{
			Uuid:        uuid,
			GroupID:     gid,
			Type:        types.NodeType(ntype),
			Name:        name,
			Content:     content,
			Summary:     summary,
			EntityType:  entityType,
			EpisodeType: types.EpisodeType(episodeType),
			CreatedAt:   createdAt,
			UpdatedAt:   updatedAt,
			ValidFrom:   validFrom,
			Level:       level,
		}
		if validTo.Valid {
			node.ValidTo = &validTo.Time
		}
		node.Embedding = scanFloat32Array(rawEmb)
		node.NameEmbedding = scanFloat32Array(rawNameEmb)
		if metaJSON.Valid {
			var meta map[string]interface{}
			if err := json.Unmarshal([]byte(metaJSON.String), &meta); err == nil {
				node.Metadata = meta
			}
		}
		if sourceJSON.Valid {
			var sids []string
			if err := json.Unmarshal([]byte(sourceJSON.String), &sids); err == nil {
				node.SourceIDs = sids
			}
		}
		if edgesJSON.Valid {
			var ee []string
			if err := json.Unmarshal([]byte(edgesJSON.String), &ee); err == nil {
				node.EntityEdges = ee
			}
		}
		nodes = append(nodes, node)
	}
	return nodes, nil
}

func (d *DuckPGQDriver) queryEdges(ctx context.Context, query string, args ...interface{}) ([]*types.Edge, error) {
	rows, err := d.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var edges []*types.Edge
	for rows.Next() {
		edge, err := d.scanEdge(rows)
		if err != nil {
			continue
		}
		edges = append(edges, edge)
	}
	return edges, nil
}

func (d *DuckPGQDriver) querySingleEdge(ctx context.Context, query string, args ...interface{}) (*types.Edge, error) {
	rows, err := d.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	if !rows.Next() {
		return nil, nil
	}
	return d.scanEdge(rows)
}

func (d *DuckPGQDriver) scanEdge(rows *sql.Rows) (*types.Edge, error) {
	var (
		uuid, gid, sourceID, targetID, etype, name, fact string
		rawFactEmb, rawEmb                               interface{}
		episodesJSON, attrJSON                           sql.NullString
		sourceIDsJSON                                    sql.NullString
		createdAt, updatedAt, validFrom                  time.Time
		validTo, expiredAt, validAt, invalidAt           sql.NullTime
		strength                                         float64
	)

	if err := rows.Scan(&uuid, &gid, &sourceID, &targetID, &etype, &name, &fact,
		&rawFactEmb, &rawEmb, &episodesJSON, &attrJSON,
		&createdAt, &updatedAt, &validFrom, &validTo, &expiredAt, &validAt, &invalidAt,
		&sourceIDsJSON, &strength); err != nil {
		return nil, err
	}

	edge := &types.Edge{
		Name:     name,
		Fact:     fact,
		Summary:  fact,
		Type:     types.EdgeType(etype),
		SourceID: sourceID,
		TargetID: targetID,
		Strength: strength,
	}
	edge.Uuid = uuid
	edge.GroupID = gid
	edge.SourceNodeID = sourceID
	edge.TargetNodeID = targetID
	edge.CreatedAt = createdAt
	edge.UpdatedAt = updatedAt
	edge.ValidFrom = validFrom

	if validTo.Valid {
		edge.ValidTo = &validTo.Time
	}
	if expiredAt.Valid {
		edge.ExpiredAt = &expiredAt.Time
	}
	if validAt.Valid {
		edge.ValidAt = &validAt.Time
	}
	if invalidAt.Valid {
		edge.InvalidAt = &invalidAt.Time
	}

	edge.FactEmbedding = scanFloat32Array(rawFactEmb)
	edge.Embedding = scanFloat32Array(rawEmb)

	if episodesJSON.Valid {
		var eps []string
		if err := json.Unmarshal([]byte(episodesJSON.String), &eps); err == nil {
			edge.Episodes = eps
		}
	}
	if attrJSON.Valid {
		var attrs map[string]interface{}
		if err := json.Unmarshal([]byte(attrJSON.String), &attrs); err == nil {
			edge.Attributes = attrs
		}
	}
	if sourceIDsJSON.Valid {
		var sids []string
		if err := json.Unmarshal([]byte(sourceIDsJSON.String), &sids); err == nil {
			edge.SourceIDs = sids
		}
	}

	return edge, nil
}

// scanFloat32Array converts a DuckDB native FLOAT[N] array (returned as []interface{})
// into a Go []float32 slice. Returns nil for NULL values.
func scanFloat32Array(raw interface{}) []float32 {
	if raw == nil {
		return nil
	}
	arr, ok := raw.([]interface{})
	if !ok {
		return nil
	}
	result := make([]float32, len(arr))
	for i, v := range arr {
		switch f := v.(type) {
		case float64:
			result[i] = float32(f)
		case float32:
			result[i] = f
		}
	}
	return result
}

// float32SliceToString converts a []float32 to a DuckDB array literal string
// suitable for casting with ?::FLOAT[N]. Returns nil for empty/nil slices (SQL NULL).
func float32SliceToString(v []float32) interface{} {
	if len(v) == 0 {
		return nil
	}
	var b strings.Builder
	b.WriteByte('[')
	for i, f := range v {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%g", f)
	}
	b.WriteByte(']')
	return b.String()
}

func nilTime(t *time.Time) interface{} {
	if t == nil {
		return nil
	}
	return *t
}

func duckCosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// Compile-time interface checks
var _ GraphDriver = (*DuckPGQDriver)(nil)
var _ GraphExporter = (*DuckPGQDriver)(nil)
