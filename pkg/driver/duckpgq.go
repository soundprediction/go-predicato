//go:build system_duckpgq

package driver

import (
	"context"
	"database/sql"
	_ "embed"
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
// predicateDescriptionsTSV is the embedded predicate-to-template mapping table.
// Each line has: PREDICATE<tab>template with {subject} and {object} placeholders.
//
//go:embed predicate_descriptions.tsv
var predicateDescriptionsTSV string

// createPredicateMapTable creates a temp table _predicate_map from the
// embedded TSV, for use in edge INSERT queries.
func createPredicateMapTable(ctx context.Context, conn interface {
	ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error)
}) error {
	if _, err := conn.ExecContext(ctx,
		"CREATE OR REPLACE TEMP TABLE _predicate_map (predicate VARCHAR, template VARCHAR)"); err != nil {
		return err
	}

	var b strings.Builder
	b.WriteString("INSERT INTO _predicate_map VALUES\n")
	first := true
	for _, line := range strings.Split(predicateDescriptionsTSV, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "\t", 2)
		if len(parts) != 2 {
			continue
		}
		pred := strings.TrimSpace(parts[0])
		tmpl := strings.TrimSpace(parts[1])
		if pred == "" || tmpl == "" {
			continue
		}
		if !first {
			b.WriteString(",\n")
		}
		escaped := strings.ReplaceAll(tmpl, "'", "''")
		fmt.Fprintf(&b, "  ('%s', '%s')", pred, escaped)
		first = false
	}
	if first {
		return nil // no entries
	}
	_, err := conn.ExecContext(ctx, b.String())
	return err
}

// factDescriptionExpr is a SQL expression that looks up the predicate in
// _predicate_map and substitutes {subject}/{object} into the template.
// Falls back to "subject <predicate> object" when no template exists.
// The caller must ensure _predicate_map has been created on the same connection.
const factDescriptionExpr = `COALESCE(
				(SELECT REPLACE(REPLACE(pm.template, '{subject}', t.subject), '{object}', t.object)
				 FROM _predicate_map pm WHERE pm.predicate = t.predicate),
				t.subject || ' ' || LOWER(REPLACE(t.predicate, '_', ' ')) || ' ' || t.object
			)`

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

	// Use a dedicated connection so the _predicate_map temp table persists.
	conn, err := d.db.Conn(ctx)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("failed to acquire connection: %w", err)
	}
	defer conn.Close()

	// Create predicate description mapping table for natural fact text.
	if err := createPredicateMapTable(ctx, conn); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to create predicate map: %w", err)
	}
	defer func() { _, _ = conn.ExecContext(ctx, "DROP TABLE IF EXISTS _predicate_map") }()

	// Count input rows for progress reporting
	var inputNodeCount, inputTripleCount int64
	row := conn.QueryRowContext(ctx, fmt.Sprintf("SELECT COUNT(*) FROM read_parquet('%s')", nodesPath))
	if err := row.Scan(&inputNodeCount); err == nil {
		fmt.Printf("  Input: %d nodes in %s\n", inputNodeCount, nodesPath)
	}
	row = conn.QueryRowContext(ctx, fmt.Sprintf("SELECT COUNT(*) FROM read_parquet('%s')", triplesPath))
	if err := row.Scan(&inputTripleCount); err == nil {
		fmt.Printf("  Input: %d triples in %s\n", inputTripleCount, triplesPath)
	}

	// Phase 1: Load nodes → entities
	fmt.Printf("  Phase 1/4: Loading nodes into entities...\n")
	phaseStart := time.Now()
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

	res, err := conn.ExecContext(ctx, nodeQuery)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("failed to bulk load nodes from parquet: %w", err)
	}
	nodesLoaded, _ := res.RowsAffected()
	fmt.Printf("  Phase 1/4: Done — %d nodes loaded in %s\n", nodesLoaded, time.Since(phaseStart).Round(time.Second))

	// Phase 2: Load triples → edges (with entity name resolution via JOIN)
	// The JOIN can produce multiple rows per triple when entities share names, so we
	// deduplicate with ROW_NUMBER to keep exactly one row per (id, group_id).
	fmt.Printf("  Phase 2/4: Loading triples into edges (with entity resolution JOIN)...\n")
	phaseStart = time.Now()
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
			`+factDescriptionExpr+`,
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
			AND obj.group_id = COALESCE(t.group_id, '%s')
		QUALIFY ROW_NUMBER() OVER (PARTITION BY t.id, COALESCE(t.group_id, '%s') ORDER BY subj.uuid) = 1`,
		groupID, d.embeddingDim, d.embeddingDim, triplesPath, groupID, groupID, groupID)

	res, err = conn.ExecContext(ctx, edgeQuery)
	if err != nil {
		return nodesLoaded, 0, 0, fmt.Errorf("failed to bulk load edges from parquet: %w", err)
	}
	edgesLoaded, _ := res.RowsAffected()
	fmt.Printf("  Phase 2/4: Done — %d edges loaded in %s\n", edgesLoaded, time.Since(phaseStart).Round(time.Second))

	// Phase 3: Load rules (optional)
	var rulesLoaded int64
	rulesPath := resolveParquetPath(inputDir, "extracted_rules.parquet.gz")
	if rulesPath == "" {
		rulesPath = resolveParquetPath(inputDir, "extracted_rules.parquet")
	}
	if rulesPath != "" {
		fmt.Printf("  Phase 3/4: Loading rules...\n")
		phaseStart = time.Now()
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

		res, err = conn.ExecContext(ctx, ruleQuery)
		if err != nil {
			return nodesLoaded, edgesLoaded, 0, fmt.Errorf("failed to bulk load rules from parquet: %w", err)
		}
		rulesLoaded, _ = res.RowsAffected()
		fmt.Printf("  Phase 3/4: Done — %d rules loaded in %s\n", rulesLoaded, time.Since(phaseStart).Round(time.Second))
	} else {
		fmt.Printf("  Phase 3/4: No rules parquet found — skipping\n")
	}

	// Recreate property graph to pick up new data
	fmt.Printf("  Phase 4/4: Creating property graph...\n")
	phaseStart = time.Now()
	pgStmt := `CREATE OR REPLACE PROPERTY GRAPH knowledge_graph
		VERTEX TABLES (entities LABEL entity)
		EDGE TABLES (
			edges SOURCE KEY (source_id, group_id) REFERENCES entities (uuid, group_id)
			      DESTINATION KEY (target_id, group_id) REFERENCES entities (uuid, group_id)
			      LABEL relates_to
		)`
	_, _ = conn.ExecContext(ctx, pgStmt)
	fmt.Printf("  Phase 4/4: Done in %s\n", time.Since(phaseStart).Round(time.Second))

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
	fmt.Printf("  Building communities for group %q...\n", groupID)
	start := time.Now()

	conn, err := d.db.Conn(ctx)
	if err != nil {
		return fmt.Errorf("failed to acquire connection: %w", err)
	}
	defer conn.Close()

	// Step 1: Load adjacency list in bulk.
	// Each edge connects two entity UUIDs; we build a bidirectional neighbor map.
	rows, err := conn.QueryContext(ctx, `
		SELECT source_id, target_id FROM edges
		WHERE group_id = ? AND type = 'entity'
		  AND source_id != '' AND target_id != ''`,
		groupID)
	if err != nil {
		return fmt.Errorf("failed to query edges for communities: %w", err)
	}
	defer rows.Close()

	neighbors := make(map[string][]string)
	for rows.Next() {
		var src, tgt string
		if err := rows.Scan(&src, &tgt); err != nil {
			return fmt.Errorf("scanning edge: %w", err)
		}
		neighbors[src] = append(neighbors[src], tgt)
		neighbors[tgt] = append(neighbors[tgt], src)
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("iterating edges: %w", err)
	}

	if len(neighbors) == 0 {
		fmt.Println("  No edges found — skipping community detection")
		return nil
	}

	// Step 2: Label propagation (same algorithm as pkg/community/label_propagation.go).
	// Each node starts with its own UUID as its label.
	label := make(map[string]string, len(neighbors))
	for node := range neighbors {
		label[node] = node
	}

	const maxIter = 100
	for iter := 0; iter < maxIter; iter++ {
		changed := false
		for node, nbs := range neighbors {
			// Count community labels among neighbors.
			counts := make(map[string]int, len(nbs))
			for _, nb := range nbs {
				counts[label[nb]]++
			}
			// Find the label with the highest count (break ties by label string).
			bestLabel := label[node]
			bestCount := 0
			for lbl, cnt := range counts {
				if cnt > bestCount || (cnt == bestCount && lbl < bestLabel) {
					bestLabel = lbl
					bestCount = cnt
				}
			}
			if bestCount > 1 && bestLabel != label[node] {
				label[node] = bestLabel
				changed = true
			}
		}
		if !changed {
			break
		}
	}

	// Step 3: Group nodes by their community label, keep only multi-node communities.
	clusters := make(map[string][]string)
	for node, lbl := range label {
		clusters[lbl] = append(clusters[lbl], node)
	}

	var multiClusters [][]string
	for _, members := range clusters {
		if len(members) > 1 {
			multiClusters = append(multiClusters, members)
		}
	}

	if len(multiClusters) == 0 {
		fmt.Println("  No multi-node communities found")
		return nil
	}

	// Step 4: For each community, find the most-connected member name to use as community name.
	// Load entity names in bulk.
	entityNames := make(map[string]string)
	nameRows, err := conn.QueryContext(ctx, `SELECT uuid, name FROM entities WHERE group_id = ? AND type = 'entity'`, groupID)
	if err != nil {
		return fmt.Errorf("failed to query entity names: %w", err)
	}
	defer nameRows.Close()
	for nameRows.Next() {
		var uuid, name string
		if err := nameRows.Scan(&uuid, &name); err != nil {
			return fmt.Errorf("scanning entity name: %w", err)
		}
		entityNames[uuid] = name
	}

	// Step 5: Insert community nodes and HAS_MEMBER edges.
	now := time.Now().UTC()
	var communityCount, edgeCount int64
	for i, members := range multiClusters {
		// Find the most-connected member for the community name.
		bestUUID := members[0]
		bestDegree := len(neighbors[bestUUID])
		for _, m := range members[1:] {
			if deg := len(neighbors[m]); deg > bestDegree {
				bestUUID = m
				bestDegree = deg
			}
		}
		communityName := entityNames[bestUUID]
		if communityName == "" {
			communityName = fmt.Sprintf("Community %d", i+1)
		}

		commUUID := fmt.Sprintf("comm_%s_%d", groupID, i)

		// Insert community node.
		_, err := conn.ExecContext(ctx, `INSERT INTO entities (
				uuid, group_id, type, name, content, entity_type,
				source_ids, created_at, updated_at, valid_from
			) VALUES (?, ?, 'community', ?, ?, 'community', '["community"]', ?, ?, ?)`,
			commUUID, groupID, communityName,
			fmt.Sprintf("%d members", len(members)),
			now, now, now)
		if err != nil {
			return fmt.Errorf("failed to insert community node: %w", err)
		}
		communityCount++

		// Insert HAS_MEMBER edges.
		for _, memberUUID := range members {
			edgeUUID := fmt.Sprintf("comm_edge_%s_%d_%s", groupID, i, memberUUID)
			_, err := conn.ExecContext(ctx, `INSERT INTO edges (
					uuid, group_id, source_id, target_id, type, name,
					fact, source_ids, strength,
					created_at, updated_at, valid_from
				) VALUES (?, ?, ?, ?, 'community', 'HAS_MEMBER', '', '["community"]', 1.0, ?, ?, ?)`,
				edgeUUID, groupID, commUUID, memberUUID, now, now, now)
			if err != nil {
				return fmt.Errorf("failed to insert community edge: %w", err)
			}
			edgeCount++
		}
	}

	fmt.Printf("  Communities built: %d communities, %d membership edges in %s\n",
		communityCount, edgeCount, time.Since(start).Round(time.Second))
	return nil
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

// TypeCount holds a value and its occurrence count.
type TypeCount struct {
	Value string
	Count int64
}

// EmbeddingCoverage holds per-collection embedding coverage fractions.
type EmbeddingCoverage struct {
	NodesFraction   float64
	TriplesFraction float64
	RulesFraction   float64
}

// ParquetStats holds frequency statistics about a set of predicato parquet files.
type ParquetStats struct {
	EmbeddingCoverage EmbeddingCoverage
	TopEntityTypes    []TypeCount
	TopPredicates     []TypeCount
	TopRuleTypes      []TypeCount
	TopScopes         []TypeCount
	NodeCount         int64
	TripleCount       int64
	RuleCount         int64
}

// ExploreParquet reads parquet files from inputDir and returns frequency statistics.
// It uses a separate in-memory DuckDB connection — no output database file is required.
func (d *DuckPGQDriver) ExploreParquet(ctx context.Context, inputDir string) (*ParquetStats, error) {
	nodesPath := resolveParquetPath(inputDir, "nodes.parquet")
	if nodesPath == "" {
		return nil, fmt.Errorf("nodes.parquet not found in %s", inputDir)
	}
	triplesPath := resolveParquetPath(inputDir, "extracted_triples.parquet.gz")
	if triplesPath == "" {
		triplesPath = resolveParquetPath(inputDir, "extracted_triples.parquet")
	}
	if triplesPath == "" {
		return nil, fmt.Errorf("extracted_triples parquet not found in %s", inputDir)
	}

	// Open a separate in-memory DuckDB — no output file needed.
	memDB, err := sql.Open("duckdb", "")
	if err != nil {
		return nil, fmt.Errorf("failed to open in-memory DuckDB: %w", err)
	}
	defer memDB.Close()

	stats := &ParquetStats{}
	dim := d.embeddingDim

	// Node count.
	row := memDB.QueryRowContext(ctx, fmt.Sprintf(`SELECT COUNT(*) FROM read_parquet('%s')`, nodesPath))
	if err := row.Scan(&stats.NodeCount); err != nil {
		return nil, fmt.Errorf("node count query failed: %w", err)
	}

	// Top entity types.
	rows, err := memDB.QueryContext(ctx, fmt.Sprintf(
		`SELECT COALESCE(type, '') AS val, COUNT(*) AS cnt
		 FROM read_parquet('%s')
		 GROUP BY type ORDER BY cnt DESC LIMIT 20`, nodesPath))
	if err != nil {
		return nil, fmt.Errorf("top entity types query failed: %w", err)
	}
	if stats.TopEntityTypes, err = scanTypeCounts(rows); err != nil {
		return nil, err
	}

	// Nodes embedding coverage.
	row = memDB.QueryRowContext(ctx, fmt.Sprintf(
		`SELECT COUNT(*) FILTER (WHERE embedding IS NOT NULL AND array_length(embedding) = %d) * 1.0
		      / NULLIF(COUNT(*), 0)
		 FROM read_parquet('%s')`, dim, nodesPath))
	if err := row.Scan(&stats.EmbeddingCoverage.NodesFraction); err != nil {
		return nil, fmt.Errorf("nodes embedding coverage query failed: %w", err)
	}

	// Triple count.
	row = memDB.QueryRowContext(ctx, fmt.Sprintf(`SELECT COUNT(*) FROM read_parquet('%s')`, triplesPath))
	if err := row.Scan(&stats.TripleCount); err != nil {
		return nil, fmt.Errorf("triple count query failed: %w", err)
	}

	// Top predicates.
	rows, err = memDB.QueryContext(ctx, fmt.Sprintf(
		`SELECT COALESCE(predicate, '') AS val, COUNT(*) AS cnt
		 FROM read_parquet('%s')
		 GROUP BY predicate ORDER BY cnt DESC LIMIT 20`, triplesPath))
	if err != nil {
		return nil, fmt.Errorf("top predicates query failed: %w", err)
	}
	if stats.TopPredicates, err = scanTypeCounts(rows); err != nil {
		return nil, err
	}

	// Triples embedding coverage.
	row = memDB.QueryRowContext(ctx, fmt.Sprintf(
		`SELECT COUNT(*) FILTER (WHERE embedding IS NOT NULL AND array_length(embedding) = %d) * 1.0
		      / NULLIF(COUNT(*), 0)
		 FROM read_parquet('%s')`, dim, triplesPath))
	if err := row.Scan(&stats.EmbeddingCoverage.TriplesFraction); err != nil {
		return nil, fmt.Errorf("triples embedding coverage query failed: %w", err)
	}

	// Rules (optional).
	rulesPath := resolveParquetPath(inputDir, "extracted_rules.parquet.gz")
	if rulesPath == "" {
		rulesPath = resolveParquetPath(inputDir, "extracted_rules.parquet")
	}
	if rulesPath != "" {
		row = memDB.QueryRowContext(ctx, fmt.Sprintf(`SELECT COUNT(*) FROM read_parquet('%s')`, rulesPath))
		if err := row.Scan(&stats.RuleCount); err != nil {
			return nil, fmt.Errorf("rule count query failed: %w", err)
		}

		rows, err = memDB.QueryContext(ctx, fmt.Sprintf(
			`SELECT COALESCE(rule_type, '') AS val, COUNT(*) AS cnt
			 FROM read_parquet('%s')
			 GROUP BY rule_type ORDER BY cnt DESC LIMIT 20`, rulesPath))
		if err != nil {
			return nil, fmt.Errorf("top rule types query failed: %w", err)
		}
		if stats.TopRuleTypes, err = scanTypeCounts(rows); err != nil {
			return nil, err
		}

		rows, err = memDB.QueryContext(ctx, fmt.Sprintf(
			`SELECT COALESCE(scope, '') AS val, COUNT(*) AS cnt
			 FROM read_parquet('%s')
			 GROUP BY scope ORDER BY cnt DESC LIMIT 20`, rulesPath))
		if err != nil {
			return nil, fmt.Errorf("top scopes query failed: %w", err)
		}
		if stats.TopScopes, err = scanTypeCounts(rows); err != nil {
			return nil, err
		}

		row = memDB.QueryRowContext(ctx, fmt.Sprintf(
			`SELECT COUNT(*) FILTER (WHERE embedding IS NOT NULL AND array_length(embedding) = %d) * 1.0
			      / NULLIF(COUNT(*), 0)
			 FROM read_parquet('%s')`, dim, rulesPath))
		if err := row.Scan(&stats.EmbeddingCoverage.RulesFraction); err != nil {
			return nil, fmt.Errorf("rules embedding coverage query failed: %w", err)
		}
	}

	return stats, nil
}

// scanTypeCounts reads (value, count) rows into a TypeCount slice.
func scanTypeCounts(rows *sql.Rows) ([]TypeCount, error) {
	defer rows.Close()
	var result []TypeCount
	for rows.Next() {
		var tc TypeCount
		if err := rows.Scan(&tc.Value, &tc.Count); err != nil {
			return nil, fmt.Errorf("scanning type count: %w", err)
		}
		result = append(result, tc)
	}
	return result, rows.Err()
}

// BulkLoadFromParquetWithFilter loads a topic-filtered subset of the parquet knowledge graph
// into the DuckPGQ database using a multi-phase approach:
//
//  1. Filter nodes by embedding similarity to topic vector (Threshold)
//  2. Filter triples by embedding similarity (TripleThreshold, lower)
//  3. Filter rules by embedding similarity (TripleThreshold)
//  4. Build core qualifying node set = filtered nodes + triple endpoints + rule-referenced nodes
//  5. Insert core nodes into entities
//  6. Collect ALL edges where EITHER subject OR object is a core node
//  7. Expand node set with neighbor endpoints from collected edges
//  8. Insert neighbor nodes
//  9. Insert all collected edges
//  10. Insert filtered rules
//  11. Recreate property graph
//
// If filter is nil, the method delegates to BulkLoadFromParquet (full load).
// Returns (nodesLoaded, edgesLoaded, rulesLoaded, error).
func (d *DuckPGQDriver) BulkLoadFromParquetWithFilter(
	ctx context.Context,
	inputDir string,
	groupID string,
	filter *ParquetTopicFilter,
) (int64, int64, int64, error) {
	if filter == nil {
		return d.BulkLoadFromParquet(ctx, inputDir, groupID)
	}
	if len(filter.Embedding) != d.embeddingDim {
		return 0, 0, 0, fmt.Errorf("embedding dimension mismatch: filter has %d, driver expects %d",
			len(filter.Embedding), d.embeddingDim)
	}

	// Determine data source: PostgreSQL or parquet files.
	usePostgres := filter.PostgresConnStr != ""
	var nodesPath, triplesPath, rulesPath string

	if usePostgres {
		// Install postgres_scanner and attach the remote database.
		for _, stmt := range []string{
			"INSTALL postgres_scanner",
			"LOAD postgres_scanner",
			fmt.Sprintf("ATTACH 'dbname=glancedb %s' AS pg (TYPE POSTGRES, READ_ONLY)", filter.PostgresConnStr),
		} {
			if _, err := d.db.ExecContext(ctx, stmt); err != nil {
				return 0, 0, 0, fmt.Errorf("postgres setup failed (%s): %w", stmt, err)
			}
		}
		// Create source views that normalize PG vector(1024) → FLOAT[] via cast.
		srcViews := map[string]string{
			"_src_nodes":   "SELECT *, embedding::FLOAT[] AS emb FROM pg.extracted_nodes",
			"_src_triples": "SELECT *, embedding::FLOAT[] AS emb FROM pg.extracted_triples",
			"_src_rules":   "SELECT *, embedding::FLOAT[] AS emb FROM pg.extracted_rules",
		}
		for name, query := range srcViews {
			if _, err := d.db.ExecContext(ctx, fmt.Sprintf("CREATE OR REPLACE VIEW %s AS %s", name, query)); err != nil {
				return 0, 0, 0, fmt.Errorf("failed to create view %s: %w", name, err)
			}
		}
		fmt.Println("  Source: PostgreSQL (via postgres_scanner)")
	} else {
		nodesPath = resolveParquetPath(inputDir, "nodes.parquet")
		if nodesPath == "" {
			return 0, 0, 0, fmt.Errorf("nodes.parquet not found in %s", inputDir)
		}
		triplesPath = resolveParquetPath(inputDir, "extracted_triples.parquet.gz")
		if triplesPath == "" {
			triplesPath = resolveParquetPath(inputDir, "extracted_triples.parquet")
		}
		if triplesPath == "" {
			return 0, 0, 0, fmt.Errorf("extracted_triples parquet not found in %s", inputDir)
		}
		rulesPath = resolveParquetPath(inputDir, "extracted_rules.parquet.gz")
		if rulesPath == "" {
			rulesPath = resolveParquetPath(inputDir, "extracted_rules.parquet")
		}
		// Create source views over parquet files (embedding already FLOAT[]).
		nodesView := fmt.Sprintf("CREATE OR REPLACE VIEW _src_nodes AS SELECT *, embedding AS emb FROM read_parquet('%s')", nodesPath)
		triplesView := fmt.Sprintf("CREATE OR REPLACE VIEW _src_triples AS SELECT *, embedding AS emb FROM read_parquet('%s')", triplesPath)
		if _, err := d.db.ExecContext(ctx, nodesView); err != nil {
			return 0, 0, 0, fmt.Errorf("failed to create _src_nodes view: %w", err)
		}
		if _, err := d.db.ExecContext(ctx, triplesView); err != nil {
			return 0, 0, 0, fmt.Errorf("failed to create _src_triples view: %w", err)
		}
		if rulesPath != "" {
			rulesView := fmt.Sprintf("CREATE OR REPLACE VIEW _src_rules AS SELECT *, embedding AS emb FROM read_parquet('%s')", rulesPath)
			if _, err := d.db.ExecContext(ctx, rulesView); err != nil {
				return 0, 0, 0, fmt.Errorf("failed to create _src_rules view: %w", err)
			}
		}
		fmt.Println("  Source: Parquet files")
	}

	// Use a dedicated connection so TEMP tables are visible across all statements.
	conn, err := d.db.Conn(ctx)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("failed to acquire connection: %w", err)
	}
	defer conn.Close()

	// Propagate temp_directory and memory settings to this connection so
	// DuckDB can spill temp tables to disk instead of OOM-ing.
	if rows, err := d.db.QueryContext(ctx, "SELECT current_setting('temp_directory')"); err == nil {
		var tempDir string
		if rows.Next() {
			rows.Scan(&tempDir)
		}
		rows.Close()
		if tempDir != "" {
			conn.ExecContext(ctx, fmt.Sprintf("SET temp_directory = '%s'", tempDir))
		}
	}
	if rows, err := d.db.QueryContext(ctx, "SELECT current_setting('memory_limit')"); err == nil {
		var memLimit string
		if rows.Next() {
			rows.Scan(&memLimit)
		}
		rows.Close()
		if memLimit != "" {
			conn.ExecContext(ctx, fmt.Sprintf("SET memory_limit = '%s'", memLimit))
		}
	}
	if rows, err := d.db.QueryContext(ctx, "SELECT current_setting('threads')"); err == nil {
		var threads string
		if rows.Next() {
			rows.Scan(&threads)
		}
		rows.Close()
		if threads != "" {
			conn.ExecContext(ctx, fmt.Sprintf("SET threads = %s", threads))
		}
	}
	conn.ExecContext(ctx, "SET max_temp_directory_size = '500GB'")

	tempTables := []string{
		"_filt_nodes", "_filt_triples", "_filt_rules",
		"_core_names", "_extra_edges", "_extra_rules",
		"_valid_extra_edges", "_neighbor_names",
		"_predicate_map",
	}
	for _, tbl := range tempTables {
		_, _ = conn.ExecContext(ctx, "DROP TABLE IF EXISTS "+tbl)
	}
	defer func() {
		for _, tbl := range tempTables {
			_, _ = conn.ExecContext(ctx, "DROP TABLE IF EXISTS "+tbl)
		}
		// Clean up source views.
		for _, v := range []string{"_src_nodes", "_src_triples", "_src_rules"} {
			_, _ = conn.ExecContext(ctx, "DROP VIEW IF EXISTS "+v)
		}
		if usePostgres {
			_, _ = conn.ExecContext(ctx, "DETACH pg")
		}
	}()

	// Create predicate description mapping table for natural fact text.
	if err := createPredicateMapTable(ctx, conn); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to create predicate map: %w", err)
	}

	dim := d.embeddingDim
	vecLit := float32SliceLiteral(filter.Embedding)

	// TripleThreshold defaults to Threshold when not set.
	tripleThreshold := filter.TripleThreshold
	if tripleThreshold == 0 {
		tripleThreshold = filter.Threshold
	}

	// EdgeThreshold defaults to 0.4 when not set.
	edgeThreshold := filter.EdgeThreshold
	if edgeThreshold == 0 {
		edgeThreshold = 0.4
	}

	// EdgeBatchSize defaults to 10000 when not set.
	edgeBatchSize := filter.EdgeBatchSize
	if edgeBatchSize == 0 {
		edgeBatchSize = 10000
	}

	totalPhases := 12

	// Phase 1: Filter nodes by embedding similarity.
	fmt.Printf("  Phase 1/%d: Filtering nodes by embedding (threshold %.2f)...\n", totalPhases, filter.Threshold)
	phaseStart := time.Now()
	// Two-step: materialize valid embeddings first, then compute similarity.
	// DuckDB's optimizer may reorder WHERE predicates, so list_cosine_similarity
	// can be evaluated on rows with NULL-containing lists before list_count filters them.
	validNodesSQL := fmt.Sprintf(`CREATE TEMP TABLE _valid_emb_nodes AS
		SELECT id, name, LOWER(TRIM(name)) AS nm, emb
		FROM _src_nodes
		WHERE list_count(emb) = %d`,
		dim)
	if _, err := conn.ExecContext(ctx, validNodesSQL); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to create valid embedding nodes: %w", err)
	}
	filtNodesSQL := fmt.Sprintf(`CREATE TEMP TABLE _filt_nodes AS
		SELECT id, name, nm
		FROM _valid_emb_nodes
		WHERE list_cosine_similarity(emb::FLOAT[], %s::FLOAT[]) >= %g`,
		vecLit, filter.Threshold)
	if _, err := conn.ExecContext(ctx, filtNodesSQL); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to filter nodes: %w", err)
	}
	conn.ExecContext(ctx, "DROP TABLE IF EXISTS _valid_emb_nodes")
	var filtNodeCount int64
	if row := conn.QueryRowContext(ctx, "SELECT COUNT(*) FROM _filt_nodes"); row != nil {
		_ = row.Scan(&filtNodeCount)
	}
	fmt.Printf("  Phase 1/%d: Done — %d nodes match threshold in %s\n", totalPhases, filtNodeCount, time.Since(phaseStart).Round(time.Second))

	// Phase 2: Filter triples by embedding similarity (lower threshold).
	fmt.Printf("  Phase 2/%d: Filtering triples by embedding (threshold %.2f)...\n", totalPhases, tripleThreshold)
	phaseStart = time.Now()
	validTriplesSQL := fmt.Sprintf(`CREATE TEMP TABLE _valid_emb_triples AS
		SELECT * FROM _src_triples
		WHERE list_count(emb) = %d`,
		dim)
	if _, err := conn.ExecContext(ctx, validTriplesSQL); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to create valid embedding triples: %w", err)
	}
	filtTriplesSQL := fmt.Sprintf(`CREATE TEMP TABLE _filt_triples AS
		SELECT * FROM _valid_emb_triples
		WHERE list_cosine_similarity(emb::FLOAT[], %s::FLOAT[]) >= %g`,
		vecLit, tripleThreshold)
	// Exclude specific predicates if configured
	if len(filter.ExcludePredicates) > 0 {
		quoted := make([]string, len(filter.ExcludePredicates))
		for i, p := range filter.ExcludePredicates {
			quoted[i] = fmt.Sprintf("'%s'", strings.ReplaceAll(strings.ToLower(p), "'", "''"))
		}
		filtTriplesSQL += fmt.Sprintf("\n\t\tAND LOWER(TRIM(predicate)) NOT IN (%s)", strings.Join(quoted, ", "))
	}
	if _, err := conn.ExecContext(ctx, filtTriplesSQL); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to filter triples: %w", err)
	}
	conn.ExecContext(ctx, "DROP TABLE IF EXISTS _valid_emb_triples")
	var filtTripleCount int64
	if row := conn.QueryRowContext(ctx, "SELECT COUNT(*) FROM _filt_triples"); row != nil {
		_ = row.Scan(&filtTripleCount)
	}
	fmt.Printf("  Phase 2/%d: Done — %d triples match threshold in %s\n", totalPhases, filtTripleCount, time.Since(phaseStart).Round(time.Second))

	// Phase 3: Filter rules by embedding similarity (same lower threshold).
	fmt.Printf("  Phase 3/%d: Filtering rules by embedding (threshold %.2f)...\n", totalPhases, tripleThreshold)
	phaseStart = time.Now()
	hasRules := usePostgres || rulesPath != ""
	if hasRules {
		validRulesSQL := fmt.Sprintf(`CREATE TEMP TABLE _valid_emb_rules AS
			SELECT * FROM _src_rules
			WHERE list_count(emb) = %d`,
			dim)
		if _, err := conn.ExecContext(ctx, validRulesSQL); err != nil {
			return 0, 0, 0, fmt.Errorf("failed to create valid embedding rules: %w", err)
		}
		filtRulesSQL := fmt.Sprintf(`CREATE TEMP TABLE _filt_rules AS
			SELECT * FROM _valid_emb_rules
			WHERE list_cosine_similarity(emb::FLOAT[], %s::FLOAT[]) >= %g`,
			vecLit, tripleThreshold)
		if _, err := conn.ExecContext(ctx, filtRulesSQL); err != nil {
			return 0, 0, 0, fmt.Errorf("failed to filter rules: %w", err)
		}
		conn.ExecContext(ctx, "DROP TABLE IF EXISTS _valid_emb_rules")
	} else {
		if _, err := conn.ExecContext(ctx, `CREATE TEMP TABLE _filt_rules (
			id VARCHAR, source_id VARCHAR, antecedent VARCHAR, consequent VARCHAR,
			exception VARCHAR, rule_type VARCHAR, scope VARCHAR,
			source_attribution VARCHAR, embedding FLOAT[], confidence DOUBLE,
			created_at TIMESTAMP, chunk_index INTEGER
		)`); err != nil {
			return 0, 0, 0, fmt.Errorf("failed to create empty rules table: %w", err)
		}
	}
	var filtRuleCount int64
	if row := conn.QueryRowContext(ctx, "SELECT COUNT(*) FROM _filt_rules"); row != nil {
		_ = row.Scan(&filtRuleCount)
	}
	fmt.Printf("  Phase 3/%d: Done — %d rules match threshold in %s\n", totalPhases, filtRuleCount, time.Since(phaseStart).Round(time.Second))

	// Phase 4: Build core qualifying node set = filtered nodes + triple endpoints + rule-referenced nodes.
	// The rule-referenced lookup is done in two passes to avoid a full O(nodes×rules) cross-join:
	//   4a: Build initial core set from filtered nodes + triple endpoints (fast).
	//   4b: Check which initial core names appear in rule text, then scan only the
	//        parquet partitions that might contain additional rule-referenced names
	//        in batches to limit memory/time.
	fmt.Printf("  Phase 4/%d: Building core qualifying node set...\n", totalPhases)
	phaseStart = time.Now()

	// 4a: Fast core set from filtered nodes + triple endpoints.
	coreNamesSQL := `CREATE TEMP TABLE _core_names AS
		SELECT nm FROM _filt_nodes
		UNION
		SELECT DISTINCT LOWER(TRIM(subject)) AS nm FROM _filt_triples
		UNION
		SELECT DISTINCT LOWER(TRIM(object)) AS nm FROM _filt_triples`
	if _, err := conn.ExecContext(ctx, coreNamesSQL); err != nil {
		return 0, 0, 0, fmt.Errorf("failed to build initial core node set: %w", err)
	}
	var initialCoreCount int64
	if row := conn.QueryRowContext(ctx, "SELECT COUNT(*) FROM _core_names"); row != nil {
		_ = row.Scan(&initialCoreCount)
	}
	fmt.Printf("    Phase 4a: %d core names from nodes+triples in %s\n", initialCoreCount, time.Since(phaseStart).Round(time.Second))

	// 4b: Find additional names mentioned in rules by scanning nodes against
	// a single concatenated rule-text blob. This avoids an O(nodes × rules) EXISTS
	// correlated subquery — instead each node does one position() against the blob.
	// For parquet: process partitions in chunks for progress + memory control.
	// For postgres: query _src_nodes directly (PG handles the scan).
	if filtRuleCount > 0 {
		rulePhaseStart := time.Now()
		// Build a single blob of all rule text for substring matching.
		blobSQL := `CREATE TEMP TABLE _rule_blob AS
			SELECT string_agg(LOWER(COALESCE(antecedent,'') || ' ' || COALESCE(consequent,'')), ' ||| ') AS txt
			FROM _filt_rules`
		if _, err := conn.ExecContext(ctx, blobSQL); err != nil {
			fmt.Printf("    Warning: failed to build rule text blob: %v\n", err)
		} else if usePostgres {
			// Postgres path: query _src_nodes directly, no chunking needed.
			ruleNameSQL := `INSERT INTO _core_names
				SELECT DISTINCT nm FROM (
					SELECT LOWER(TRIM(n.name)) AS nm
					FROM _src_nodes n
				) sub
				WHERE nm NOT IN (SELECT nm FROM _core_names)
				  AND EXISTS (SELECT 1 FROM _rule_blob WHERE position(sub.nm IN txt) > 0)`
			res, err := conn.ExecContext(ctx, ruleNameSQL)
			var ruleNamesAdded int64
			if err != nil {
				fmt.Printf("    Warning: rule-referenced name scan failed: %v\n", err)
			} else {
				ruleNamesAdded, _ = res.RowsAffected()
			}
			conn.ExecContext(ctx, "DROP TABLE IF EXISTS _rule_blob")
			fmt.Printf("    Phase 4b: %d rule-referenced names added in %s\n",
				ruleNamesAdded, time.Since(rulePhaseStart).Round(time.Second))
		} else {
			// Parquet path: discover partition files and process in chunks.
			var partFiles []string
			partRows, err := conn.QueryContext(ctx, fmt.Sprintf(
				`SELECT file FROM glob('%s')`, nodesPath))
			if err == nil {
				for partRows.Next() {
					var f string
					if partRows.Scan(&f) == nil {
						partFiles = append(partFiles, f)
					}
				}
				partRows.Close()
			}

			if len(partFiles) == 0 {
				// Single file, not a partitioned directory.
				partFiles = []string{nodesPath}
			}

			// Process in chunks.
			const chunkSize = 20
			var ruleNamesAdded int64
			for i := 0; i < len(partFiles); i += chunkSize {
				end := i + chunkSize
				if end > len(partFiles) {
					end = len(partFiles)
				}
				// Build a list of partition files for this chunk.
				chunkList := "'" + partFiles[i] + "'"
				for _, f := range partFiles[i+1 : end] {
					chunkList += ",'" + f + "'"
				}
				chunkSQL := fmt.Sprintf(`INSERT INTO _core_names
					SELECT DISTINCT nm FROM (
						SELECT LOWER(TRIM(n.name)) AS nm
						FROM read_parquet([%s]) n
					) sub
					WHERE nm NOT IN (SELECT nm FROM _core_names)
					  AND EXISTS (SELECT 1 FROM _rule_blob WHERE position(sub.nm IN txt) > 0)`,
					chunkList)
				res, err := conn.ExecContext(ctx, chunkSQL)
				if err != nil {
					fmt.Printf("    Warning: chunk %d-%d/%d failed: %v\n", i+1, end, len(partFiles), err)
					continue
				}
				n, _ := res.RowsAffected()
				ruleNamesAdded += n
				fmt.Printf("    Phase 4b: chunk %d-%d/%d done (+%d names)\n",
					i+1, end, len(partFiles), n)
			}
			conn.ExecContext(ctx, "DROP TABLE IF EXISTS _rule_blob")
			fmt.Printf("    Phase 4b: %d rule-referenced names added in %s\n",
				ruleNamesAdded, time.Since(rulePhaseStart).Round(time.Second))
		}
	}

	var coreNameCount int64
	if row := conn.QueryRowContext(ctx, "SELECT COUNT(*) FROM _core_names"); row != nil {
		_ = row.Scan(&coreNameCount)
	}
	fmt.Printf("  Phase 4/%d: Done — %d core node names in %s\n", totalPhases, coreNameCount, time.Since(phaseStart).Round(time.Second))

	// Phase 5: Insert core nodes into entities.
	fmt.Printf("  Phase 5/%d: Inserting core nodes...\n", totalPhases)
	phaseStart = time.Now()
	nodesLimit := ""
	if filter.MaxNodes > 0 {
		nodesLimit = fmt.Sprintf("LIMIT %d", filter.MaxNodes)
	}
	coreNodeInsertSQL := fmt.Sprintf(`INSERT INTO entities (
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
			CASE WHEN emb IS NOT NULL AND array_length(emb) = %d THEN emb ELSE NULL END,
			CASE WHEN emb IS NOT NULL AND array_length(emb) = %d THEN emb ELSE NULL END,
			json_array(COALESCE(source_id, 'wikidata')),
			COALESCE(created_at, CURRENT_TIMESTAMP),
			CURRENT_TIMESTAMP,
			COALESCE(created_at, CURRENT_TIMESTAMP)
		FROM _src_nodes
		WHERE LOWER(TRIM(name)) IN (SELECT nm FROM _core_names)
		%s`,
		groupID, dim, dim, nodesLimit)
	res, err := conn.ExecContext(ctx, coreNodeInsertSQL)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("failed to insert core nodes: %w", err)
	}
	coreNodesLoaded, _ := res.RowsAffected()
	fmt.Printf("  Phase 5/%d: Done — %d core nodes inserted in %s\n", totalPhases, coreNodesLoaded, time.Since(phaseStart).Round(time.Second))

	// Create a lightweight entity lookup table (uuid + name + group_id only, no embeddings)
	// for edge INSERT JOINs. The full entities table with 1024-dim embeddings is too large
	// to join against repeatedly without OOM.
	conn.ExecContext(ctx, "DROP TABLE IF EXISTS _entity_lookup")
	if _, err := conn.ExecContext(ctx, `CREATE TEMP TABLE _entity_lookup AS
		SELECT uuid, LOWER(TRIM(name)) AS nm, group_id FROM entities WHERE type = 'entity'`); err != nil {
		return coreNodesLoaded, 0, 0, fmt.Errorf("failed to create entity lookup: %w", err)
	}

	// Phase 6: Insert filtered triples as edges in batches (entity resolution JOIN + dedup).
	fmt.Printf("  Phase 6/%d: Inserting filtered triples as edges (batch size %d)...\n", totalPhases, edgeBatchSize)
	phaseStart = time.Now()
	// Add a rowid column for OFFSET/LIMIT batching.
	conn.ExecContext(ctx, "ALTER TABLE _filt_triples ADD COLUMN IF NOT EXISTS _rownum INTEGER")
	conn.ExecContext(ctx, "UPDATE _filt_triples SET _rownum = rowid")
	var filtEdgesLoaded int64
	for offset := int64(0); ; offset += edgeBatchSize {
		batchSQL := fmt.Sprintf(`INSERT INTO edges (
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
				`+factDescriptionExpr+`,
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
			FROM (SELECT * FROM _filt_triples ORDER BY _rownum LIMIT %d OFFSET %d) t
			JOIN _entity_lookup subj
				ON LOWER(TRIM(t.subject)) = subj.nm
				AND subj.group_id = COALESCE(t.group_id, '%s')
			JOIN _entity_lookup obj
				ON LOWER(TRIM(t.object)) = obj.nm
				AND obj.group_id = COALESCE(t.group_id, '%s')
			QUALIFY ROW_NUMBER() OVER (PARTITION BY t.id, COALESCE(t.group_id, '%s') ORDER BY subj.uuid) = 1`,
			groupID, dim, dim, edgeBatchSize, offset, groupID, groupID, groupID)
		res, err = conn.ExecContext(ctx, batchSQL)
		if err != nil {
			return coreNodesLoaded, filtEdgesLoaded, 0, fmt.Errorf("failed to insert filtered triple edges (offset %d): %w", offset, err)
		}
		n, _ := res.RowsAffected()
		filtEdgesLoaded += n
		if n == 0 && offset >= filtTripleCount {
			break
		}
		fmt.Printf("    Phase 6: batch %d-%d done (+%d edges, %d total)\n",
			offset+1, offset+edgeBatchSize, n, filtEdgesLoaded)
		if offset+edgeBatchSize >= filtTripleCount {
			break
		}
	}
	fmt.Printf("  Phase 6/%d: Done — %d filtered triple edges inserted in %s\n", totalPhases, filtEdgesLoaded, time.Since(phaseStart).Round(time.Second))

	// Phase 7: Insert filtered rules as edges.
	fmt.Printf("  Phase 7/%d: Inserting filtered rules...\n", totalPhases)
	phaseStart = time.Now()
	ruleInsertSQL := fmt.Sprintf(`INSERT INTO edges (
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
		FROM _filt_rules`,
		groupID, dim, dim)
	res, err = conn.ExecContext(ctx, ruleInsertSQL)
	if err != nil {
		return coreNodesLoaded, filtEdgesLoaded, 0, fmt.Errorf("failed to insert filtered rules: %w", err)
	}
	rulesLoaded, _ := res.RowsAffected()
	fmt.Printf("  Phase 7/%d: Done — %d rules inserted in %s\n", totalPhases, rulesLoaded, time.Since(phaseStart).Round(time.Second))

	// Phase 8: Collect additional edges (triples + rules) touching core nodes, filtered by edge threshold.
	// Excludes already-inserted filtered triples and rules.
	fmt.Printf("  Phase 8/%d: Collecting extra edges touching core nodes (threshold %.2f)...\n", totalPhases, edgeThreshold)
	phaseStart = time.Now()
	edgesLimit := ""
	if filter.MaxEdges > 0 {
		edgesLimit = fmt.Sprintf("LIMIT %d", filter.MaxEdges)
	}
	// Step 1: materialize valid extra triples (touching core nodes, not already filtered).
	validExtraTriplesSQL := fmt.Sprintf(`CREATE TEMP TABLE _valid_extra_edges AS
		SELECT * FROM _src_triples
		WHERE list_count(emb) = %d
		  AND (LOWER(TRIM(subject)) IN (SELECT nm FROM _core_names)
		   OR  LOWER(TRIM(object))  IN (SELECT nm FROM _core_names))
		  AND id NOT IN (SELECT id FROM _filt_triples)`,
		dim)
	// Exclude specific predicates from expansion edges too
	if len(filter.ExcludePredicates) > 0 {
		quoted := make([]string, len(filter.ExcludePredicates))
		for i, p := range filter.ExcludePredicates {
			quoted[i] = fmt.Sprintf("'%s'", strings.ReplaceAll(strings.ToLower(p), "'", "''"))
		}
		validExtraTriplesSQL += fmt.Sprintf("\n\t\t  AND LOWER(TRIM(predicate)) NOT IN (%s)", strings.Join(quoted, ", "))
	}
	if _, err := conn.ExecContext(ctx, validExtraTriplesSQL); err != nil {
		return coreNodesLoaded, filtEdgesLoaded, rulesLoaded, fmt.Errorf("failed to create valid extra edges: %w", err)
	}
	// Step 2: filter by edge threshold on embedding similarity.
	extraEdgesSQL := fmt.Sprintf(`CREATE TEMP TABLE _extra_edges AS
		SELECT * FROM _valid_extra_edges
		WHERE list_cosine_similarity(emb::FLOAT[], %s::FLOAT[]) >= %g
		%s`,
		vecLit, edgeThreshold, edgesLimit)
	if _, err := conn.ExecContext(ctx, extraEdgesSQL); err != nil {
		return coreNodesLoaded, filtEdgesLoaded, rulesLoaded, fmt.Errorf("failed to filter extra edges: %w", err)
	}
	conn.ExecContext(ctx, "DROP TABLE IF EXISTS _valid_extra_edges")
	// Also collect extra rules touching core nodes (not already in _filt_rules).
	if hasRules {
		extraRulesSQL := fmt.Sprintf(`CREATE TEMP TABLE _extra_rules AS
			SELECT * FROM _src_rules
			WHERE list_count(emb) = %d
			  AND id NOT IN (SELECT id FROM _filt_rules)
			  AND list_cosine_similarity(emb::FLOAT[], %s::FLOAT[]) >= %g
			  AND EXISTS (
				SELECT 1 FROM _core_names cn
				WHERE position(cn.nm IN LOWER(antecedent)) > 0
				   OR position(cn.nm IN LOWER(consequent)) > 0
			  )`,
			dim, vecLit, edgeThreshold)
		if _, err := conn.ExecContext(ctx, extraRulesSQL); err != nil {
			return coreNodesLoaded, filtEdgesLoaded, rulesLoaded, fmt.Errorf("failed to collect extra rules: %w", err)
		}
	} else {
		conn.ExecContext(ctx, `CREATE TEMP TABLE _extra_rules (
			id VARCHAR, source_id VARCHAR, antecedent VARCHAR, consequent VARCHAR,
			exception VARCHAR, rule_type VARCHAR, scope VARCHAR,
			source_attribution VARCHAR, embedding FLOAT[], confidence DOUBLE,
			created_at TIMESTAMP, chunk_index INTEGER
		)`)
	}
	var extraEdgeCount int64
	if row := conn.QueryRowContext(ctx, "SELECT COUNT(*) FROM _extra_edges"); row != nil {
		_ = row.Scan(&extraEdgeCount)
	}
	var extraRuleCount int64
	if row := conn.QueryRowContext(ctx, "SELECT COUNT(*) FROM _extra_rules"); row != nil {
		_ = row.Scan(&extraRuleCount)
	}
	fmt.Printf("  Phase 8/%d: Done — %d extra triples + %d extra rules collected in %s\n", totalPhases, extraEdgeCount, extraRuleCount, time.Since(phaseStart).Round(time.Second))

	// Drop embedding columns from temp tables — they're no longer needed after
	// similarity filtering and consume ~4KB per row (1024 floats).
	for _, tbl := range []string{"_extra_edges", "_extra_rules", "_filt_triples", "_filt_rules"} {
		conn.ExecContext(ctx, fmt.Sprintf("ALTER TABLE %s DROP COLUMN IF EXISTS emb", tbl))
		conn.ExecContext(ctx, fmt.Sprintf("ALTER TABLE %s DROP COLUMN IF EXISTS embedding", tbl))
	}

	// Phase 9: Find neighbor nodes from extra edges (endpoints not already in core set).
	fmt.Printf("  Phase 9/%d: Finding neighbor nodes...\n", totalPhases)
	phaseStart = time.Now()
	neighborSQL := `CREATE TEMP TABLE _neighbor_names AS
		SELECT DISTINCT LOWER(TRIM(subject)) AS nm FROM _extra_edges
		WHERE LOWER(TRIM(subject)) NOT IN (SELECT nm FROM _core_names)
		UNION
		SELECT DISTINCT LOWER(TRIM(object)) AS nm FROM _extra_edges
		WHERE LOWER(TRIM(object)) NOT IN (SELECT nm FROM _core_names)`
	if _, err := conn.ExecContext(ctx, neighborSQL); err != nil {
		return coreNodesLoaded, filtEdgesLoaded, rulesLoaded, fmt.Errorf("failed to find neighbor nodes: %w", err)
	}
	var neighborCount int64
	if row := conn.QueryRowContext(ctx, "SELECT COUNT(*) FROM _neighbor_names"); row != nil {
		_ = row.Scan(&neighborCount)
	}
	fmt.Printf("  Phase 9/%d: Done — %d neighbor node names in %s\n", totalPhases, neighborCount, time.Since(phaseStart).Round(time.Second))

	// Phase 10: Insert neighbor nodes.
	fmt.Printf("  Phase 10/%d: Inserting neighbor nodes...\n", totalPhases)
	phaseStart = time.Now()
	neighborNodeInsertSQL := fmt.Sprintf(`INSERT INTO entities (
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
			CASE WHEN emb IS NOT NULL AND array_length(emb) = %d THEN emb ELSE NULL END,
			CASE WHEN emb IS NOT NULL AND array_length(emb) = %d THEN emb ELSE NULL END,
			json_array(COALESCE(source_id, 'wikidata')),
			COALESCE(created_at, CURRENT_TIMESTAMP),
			CURRENT_TIMESTAMP,
			COALESCE(created_at, CURRENT_TIMESTAMP)
		FROM _src_nodes
		WHERE LOWER(TRIM(name)) IN (SELECT nm FROM _neighbor_names)`,
		groupID, dim, dim)
	res, err = conn.ExecContext(ctx, neighborNodeInsertSQL)
	if err != nil {
		return coreNodesLoaded, filtEdgesLoaded, rulesLoaded, fmt.Errorf("failed to insert neighbor nodes: %w", err)
	}
	neighborNodesLoaded, _ := res.RowsAffected()
	nodesLoaded := coreNodesLoaded + neighborNodesLoaded
	fmt.Printf("  Phase 10/%d: Done — %d neighbor nodes inserted (%d total) in %s\n",
		totalPhases, neighborNodesLoaded, nodesLoaded, time.Since(phaseStart).Round(time.Second))

	// Refresh _entity_lookup to include neighbor nodes inserted in Phase 10.
	conn.ExecContext(ctx, "DROP TABLE IF EXISTS _entity_lookup")
	if _, err := conn.ExecContext(ctx, `CREATE TEMP TABLE _entity_lookup AS
		SELECT uuid, LOWER(TRIM(name)) AS nm, group_id FROM entities WHERE type = 'entity'`); err != nil {
		return nodesLoaded, filtEdgesLoaded, rulesLoaded, fmt.Errorf("failed to refresh entity lookup: %w", err)
	}

	// Phase 11: Insert extra edges + extra rules (batched).
	fmt.Printf("  Phase 11/%d: Inserting extra edges and rules (batch size %d)...\n", totalPhases, edgeBatchSize)
	phaseStart = time.Now()
	conn.ExecContext(ctx, "ALTER TABLE _extra_edges ADD COLUMN IF NOT EXISTS _rownum INTEGER")
	conn.ExecContext(ctx, "UPDATE _extra_edges SET _rownum = rowid")
	var extraEdgesInserted int64
	for offset := int64(0); ; offset += edgeBatchSize {
		batchSQL := fmt.Sprintf(`INSERT INTO edges (
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
				`+factDescriptionExpr+`,
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
			FROM (SELECT * FROM _extra_edges ORDER BY _rownum LIMIT %d OFFSET %d) t
			JOIN _entity_lookup subj
				ON LOWER(TRIM(t.subject)) = subj.nm
				AND subj.group_id = COALESCE(t.group_id, '%s')
			JOIN _entity_lookup obj
				ON LOWER(TRIM(t.object)) = obj.nm
				AND obj.group_id = COALESCE(t.group_id, '%s')
			QUALIFY ROW_NUMBER() OVER (PARTITION BY t.id, COALESCE(t.group_id, '%s') ORDER BY subj.uuid) = 1`,
			groupID, dim, dim, edgeBatchSize, offset, groupID, groupID, groupID)
		res, err = conn.ExecContext(ctx, batchSQL)
		if err != nil {
			return nodesLoaded, filtEdgesLoaded, rulesLoaded, fmt.Errorf("failed to insert extra edges (offset %d): %w", offset, err)
		}
		n, _ := res.RowsAffected()
		extraEdgesInserted += n
		if n == 0 && offset >= extraEdgeCount {
			break
		}
		fmt.Printf("    Phase 11: batch %d-%d done (+%d edges, %d total)\n",
			offset+1, offset+edgeBatchSize, n, extraEdgesInserted)
		if offset+edgeBatchSize >= extraEdgeCount {
			break
		}
	}

	// Insert extra rules as edges.
	extraRuleInsertSQL := fmt.Sprintf(`INSERT INTO edges (
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
		FROM _extra_rules`,
		groupID, dim, dim)
	res, err = conn.ExecContext(ctx, extraRuleInsertSQL)
	if err != nil {
		return nodesLoaded, filtEdgesLoaded + extraEdgesInserted, rulesLoaded, fmt.Errorf("failed to insert extra rules: %w", err)
	}
	extraRulesInserted, _ := res.RowsAffected()
	edgesLoaded := filtEdgesLoaded + extraEdgesInserted
	rulesLoaded += extraRulesInserted
	fmt.Printf("  Phase 11/%d: Done — %d extra edges + %d extra rules inserted in %s\n",
		totalPhases, extraEdgesInserted, extraRulesInserted, time.Since(phaseStart).Round(time.Second))

	// Phase 12: Recreate property graph to pick up new data.
	fmt.Printf("  Phase 12/%d: Creating property graph...\n", totalPhases)
	phaseStart = time.Now()
	pgStmt := `CREATE OR REPLACE PROPERTY GRAPH knowledge_graph
		VERTEX TABLES (entities LABEL entity)
		EDGE TABLES (
			edges SOURCE KEY (source_id, group_id) REFERENCES entities (uuid, group_id)
			      DESTINATION KEY (target_id, group_id) REFERENCES entities (uuid, group_id)
			      LABEL relates_to
		)`
	_, _ = conn.ExecContext(ctx, pgStmt)
	fmt.Printf("  Phase 12/%d: Done in %s\n", totalPhases, time.Since(phaseStart).Round(time.Second))

	return nodesLoaded, edgesLoaded, rulesLoaded, nil
}

// SimilarTriple holds a triple with its cosine similarity score to a query vector.
type SimilarTriple struct {
	Similarity float64
	Subject    string
	Predicate  string
	Object     string
}

// SimilarRule holds a rule with its cosine similarity score to a query vector.
type SimilarRule struct {
	Similarity float64
	Antecedent string
	Consequent string
}

// TopSimilarTriples returns the top-N triples from the parquet files ranked by
// cosine similarity of their embedding to vec. Uses an in-memory DuckDB connection.
func (d *DuckPGQDriver) TopSimilarTriples(ctx context.Context, inputDir string, vec []float32, limit int) ([]SimilarTriple, error) {
	if len(vec) != d.embeddingDim {
		return nil, fmt.Errorf("vector dimension mismatch: got %d, expected %d", len(vec), d.embeddingDim)
	}
	triplesPath := resolveParquetPath(inputDir, "extracted_triples.parquet.gz")
	if triplesPath == "" {
		triplesPath = resolveParquetPath(inputDir, "extracted_triples.parquet")
	}
	if triplesPath == "" {
		return nil, fmt.Errorf("extracted_triples parquet not found in %s", inputDir)
	}

	memDB, err := sql.Open("duckdb", "")
	if err != nil {
		return nil, fmt.Errorf("failed to open in-memory DuckDB: %w", err)
	}
	defer memDB.Close()

	dim := d.embeddingDim
	vecLit := float32SliceLiteral(vec)
	q := fmt.Sprintf(`
		SELECT COALESCE(subject,''), COALESCE(predicate,''), COALESCE(object,''),
		       list_cosine_similarity(embedding::FLOAT[], %s::FLOAT[]) AS sim
		FROM read_parquet('%s')
		WHERE list_count(embedding) = %d
		ORDER BY sim DESC
		LIMIT %d`,
		vecLit, triplesPath, dim, limit)

	rows, err := memDB.QueryContext(ctx, q)
	if err != nil {
		return nil, fmt.Errorf("top similar triples query failed: %w", err)
	}
	defer rows.Close()

	var result []SimilarTriple
	for rows.Next() {
		var r SimilarTriple
		if err := rows.Scan(&r.Subject, &r.Predicate, &r.Object, &r.Similarity); err != nil {
			return nil, fmt.Errorf("scanning similar triple: %w", err)
		}
		result = append(result, r)
	}
	return result, rows.Err()
}

// TopSimilarRules returns the top-N rules from the parquet files ranked by
// cosine similarity of their embedding to vec. Uses an in-memory DuckDB connection.
func (d *DuckPGQDriver) TopSimilarRules(ctx context.Context, inputDir string, vec []float32, limit int) ([]SimilarRule, error) {
	if len(vec) != d.embeddingDim {
		return nil, fmt.Errorf("vector dimension mismatch: got %d, expected %d", len(vec), d.embeddingDim)
	}
	rulesPath := resolveParquetPath(inputDir, "extracted_rules.parquet.gz")
	if rulesPath == "" {
		rulesPath = resolveParquetPath(inputDir, "extracted_rules.parquet")
	}
	if rulesPath == "" {
		return nil, nil // rules are optional
	}

	memDB, err := sql.Open("duckdb", "")
	if err != nil {
		return nil, fmt.Errorf("failed to open in-memory DuckDB: %w", err)
	}
	defer memDB.Close()

	dim := d.embeddingDim
	vecLit := float32SliceLiteral(vec)
	q := fmt.Sprintf(`
		SELECT COALESCE(antecedent,''), COALESCE(consequent,''),
		       list_cosine_similarity(embedding::FLOAT[], %s::FLOAT[]) AS sim
		FROM read_parquet('%s')
		WHERE list_count(embedding) = %d
		ORDER BY sim DESC
		LIMIT %d`,
		vecLit, rulesPath, dim, limit)

	rows, err := memDB.QueryContext(ctx, q)
	if err != nil {
		return nil, fmt.Errorf("top similar rules query failed: %w", err)
	}
	defer rows.Close()

	var result []SimilarRule
	for rows.Next() {
		var r SimilarRule
		if err := rows.Scan(&r.Antecedent, &r.Consequent, &r.Similarity); err != nil {
			return nil, fmt.Errorf("scanning similar rule: %w", err)
		}
		result = append(result, r)
	}
	return result, rows.Err()
}

// float32SliceLiteral formats a []float32 as a DuckDB array literal suitable
// for inline SQL embedding, e.g. `[0.1, 0.2, 0.3]::FLOAT[3]`.
func float32SliceLiteral(v []float32) string {
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

// Compile-time interface checks
var _ GraphDriver = (*DuckPGQDriver)(nil)
var _ GraphExporter = (*DuckPGQDriver)(nil)
