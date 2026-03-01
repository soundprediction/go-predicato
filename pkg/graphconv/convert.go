//go:build system_duckpgq

package graphconv

import (
	"database/sql"
	"fmt"
	"os"
	"sort"
	"strings"

	_ "github.com/marcboeker/go-duckdb"
)

// Options configures the conversion.
type Options struct {
	Source       string
	Dest         string
	EmbeddingDim int
	CSRTable     string // icebug table prefix (default "predicato")
	GroupID      string // optional filter
	Directed     bool
	SchemaOut    string // optional schema.cypher output path
}

// entityTypes are the node types stored in the DuckPGQ entities table.
var entityTypes = []string{"entity", "episodic", "community", "rule"}

// ToIcebug converts a DuckPGQ database to icebug CSR format.
func ToIcebug(opts Options) error {
	if opts.CSRTable == "" {
		opts.CSRTable = "predicato"
	}

	destDB, err := openDuckDB(opts.Dest)
	if err != nil {
		return fmt.Errorf("open dest: %w", err)
	}
	defer destDB.Close()

	// Attach source
	if _, err := destDB.Exec(fmt.Sprintf(`ATTACH '%s' AS src (READ_ONLY)`, opts.Source)); err != nil {
		return fmt.Errorf("attach source: %w", err)
	}

	groupFilter := ""
	if opts.GroupID != "" {
		groupFilter = fmt.Sprintf(" AND group_id = '%s'", opts.GroupID)
	}

	// 1. Create node tables per type
	for _, typ := range entityTypes {
		table := "nodes_" + typ
		if _, err := destDB.Exec(fmt.Sprintf(`CREATE OR REPLACE TABLE %s AS
			SELECT uuid, group_id, name, content, summary,
				entity_type, episode_type, metadata,
				created_at, updated_at, valid_from, valid_to,
				source_ids, entity_edges, level,
				embedding, name_embedding
			FROM src.entities
			WHERE type = '%s'%s
			ORDER BY uuid`, table, typ, groupFilter)); err != nil {
			return fmt.Errorf("create %s: %w", table, err)
		}
		fmt.Printf("  created %s\n", table)
	}

	// 2. Create mapping tables (uuid -> contiguous csr_index)
	for _, typ := range entityTypes {
		mapping := fmt.Sprintf("%s_mapping_%s", opts.CSRTable, typ)
		src := "nodes_" + typ
		if _, err := destDB.Exec(fmt.Sprintf(`CREATE OR REPLACE TABLE %s AS
			SELECT uuid, (ROW_NUMBER() OVER (ORDER BY uuid)) - 1 AS csr_index
			FROM %s`, mapping, src)); err != nil {
			return fmt.Errorf("create mapping %s: %w", mapping, err)
		}
	}

	// Also create a unified mapping across all types for edge resolution
	if _, err := destDB.Exec(fmt.Sprintf(`CREATE OR REPLACE TABLE %s_mapping_all AS
		SELECT uuid, group_id, (ROW_NUMBER() OVER (ORDER BY uuid)) - 1 AS csr_index
		FROM (
			SELECT uuid, group_id FROM nodes_entity
			UNION ALL SELECT uuid, group_id FROM nodes_episodic
			UNION ALL SELECT uuid, group_id FROM nodes_community
			UNION ALL SELECT uuid, group_id FROM nodes_rule
		)`, opts.CSRTable)); err != nil {
		return fmt.Errorf("create unified mapping: %w", err)
	}

	// 3. Get distinct edge types and build CSR per type
	edgeTypes, err := getDistinctEdgeTypes(destDB, groupFilter)
	if err != nil {
		return fmt.Errorf("get edge types: %w", err)
	}

	if len(edgeTypes) == 0 {
		// No edges — still create empty CSR tables for the default type
		edgeTypes = []string{"entity"}
	}

	mappingAll := fmt.Sprintf("%s_mapping_all", opts.CSRTable)

	for _, eType := range edgeTypes {
		edgeFilter := fmt.Sprintf("type = '%s'%s", eType, groupFilter)

		// Create sorted edge list joined with mapping
		edgesView := fmt.Sprintf("%s_edges_sorted_%s", opts.CSRTable, eType)
		if _, err := destDB.Exec(fmt.Sprintf(`CREATE OR REPLACE TEMP TABLE %s AS
			SELECT
				sm.csr_index AS csr_source,
				tm.csr_index AS csr_target,
				e.uuid, e.name, e.fact, e.attributes, e.episodes,
				e.strength, e.created_at, e.updated_at,
				e.valid_from, e.valid_to, e.expired_at,
				e.source_ids, e.embedding, e.fact_embedding
			FROM src.edges e
			JOIN %s sm ON e.source_id = sm.uuid
			JOIN %s tm ON e.target_id = tm.uuid
			WHERE %s
			ORDER BY csr_source, csr_target`,
			edgesView, mappingAll, mappingAll, edgeFilter)); err != nil {
			return fmt.Errorf("create sorted edges for %s: %w", eType, err)
		}

		// Build indices table (the CSR column store)
		indicesTable := fmt.Sprintf("%s_indices_%s", opts.CSRTable, eType)
		if _, err := destDB.Exec(fmt.Sprintf(`CREATE OR REPLACE TABLE %s AS
			SELECT csr_source, csr_target,
				uuid, name, fact, attributes, episodes,
				strength, created_at, updated_at,
				valid_from, valid_to, expired_at,
				source_ids, embedding, fact_embedding
			FROM %s
			ORDER BY csr_source, csr_target`,
			indicesTable, edgesView)); err != nil {
			return fmt.Errorf("create indices %s: %w", indicesTable, err)
		}

		// Build indptr table from the sorted edges
		if err := buildIndptr(destDB, opts.CSRTable, eType, mappingAll); err != nil {
			return fmt.Errorf("build indptr for %s: %w", eType, err)
		}

		// Cleanup temp
		destDB.Exec(fmt.Sprintf("DROP TABLE IF EXISTS %s", edgesView))
	}

	// 4. Metadata table
	totalNodes, _ := countRows(destDB, mappingAll)
	totalEdges := int64(0)
	for _, eType := range edgeTypes {
		n, _ := countRows(destDB, fmt.Sprintf("%s_indices_%s", opts.CSRTable, eType))
		totalEdges += n
	}

	directedInt := 0
	if opts.Directed {
		directedInt = 1
	}
	if _, err := destDB.Exec(fmt.Sprintf(`CREATE OR REPLACE TABLE %s_metadata (
		key VARCHAR, value VARCHAR
	)`, opts.CSRTable)); err != nil {
		return fmt.Errorf("create metadata: %w", err)
	}
	metaInsert := fmt.Sprintf(`INSERT INTO %s_metadata VALUES
		('n_nodes', '%d'),
		('n_edges', '%d'),
		('directed', '%d'),
		('embedding_dim', '%d'),
		('csr_table', '%s')`,
		opts.CSRTable, totalNodes, totalEdges, directedInt, opts.EmbeddingDim, opts.CSRTable)
	if _, err := destDB.Exec(metaInsert); err != nil {
		return fmt.Errorf("insert metadata: %w", err)
	}

	// 5. Optional schema.cypher
	if opts.SchemaOut != "" {
		if err := writeSchema(opts, edgeTypes); err != nil {
			return fmt.Errorf("write schema: %w", err)
		}
	}

	fmt.Printf("  conversion complete: %d nodes, %d edges\n", totalNodes, totalEdges)
	return nil
}

// FromIcebug converts an icebug CSR format database to DuckPGQ format.
func FromIcebug(opts Options) error {
	if opts.CSRTable == "" {
		opts.CSRTable = "predicato"
	}
	floatN := fmt.Sprintf("FLOAT[%d]", opts.EmbeddingDim)

	destDB, err := openDuckDB(opts.Dest)
	if err != nil {
		return fmt.Errorf("open dest: %w", err)
	}
	defer destDB.Close()

	// Attach source
	if _, err := destDB.Exec(fmt.Sprintf(`ATTACH '%s' AS src (READ_ONLY)`, opts.Source)); err != nil {
		return fmt.Errorf("attach source: %w", err)
	}

	// 1. Create DuckPGQ entities table
	if _, err := destDB.Exec(fmt.Sprintf(`CREATE TABLE IF NOT EXISTS entities (
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
	)`, floatN, floatN)); err != nil {
		return fmt.Errorf("create entities: %w", err)
	}

	// 2. Create DuckPGQ edges table
	if _, err := destDB.Exec(fmt.Sprintf(`CREATE TABLE IF NOT EXISTS edges (
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
	)`, floatN, floatN)); err != nil {
		return fmt.Errorf("create edges: %w", err)
	}

	// 3. Import nodes from each nodes_* table
	nodesTables, err := discoverTables(destDB, "src", "nodes_")
	if err != nil {
		return fmt.Errorf("discover node tables: %w", err)
	}

	for _, nt := range nodesTables {
		typ := strings.TrimPrefix(nt, "nodes_")
		cols, err := getColumns(destDB, "src", nt)
		if err != nil {
			return fmt.Errorf("get columns for %s: %w", nt, err)
		}

		// Build INSERT selecting available columns, defaulting missing ones
		insertCols := buildEntityInsert(cols, typ, floatN, nt)
		if _, err := destDB.Exec(insertCols); err != nil {
			return fmt.Errorf("import %s: %w", nt, err)
		}
		n, _ := countRows(destDB, fmt.Sprintf("src.%s", nt))
		fmt.Printf("  imported %d nodes from %s (type=%s)\n", n, nt, typ)
	}

	// 4. Reconstruct edges from CSR
	// Discover mapping and indices tables
	mappingTables, err := discoverTables(destDB, "src", opts.CSRTable+"_mapping_")
	if err != nil {
		return fmt.Errorf("discover mapping tables: %w", err)
	}
	indicesTables, err := discoverTables(destDB, "src", opts.CSRTable+"_indices_")
	if err != nil {
		return fmt.Errorf("discover indices tables: %w", err)
	}

	// Find the unified mapping (mapping_all) or fall back to per-type
	unifiedMapping := ""
	for _, mt := range mappingTables {
		if strings.HasSuffix(mt, "_all") {
			unifiedMapping = mt
			break
		}
	}

	for _, it := range indicesTables {
		eType := strings.TrimPrefix(it, opts.CSRTable+"_indices_")
		mapping := unifiedMapping
		if mapping == "" {
			// Try per-type mapping
			for _, mt := range mappingTables {
				if strings.HasSuffix(mt, "_"+eType) {
					mapping = mt
					break
				}
			}
		}
		if mapping == "" {
			fmt.Printf("  warning: no mapping table for edge type %s, skipping\n", eType)
			continue
		}

		// Join indices with mapping to reconstruct source_id/target_id
		cols, err := getColumns(destDB, "src", it)
		if err != nil {
			return fmt.Errorf("get columns for %s: %w", it, err)
		}

		edgeInsert := buildEdgeInsert(cols, eType, floatN, it, mapping)
		if _, err := destDB.Exec(edgeInsert); err != nil {
			return fmt.Errorf("import edges from %s: %w", it, err)
		}
		n, _ := countRows(destDB, fmt.Sprintf("src.%s", it))
		fmt.Printf("  imported %d edges from %s (type=%s)\n", n, it, eType)
	}

	// 5. Create property graph DDL (non-fatal)
	pgStmt := `CREATE OR REPLACE PROPERTY GRAPH knowledge_graph
		VERTEX TABLES (entities LABEL entity)
		EDGE TABLES (
			edges SOURCE KEY (source_id, group_id) REFERENCES entities (uuid, group_id)
			      DESTINATION KEY (target_id, group_id) REFERENCES entities (uuid, group_id)
			      LABEL relates_to
		)`
	_, _ = destDB.Exec("INSTALL duckpgq FROM community")
	_, _ = destDB.Exec("LOAD duckpgq")
	_, _ = destDB.Exec(pgStmt)

	// 6. Create indices (non-fatal)
	createIndices(destDB)

	entityCount, _ := countRows(destDB, "entities")
	edgeCount, _ := countRows(destDB, "edges")
	fmt.Printf("  conversion complete: %d entities, %d edges\n", entityCount, edgeCount)
	return nil
}

// --- helpers ---

func openDuckDB(path string) (*sql.DB, error) {
	db, err := sql.Open("duckdb", path)
	if err != nil {
		return nil, err
	}
	// Increase memory limit for large graphs
	db.Exec("SET memory_limit='4GB'")
	return db, nil
}

func getDistinctEdgeTypes(db *sql.DB, groupFilter string) ([]string, error) {
	where := "WHERE 1=1" + groupFilter
	rows, err := db.Query(fmt.Sprintf("SELECT DISTINCT type FROM src.edges %s ORDER BY type", where))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var types []string
	for rows.Next() {
		var t string
		if err := rows.Scan(&t); err != nil {
			return nil, err
		}
		types = append(types, t)
	}
	return types, rows.Err()
}

func buildIndptr(db *sql.DB, prefix, eType, mappingAll string) error {
	indptrTable := fmt.Sprintf("%s_indptr_%s", prefix, eType)
	indicesTable := fmt.Sprintf("%s_indices_%s", prefix, eType)

	// Get total number of nodes
	var nNodes int64
	if err := db.QueryRow(fmt.Sprintf("SELECT COUNT(*) FROM %s", mappingAll)).Scan(&nNodes); err != nil {
		return err
	}

	// Build degree counts per source node
	if _, err := db.Exec(fmt.Sprintf(`CREATE OR REPLACE TABLE %s AS
		WITH degrees AS (
			SELECT m.csr_index, COALESCE(cnt, 0) AS degree
			FROM %s m
			LEFT JOIN (
				SELECT csr_source, COUNT(*) AS cnt
				FROM %s
				GROUP BY csr_source
			) e ON m.csr_index = e.csr_source
		),
		cumsum AS (
			SELECT csr_index,
				SUM(degree) OVER (ORDER BY csr_index ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS indptr
			FROM degrees
		)
		SELECT csr_index, indptr FROM (
			SELECT -1 AS csr_index, 0::BIGINT AS indptr
			UNION ALL
			SELECT csr_index, indptr FROM cumsum
		)
		ORDER BY csr_index`,
		indptrTable, mappingAll, indicesTable)); err != nil {
		return err
	}
	return nil
}

func countRows(db *sql.DB, table string) (int64, error) {
	var n int64
	err := db.QueryRow(fmt.Sprintf("SELECT COUNT(*) FROM %s", table)).Scan(&n)
	return n, err
}

func discoverTables(db *sql.DB, catalog, prefix string) ([]string, error) {
	rows, err := db.Query(fmt.Sprintf(
		"SELECT table_name FROM duckdb_tables() WHERE database_name = '%s' AND table_name LIKE '%s%%' ORDER BY table_name",
		catalog, prefix))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var tables []string
	for rows.Next() {
		var t string
		if err := rows.Scan(&t); err != nil {
			return nil, err
		}
		tables = append(tables, t)
	}
	return tables, rows.Err()
}

func getColumns(db *sql.DB, catalog, table string) ([]string, error) {
	rows, err := db.Query(fmt.Sprintf(
		"SELECT column_name FROM duckdb_columns() WHERE database_name = '%s' AND table_name = '%s' ORDER BY column_index",
		catalog, table))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var cols []string
	for rows.Next() {
		var c string
		if err := rows.Scan(&c); err != nil {
			return nil, err
		}
		cols = append(cols, c)
	}
	return cols, rows.Err()
}

func hasCol(cols []string, name string) bool {
	for _, c := range cols {
		if c == name {
			return true
		}
	}
	return false
}

func buildEntityInsert(cols []string, typ, floatN, srcTable string) string {
	// Map icebug node columns to DuckPGQ entity columns
	selects := []string{
		colOrDefault(cols, "uuid", "''"),
		colOrDefault(cols, "group_id", "''"),
		fmt.Sprintf("'%s' AS type", typ),
		colOrDefault(cols, "name", "''"),
		colOrDefault(cols, "content", "''"),
		colOrDefault(cols, "summary", "''"),
		colOrDefault(cols, "entity_type", "''"),
		colOrDefault(cols, "episode_type", "''"),
		castEmbedding(cols, "embedding", floatN),
		castEmbedding(cols, "name_embedding", floatN),
		colOrDefault(cols, "metadata", "'{}'"),
		colOrDefault(cols, "created_at", "CURRENT_TIMESTAMP"),
		colOrDefault(cols, "updated_at", "CURRENT_TIMESTAMP"),
		colOrDefault(cols, "valid_from", "CURRENT_TIMESTAMP"),
		colOrDefault(cols, "valid_to", "NULL"),
		colOrDefault(cols, "source_ids", "'[]'"),
		colOrDefault(cols, "entity_edges", "'[]'"),
		colOrDefault(cols, "level", "0"),
	}
	return fmt.Sprintf(`INSERT INTO entities
		SELECT %s FROM src.%s`,
		strings.Join(selects, ", "), srcTable)
}

func buildEdgeInsert(cols []string, eType, floatN, indicesTable, mappingTable string) string {
	// Map CSR indices back to DuckPGQ edge columns
	selects := []string{
		colOrDefaultPrefixed(cols, "i", "uuid", "''"),
		"sm.group_id",
		"sm.uuid AS source_id",
		"tm.uuid AS target_id",
		fmt.Sprintf("'%s' AS type", eType),
		colOrDefaultPrefixed(cols, "i", "name", "''"),
		colOrDefaultPrefixed(cols, "i", "fact", "''"),
		castEmbeddingPrefixed(cols, "i", "fact_embedding", floatN),
		castEmbeddingPrefixed(cols, "i", "embedding", floatN),
		colOrDefaultPrefixed(cols, "i", "episodes", "'[]'"),
		colOrDefaultPrefixed(cols, "i", "attributes", "'{}'"),
		colOrDefaultPrefixed(cols, "i", "created_at", "CURRENT_TIMESTAMP"),
		colOrDefaultPrefixed(cols, "i", "updated_at", "CURRENT_TIMESTAMP"),
		colOrDefaultPrefixed(cols, "i", "valid_from", "CURRENT_TIMESTAMP"),
		colOrDefaultPrefixed(cols, "i", "valid_to", "NULL"),
		colOrDefaultPrefixed(cols, "i", "expired_at", "NULL"),
		"NULL AS valid_at",
		"NULL AS invalid_at",
		colOrDefaultPrefixed(cols, "i", "source_ids", "'[]'"),
		colOrDefaultPrefixed(cols, "i", "strength", "0.0"),
	}

	// We need to get group_id from the source mapping. The unified mapping
	// doesn't have group_id, so we join through the nodes tables.
	// For simplicity, join mapping uuid back to find group_id from any nodes table.
	return fmt.Sprintf(`INSERT INTO edges
		SELECT %s
		FROM src.%s i
		JOIN src.%s sm ON i.csr_source = sm.csr_index
		JOIN src.%s tm ON i.csr_target = tm.csr_index`,
		strings.Join(selects, ", "), indicesTable, mappingTable, mappingTable)
}

func colOrDefault(cols []string, name, def string) string {
	if hasCol(cols, name) {
		return name
	}
	return fmt.Sprintf("%s AS %s", def, name)
}

func colOrDefaultPrefixed(cols []string, prefix, name, def string) string {
	if hasCol(cols, name) {
		return fmt.Sprintf("%s.%s", prefix, name)
	}
	return fmt.Sprintf("%s AS %s", def, name)
}

func castEmbedding(cols []string, name, floatN string) string {
	if hasCol(cols, name) {
		return fmt.Sprintf("CAST(%s AS %s) AS %s", name, floatN, name)
	}
	return fmt.Sprintf("NULL::%s AS %s", floatN, name)
}

func castEmbeddingPrefixed(cols []string, prefix, name, floatN string) string {
	if hasCol(cols, name) {
		return fmt.Sprintf("CAST(%s.%s AS %s) AS %s", prefix, name, floatN, name)
	}
	return fmt.Sprintf("NULL::%s AS %s", floatN, name)
}

func createIndices(db *sql.DB) {
	indices := []string{
		`CREATE INDEX IF NOT EXISTS idx_entities_group ON entities(group_id)`,
		`CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)`,
		`CREATE INDEX IF NOT EXISTS idx_edges_group ON edges(group_id)`,
		`CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)`,
		`CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)`,
		`CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(type)`,
	}
	for _, idx := range indices {
		db.Exec(idx)
	}

	// FTS (non-fatal)
	db.Exec("INSTALL fts")
	db.Exec("LOAD fts")
	db.Exec(`PRAGMA create_fts_index('entities', 'uuid', 'name', 'summary', 'content', overwrite=1)`)
	db.Exec(`PRAGMA create_fts_index('edges', 'uuid', 'name', 'fact', overwrite=1)`)

	// HNSW (non-fatal)
	db.Exec("INSTALL vss")
	db.Exec("LOAD vss")
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_entities_embedding_hnsw ON entities USING HNSW (embedding) WITH (metric = 'cosine')`)
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_edges_embedding_hnsw ON edges USING HNSW (embedding) WITH (metric = 'cosine')`)
}

func writeSchema(opts Options, edgeTypes []string) error {
	var sb strings.Builder
	sb.WriteString("// Auto-generated schema.cypher for icebug format\n\n")

	for _, typ := range entityTypes {
		sb.WriteString(fmt.Sprintf("CREATE NODE TABLE nodes_%s (\n", typ))
		sb.WriteString("  uuid STRING PRIMARY KEY,\n")
		sb.WriteString("  group_id STRING,\n")
		sb.WriteString("  name STRING,\n")
		sb.WriteString("  content STRING,\n")
		sb.WriteString("  summary STRING,\n")
		sb.WriteString("  entity_type STRING,\n")
		sb.WriteString("  episode_type STRING,\n")
		sb.WriteString("  metadata STRING,\n")
		sb.WriteString("  created_at TIMESTAMP,\n")
		sb.WriteString("  updated_at TIMESTAMP,\n")
		sb.WriteString("  valid_from TIMESTAMP,\n")
		sb.WriteString("  valid_to TIMESTAMP,\n")
		sb.WriteString("  source_ids STRING,\n")
		sb.WriteString("  entity_edges STRING,\n")
		sb.WriteString("  level INT64\n")
		sb.WriteString(");\n\n")
	}

	// Sort edge types for deterministic output
	sorted := make([]string, len(edgeTypes))
	copy(sorted, edgeTypes)
	sort.Strings(sorted)

	for _, eType := range sorted {
		sb.WriteString(fmt.Sprintf("CREATE REL TABLE edges_%s (\n", eType))
		sb.WriteString("  FROM nodes_entity TO nodes_entity,\n")
		sb.WriteString("  uuid STRING,\n")
		sb.WriteString("  name STRING,\n")
		sb.WriteString("  fact STRING,\n")
		sb.WriteString("  attributes STRING,\n")
		sb.WriteString("  episodes STRING,\n")
		sb.WriteString("  strength DOUBLE,\n")
		sb.WriteString("  created_at TIMESTAMP,\n")
		sb.WriteString("  updated_at TIMESTAMP,\n")
		sb.WriteString("  valid_from TIMESTAMP,\n")
		sb.WriteString("  valid_to TIMESTAMP,\n")
		sb.WriteString("  expired_at TIMESTAMP,\n")
		sb.WriteString("  source_ids STRING\n")
		sb.WriteString(");\n\n")
	}

	return os.WriteFile(opts.SchemaOut, []byte(sb.String()), 0644)
}
