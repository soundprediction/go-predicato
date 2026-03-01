//go:build system_duckpgq

package graphconv

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "source.duckdb")
	icebugPath := filepath.Join(dir, "icebug.duckdb")
	roundtripPath := filepath.Join(dir, "roundtrip.duckdb")
	schemaPath := filepath.Join(dir, "schema.cypher")

	embDim := 4 // small for testing

	// 1. Create source DuckPGQ database with test data
	createTestDuckPGQ(t, srcPath, embDim)

	// 2. Convert to icebug
	err := ToIcebug(Options{
		Source:       srcPath,
		Dest:         icebugPath,
		EmbeddingDim: embDim,
		CSRTable:     "test",
		Directed:     true,
		SchemaOut:    schemaPath,
	})
	if err != nil {
		t.Fatalf("ToIcebug: %v", err)
	}

	// 3. Verify icebug tables
	verifyIcebugTables(t, icebugPath)

	// 4. Verify schema.cypher was written
	if _, err := os.Stat(schemaPath); err != nil {
		t.Fatalf("schema.cypher not written: %v", err)
	}

	// 5. Convert back to DuckPGQ
	err = FromIcebug(Options{
		Source:       icebugPath,
		Dest:         roundtripPath,
		EmbeddingDim: embDim,
		CSRTable:     "test",
		Directed:     true,
	})
	if err != nil {
		t.Fatalf("FromIcebug: %v", err)
	}

	// 6. Verify roundtrip
	verifyRoundtrip(t, srcPath, roundtripPath, embDim)
}

func createTestDuckPGQ(t *testing.T, path string, embDim int) {
	t.Helper()
	floatN := fmt.Sprintf("FLOAT[%d]", embDim)

	db, err := sql.Open("duckdb", path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()

	// Create tables matching DuckPGQ schema
	_, err = db.Exec(fmt.Sprintf(`CREATE TABLE entities (
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
	)`, floatN, floatN))
	if err != nil {
		t.Fatalf("create entities: %v", err)
	}

	_, err = db.Exec(fmt.Sprintf(`CREATE TABLE edges (
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
	)`, floatN, floatN))
	if err != nil {
		t.Fatalf("create edges: %v", err)
	}

	// Insert test entities
	entities := []struct {
		uuid, groupID, typ, name, content string
	}{
		{"e1", "g1", "entity", "Alice", "Person Alice"},
		{"e2", "g1", "entity", "Bob", "Person Bob"},
		{"e3", "g1", "entity", "Carol", "Person Carol"},
		{"ep1", "g1", "episodic", "Episode1", "First episode"},
		{"c1", "g1", "community", "Community1", "A community"},
	}
	for _, e := range entities {
		_, err := db.Exec(fmt.Sprintf(
			`INSERT INTO entities (uuid, group_id, type, name, content, embedding, name_embedding)
			VALUES ('%s', '%s', '%s', '%s', '%s', [1.0, 2.0, 3.0, 4.0]::%s, [0.1, 0.2, 0.3, 0.4]::%s)`,
			e.uuid, e.groupID, e.typ, e.name, e.content, floatN, floatN))
		if err != nil {
			t.Fatalf("insert entity %s: %v", e.uuid, err)
		}
	}

	// Insert test edges
	edges := []struct {
		uuid, source, target, typ, name, fact string
		strength                              float64
	}{
		{"edge1", "e1", "e2", "entity", "knows", "Alice knows Bob", 0.9},
		{"edge2", "e2", "e3", "entity", "knows", "Bob knows Carol", 0.8},
		{"edge3", "e1", "e3", "entity", "works_with", "Alice works with Carol", 0.7},
		{"edge4", "ep1", "e1", "episodic", "mentions", "Episode mentions Alice", 0.5},
	}
	for _, e := range edges {
		_, err := db.Exec(fmt.Sprintf(
			`INSERT INTO edges (uuid, group_id, source_id, target_id, type, name, fact, strength, embedding, fact_embedding)
			VALUES ('%s', 'g1', '%s', '%s', '%s', '%s', '%s', %f, [1.0, 2.0, 3.0, 4.0]::%s, [0.5, 0.6, 0.7, 0.8]::%s)`,
			e.uuid, e.source, e.target, e.typ, e.name, e.fact, e.strength, floatN, floatN))
		if err != nil {
			t.Fatalf("insert edge %s: %v", e.uuid, err)
		}
	}
}

func verifyIcebugTables(t *testing.T, path string) {
	t.Helper()
	db, err := sql.Open("duckdb", path+"?access_mode=READ_ONLY")
	if err != nil {
		t.Fatalf("open icebug: %v", err)
	}
	defer db.Close()

	// Check node tables exist and have correct counts
	checks := map[string]int64{
		"nodes_entity":    3,
		"nodes_episodic":  1,
		"nodes_community": 1,
		"nodes_rule":      0,
	}
	for table, expected := range checks {
		var count int64
		err := db.QueryRow(fmt.Sprintf("SELECT COUNT(*) FROM %s", table)).Scan(&count)
		if err != nil {
			t.Errorf("count %s: %v", table, err)
			continue
		}
		if count != expected {
			t.Errorf("%s: got %d rows, want %d", table, count, expected)
		}
	}

	// Check mapping tables
	var mappingCount int64
	err = db.QueryRow("SELECT COUNT(*) FROM test_mapping_all").Scan(&mappingCount)
	if err != nil {
		t.Fatalf("count mapping_all: %v", err)
	}
	if mappingCount != 5 {
		t.Errorf("mapping_all: got %d, want 5", mappingCount)
	}

	// Check indices tables exist
	for _, eType := range []string{"entity", "episodic"} {
		var n int64
		err := db.QueryRow(fmt.Sprintf("SELECT COUNT(*) FROM test_indices_%s", eType)).Scan(&n)
		if err != nil {
			t.Errorf("count indices_%s: %v", eType, err)
		}
	}

	// Check indptr tables exist
	for _, eType := range []string{"entity", "episodic"} {
		var n int64
		err := db.QueryRow(fmt.Sprintf("SELECT COUNT(*) FROM test_indptr_%s", eType)).Scan(&n)
		if err != nil {
			t.Errorf("count indptr_%s: %v", eType, err)
		}
	}

	// Check metadata
	var val string
	err = db.QueryRow("SELECT value FROM test_metadata WHERE key = 'n_nodes'").Scan(&val)
	if err != nil {
		t.Errorf("metadata n_nodes: %v", err)
	} else if val != "5" {
		t.Errorf("metadata n_nodes: got %s, want 5", val)
	}
}

func verifyRoundtrip(t *testing.T, srcPath, rtPath string, embDim int) {
	t.Helper()

	srcDB, err := sql.Open("duckdb", srcPath+"?access_mode=READ_ONLY")
	if err != nil {
		t.Fatalf("open source: %v", err)
	}
	defer srcDB.Close()

	rtDB, err := sql.Open("duckdb", rtPath+"?access_mode=READ_ONLY")
	if err != nil {
		t.Fatalf("open roundtrip: %v", err)
	}
	defer rtDB.Close()

	// Compare entity counts
	var srcEntities, rtEntities int64
	srcDB.QueryRow("SELECT COUNT(*) FROM entities").Scan(&srcEntities)
	rtDB.QueryRow("SELECT COUNT(*) FROM entities").Scan(&rtEntities)
	if srcEntities != rtEntities {
		t.Errorf("entity count: source=%d, roundtrip=%d", srcEntities, rtEntities)
	}

	// Compare edge counts
	var srcEdges, rtEdges int64
	srcDB.QueryRow("SELECT COUNT(*) FROM edges").Scan(&srcEdges)
	rtDB.QueryRow("SELECT COUNT(*) FROM edges").Scan(&rtEdges)
	if srcEdges != rtEdges {
		t.Errorf("edge count: source=%d, roundtrip=%d", srcEdges, rtEdges)
	}

	// Spot check: verify Alice exists with correct name
	var name string
	err = rtDB.QueryRow("SELECT name FROM entities WHERE uuid = 'e1'").Scan(&name)
	if err != nil {
		t.Errorf("lookup Alice: %v", err)
	} else if name != "Alice" {
		t.Errorf("Alice name: got %q, want %q", name, "Alice")
	}

	// Spot check: verify edge Alice->Bob exists
	var edgeFact string
	err = rtDB.QueryRow("SELECT fact FROM edges WHERE source_id = 'e1' AND target_id = 'e2'").Scan(&edgeFact)
	if err != nil {
		t.Errorf("lookup Alice->Bob edge: %v", err)
	} else if edgeFact != "Alice knows Bob" {
		t.Errorf("edge fact: got %q, want %q", edgeFact, "Alice knows Bob")
	}

	// Verify types preserved
	var typeCount int64
	rtDB.QueryRow("SELECT COUNT(DISTINCT type) FROM entities").Scan(&typeCount)
	if typeCount < 3 {
		t.Errorf("expected at least 3 distinct entity types, got %d", typeCount)
	}

	t.Logf("roundtrip OK: %d entities, %d edges", rtEntities, rtEdges)
}

func TestToIcebugEmptyDB(t *testing.T) {
	dir := t.TempDir()
	srcPath := filepath.Join(dir, "empty.duckdb")
	destPath := filepath.Join(dir, "empty_icebug.duckdb")

	// Create empty DuckPGQ database
	db, err := sql.Open("duckdb", srcPath)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	db.Exec(`CREATE TABLE entities (
		uuid VARCHAR, group_id VARCHAR, type VARCHAR DEFAULT 'entity',
		name VARCHAR DEFAULT '', content VARCHAR DEFAULT '', summary VARCHAR DEFAULT '',
		entity_type VARCHAR DEFAULT '', episode_type VARCHAR DEFAULT '',
		embedding FLOAT[4], name_embedding FLOAT[4],
		metadata VARCHAR DEFAULT '{}',
		created_at TIMESTAMP, updated_at TIMESTAMP, valid_from TIMESTAMP, valid_to TIMESTAMP,
		source_ids VARCHAR DEFAULT '[]', entity_edges VARCHAR DEFAULT '[]', level INTEGER DEFAULT 0,
		PRIMARY KEY (uuid, group_id)
	)`)
	db.Exec(`CREATE TABLE edges (
		uuid VARCHAR, group_id VARCHAR, source_id VARCHAR DEFAULT '', target_id VARCHAR DEFAULT '',
		type VARCHAR DEFAULT 'entity', name VARCHAR DEFAULT '', fact VARCHAR DEFAULT '',
		fact_embedding FLOAT[4], embedding FLOAT[4],
		episodes VARCHAR DEFAULT '[]', attributes VARCHAR DEFAULT '{}',
		created_at TIMESTAMP, updated_at TIMESTAMP, valid_from TIMESTAMP, valid_to TIMESTAMP,
		expired_at TIMESTAMP, valid_at TIMESTAMP, invalid_at TIMESTAMP,
		source_ids VARCHAR DEFAULT '[]', strength DOUBLE DEFAULT 0.0,
		PRIMARY KEY (uuid, group_id)
	)`)
	db.Close()

	err = ToIcebug(Options{
		Source:       srcPath,
		Dest:         destPath,
		EmbeddingDim: 4,
		Directed:     true,
	})
	if err != nil {
		t.Fatalf("ToIcebug on empty DB: %v", err)
	}
}
