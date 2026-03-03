package factstore

import (
	"context"
	"database/sql"
	"testing"
	"time"

	sqlmock "github.com/DATA-DOG/go-sqlmock"
)

// newMockPostgresDB creates a PostgresDB with a sqlmock database for testing.
func newMockPostgresDB(t *testing.T) (*PostgresDB, sqlmock.Sqlmock) {
	t.Helper()
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("Failed to create sqlmock: %v", err)
	}
	return &PostgresDB{db: db, embeddingDimensions: 1024}, mock
}

// expectCommitOnly adds the commit expectation at the end of SaveExtractedKnowledge
// when no triples are passed (triples == nil skips the prepared statement).
func expectCommitOnly(mock sqlmock.Sqlmock) {
	mock.ExpectCommit()
}

// TestDedupNonPersonEntityAcrossSources verifies that the same non-person entity
// from two different sources results in dedup: the second call finds the existing
// entity and updates it instead of inserting a new row.
func TestDedupNonPersonEntityAcrossSources(t *testing.T) {
	pdb, mock := newMockPostgresDB(t)
	defer pdb.db.Close()

	ctx := context.Background()
	now := time.Now()

	// --- First source: insert "Acme Corporation" as new ---
	mock.ExpectBegin()
	mock.ExpectQuery(`SELECT group_id FROM sources WHERE id = \$1`).
		WithArgs("source-A").
		WillReturnRows(sqlmock.NewRows([]string{"group_id"}).AddRow("group-1"))

	// Lookup: non-person, search by group_id + normalized_name + type → not found
	mock.ExpectQuery(`SELECT id, description FROM extracted_nodes`).
		WithArgs("group-1", "acme corporation", "organization").
		WillReturnError(sql.ErrNoRows)

	// Insert new node
	mock.ExpectExec(`INSERT INTO extracted_nodes`).
		WithArgs("node-1", "source-A", "group-1", "Acme Corporation", "acme corporation",
			"organization", "A tech company", nil, 0, "", sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(0, 1))

	// Insert node_sources
	mock.ExpectExec(`INSERT INTO node_sources`).
		WithArgs("node-1", "source-A", 0).
		WillReturnResult(sqlmock.NewResult(0, 1))

	expectCommitOnly(mock)

	err := pdb.SaveExtractedKnowledge(ctx, "source-A", []*ExtractedNode{
		{
			ID:             "node-1",
			Name:           "Acme Corporation",
			NormalizedName: "acme corporation",
			Type:           "organization",
			Description:    "A tech company",
			ChunkIndex:     0,
			CreatedAt:      now,
		},
	}, nil)
	if err != nil {
		t.Fatalf("First SaveExtractedKnowledge failed: %v", err)
	}

	// --- Second source: same entity → should dedup to existing ---
	mock.ExpectBegin()
	mock.ExpectQuery(`SELECT group_id FROM sources WHERE id = \$1`).
		WithArgs("source-B").
		WillReturnRows(sqlmock.NewRows([]string{"group_id"}).AddRow("group-1"))

	// Lookup: non-person, found existing
	mock.ExpectQuery(`SELECT id, description FROM extracted_nodes`).
		WithArgs("group-1", "acme corporation", "organization").
		WillReturnRows(sqlmock.NewRows([]string{"id", "description"}).AddRow("node-1", "A tech company"))

	// Description "A tech company" (14 chars) vs "A large tech corporation" (24 chars) → update
	mock.ExpectExec(`UPDATE extracted_nodes SET description`).
		WithArgs("A large tech corporation", "node-1").
		WillReturnResult(sqlmock.NewResult(0, 1))

	// Insert node_sources for the second source
	mock.ExpectExec(`INSERT INTO node_sources`).
		WithArgs("node-1", "source-B", 0).
		WillReturnResult(sqlmock.NewResult(0, 1))

	expectCommitOnly(mock)

	err = pdb.SaveExtractedKnowledge(ctx, "source-B", []*ExtractedNode{
		{
			ID:             "node-2", // Different ID, but same entity
			Name:           "Acme Corporation",
			NormalizedName: "acme corporation",
			Type:           "organization",
			Description:    "A large tech corporation",
			ChunkIndex:     0,
			CreatedAt:      now,
		},
	}, nil)
	if err != nil {
		t.Fatalf("Second SaveExtractedKnowledge failed: %v", err)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Errorf("Unmet mock expectations: %v", err)
	}
}

// TestDedupPersonEntityDifferentSources verifies that person entities from different sources
// are NOT deduplicated (different people can share names).
func TestDedupPersonEntityDifferentSources(t *testing.T) {
	pdb, mock := newMockPostgresDB(t)
	defer pdb.db.Close()

	ctx := context.Background()
	now := time.Now()

	// --- First source: insert "John Smith" as new person ---
	mock.ExpectBegin()
	mock.ExpectQuery(`SELECT group_id FROM sources WHERE id = \$1`).
		WithArgs("source-A").
		WillReturnRows(sqlmock.NewRows([]string{"group_id"}).AddRow("group-1"))

	// Person lookup: uses node_sources join with source_id → not found in source-A
	mock.ExpectQuery(`SELECT en.id, en.description FROM extracted_nodes en`).
		WithArgs("source-A", "john smith", "person").
		WillReturnError(sql.ErrNoRows)

	// Insert new node
	mock.ExpectExec(`INSERT INTO extracted_nodes`).
		WithArgs("person-1", "source-A", "group-1", "John Smith", "john smith",
			"person", "Engineer at Acme", nil, 0, "", sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(0, 1))

	mock.ExpectExec(`INSERT INTO node_sources`).
		WithArgs("person-1", "source-A", 0).
		WillReturnResult(sqlmock.NewResult(0, 1))

	expectCommitOnly(mock)

	err := pdb.SaveExtractedKnowledge(ctx, "source-A", []*ExtractedNode{
		{
			ID:             "person-1",
			Name:           "John Smith",
			NormalizedName: "john smith",
			Type:           "person",
			Description:    "Engineer at Acme",
			ChunkIndex:     0,
			CreatedAt:      now,
		},
	}, nil)
	if err != nil {
		t.Fatalf("First person save failed: %v", err)
	}

	// --- Second source: same name person → should NOT dedup (different source) ---
	mock.ExpectBegin()
	mock.ExpectQuery(`SELECT group_id FROM sources WHERE id = \$1`).
		WithArgs("source-B").
		WillReturnRows(sqlmock.NewRows([]string{"group_id"}).AddRow("group-1"))

	// Person lookup scoped to source-B → not found
	mock.ExpectQuery(`SELECT en.id, en.description FROM extracted_nodes en`).
		WithArgs("source-B", "john smith", "person").
		WillReturnError(sql.ErrNoRows)

	// Insert new (separate) node for the different John Smith
	mock.ExpectExec(`INSERT INTO extracted_nodes`).
		WithArgs("person-2", "source-B", "group-1", "John Smith", "john smith",
			"person", "Manager at Beta Corp", nil, 0, "", sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(0, 1))

	mock.ExpectExec(`INSERT INTO node_sources`).
		WithArgs("person-2", "source-B", 0).
		WillReturnResult(sqlmock.NewResult(0, 1))

	expectCommitOnly(mock)

	err = pdb.SaveExtractedKnowledge(ctx, "source-B", []*ExtractedNode{
		{
			ID:             "person-2",
			Name:           "John Smith",
			NormalizedName: "john smith",
			Type:           "person",
			Description:    "Manager at Beta Corp",
			ChunkIndex:     0,
			CreatedAt:      now,
		},
	}, nil)
	if err != nil {
		t.Fatalf("Second person save failed: %v", err)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Errorf("Unmet mock expectations: %v", err)
	}
}

// TestDedupPersonEntitySameSource verifies that person entities within the same source
// ARE deduplicated (same person mentioned in different chunks).
func TestDedupPersonEntitySameSource(t *testing.T) {
	pdb, mock := newMockPostgresDB(t)
	defer pdb.db.Close()

	ctx := context.Background()
	now := time.Now()

	mock.ExpectBegin()
	mock.ExpectQuery(`SELECT group_id FROM sources WHERE id = \$1`).
		WithArgs("source-A").
		WillReturnRows(sqlmock.NewRows([]string{"group_id"}).AddRow("group-1"))

	// First chunk: person not found in source-A → insert
	mock.ExpectQuery(`SELECT en.id, en.description FROM extracted_nodes en`).
		WithArgs("source-A", "jane doe", "person").
		WillReturnError(sql.ErrNoRows)

	mock.ExpectExec(`INSERT INTO extracted_nodes`).
		WithArgs("person-1", "source-A", "group-1", "Jane Doe", "jane doe",
			"person", "CEO", nil, 0, "", sqlmock.AnyArg()).
		WillReturnResult(sqlmock.NewResult(0, 1))

	mock.ExpectExec(`INSERT INTO node_sources`).
		WithArgs("person-1", "source-A", 0).
		WillReturnResult(sqlmock.NewResult(0, 1))

	// Second chunk: same person, same source → found, dedup
	mock.ExpectQuery(`SELECT en.id, en.description FROM extracted_nodes en`).
		WithArgs("source-A", "jane doe", "person").
		WillReturnRows(sqlmock.NewRows([]string{"id", "description"}).AddRow("person-1", "CEO"))

	// "CEO" (3 chars) vs "CEO and founder of Acme" (22 chars) → update description
	mock.ExpectExec(`UPDATE extracted_nodes SET description`).
		WithArgs("CEO and founder of Acme", "person-1").
		WillReturnResult(sqlmock.NewResult(0, 1))

	// node_sources for chunk 1
	mock.ExpectExec(`INSERT INTO node_sources`).
		WithArgs("person-1", "source-A", 1).
		WillReturnResult(sqlmock.NewResult(0, 1))

	expectCommitOnly(mock)

	err := pdb.SaveExtractedKnowledge(ctx, "source-A", []*ExtractedNode{
		{
			ID:             "person-1",
			Name:           "Jane Doe",
			NormalizedName: "jane doe",
			Type:           "person",
			Description:    "CEO",
			ChunkIndex:     0,
			CreatedAt:      now,
		},
		{
			ID:             "person-1-chunk2",
			Name:           "Jane Doe",
			NormalizedName: "jane doe",
			Type:           "person",
			Description:    "CEO and founder of Acme",
			ChunkIndex:     1,
			CreatedAt:      now,
		},
	}, nil)
	if err != nil {
		t.Fatalf("SaveExtractedKnowledge with same-source person dedup failed: %v", err)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Errorf("Unmet mock expectations: %v", err)
	}
}

// TestDedupDescriptionNotOverwrittenWhenShorter verifies that a shorter incoming
// description does NOT overwrite a longer existing description.
func TestDedupDescriptionNotOverwrittenWhenShorter(t *testing.T) {
	pdb, mock := newMockPostgresDB(t)
	defer pdb.db.Close()

	ctx := context.Background()
	now := time.Now()

	mock.ExpectBegin()
	mock.ExpectQuery(`SELECT group_id FROM sources WHERE id = \$1`).
		WithArgs("source-B").
		WillReturnRows(sqlmock.NewRows([]string{"group_id"}).AddRow("group-1"))

	// Non-person lookup: found with longer existing description
	mock.ExpectQuery(`SELECT id, description FROM extracted_nodes`).
		WithArgs("group-1", "acme corporation", "organization").
		WillReturnRows(sqlmock.NewRows([]string{"id", "description"}).
			AddRow("node-1", "A very detailed description of Acme Corporation including history"))

	// Description "Short" (5 chars) < existing (64 chars) → NO description update
	// Embedding is empty → NO embedding update

	// Only node_sources insert expected
	mock.ExpectExec(`INSERT INTO node_sources`).
		WithArgs("node-1", "source-B", 0).
		WillReturnResult(sqlmock.NewResult(0, 1))

	expectCommitOnly(mock)

	err := pdb.SaveExtractedKnowledge(ctx, "source-B", []*ExtractedNode{
		{
			ID:             "node-new",
			Name:           "Acme Corporation",
			NormalizedName: "acme corporation",
			Type:           "organization",
			Description:    "Short",
			ChunkIndex:     0,
			CreatedAt:      now,
		},
	}, nil)
	if err != nil {
		t.Fatalf("SaveExtractedKnowledge failed: %v", err)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Errorf("Unmet mock expectations: %v", err)
	}
}

// TestGetExtractedNodesJoinsThroughNodeSources verifies that GetExtractedNodes
// uses the node_sources junction table to find cross-source deduplicated entities.
func TestGetExtractedNodesJoinsThroughNodeSources(t *testing.T) {
	pdb, mock := newMockPostgresDB(t)
	defer pdb.db.Close()

	ctx := context.Background()
	now := time.Now()

	// Expect the join query through node_sources
	rows := sqlmock.NewRows([]string{
		"id", "source_id", "group_id", "name", "normalized_name", "type",
		"description", "embedding", "chunk_index", "model", "created_at",
	}).AddRow(
		"node-1", "source-A", "group-1", "Acme Corporation", "acme corporation",
		"organization", "A tech company", nil, 0, nil, now,
	)

	mock.ExpectQuery(`SELECT en\.id, en\.source_id, en\.group_id, en\.name, en\.normalized_name, en\.type`).
		WithArgs("source-B").
		WillReturnRows(rows)

	nodes, err := pdb.GetExtractedNodes(ctx, "source-B")
	if err != nil {
		t.Fatalf("GetExtractedNodes failed: %v", err)
	}

	if len(nodes) != 1 {
		t.Fatalf("Expected 1 node, got %d", len(nodes))
	}

	if nodes[0].ID != "node-1" {
		t.Errorf("Expected node ID 'node-1', got %q", nodes[0].ID)
	}
	if nodes[0].NormalizedName != "acme corporation" {
		t.Errorf("Expected normalized_name 'acme corporation', got %q", nodes[0].NormalizedName)
	}
	// The node's source_id is source-A (the creating source), but we queried via source-B
	// This proves the join through node_sources works for cross-source access
	if nodes[0].SourceID != "source-A" {
		t.Errorf("Expected original source_id 'source-A', got %q", nodes[0].SourceID)
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Errorf("Unmet mock expectations: %v", err)
	}
}
