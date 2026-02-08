-- +goose Up
-- +goose StatementBegin

-- Consolidate extracted_edges into extracted_triples.
-- The extracted_triples table is the single unified table for all knowledge triples.

-- Create extracted_triples table with all columns (if not exists from earlier migration)
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
    embedding vector(1024),
    chunk_index INT,
    model VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add new columns if the table already exists from an earlier migration
ALTER TABLE extracted_triples ADD COLUMN IF NOT EXISTS group_id VARCHAR(255);
ALTER TABLE extracted_triples ADD COLUMN IF NOT EXISTS subject_type VARCHAR(50);
ALTER TABLE extracted_triples ADD COLUMN IF NOT EXISTS object_type VARCHAR(50);
ALTER TABLE extracted_triples ADD COLUMN IF NOT EXISTS description TEXT;
ALTER TABLE extracted_triples ADD COLUMN IF NOT EXISTS model VARCHAR(255);

-- Migrate data from extracted_edges to extracted_triples
INSERT INTO extracted_triples (id, source_id, group_id, subject, subject_type, predicate, object, object_type, description, confidence, embedding, chunk_index, model, created_at)
SELECT id, source_id, group_id, source_node_name, source_node_type, relation, target_node_name, target_node_type, description, weight, embedding, chunk_index, model, created_at
FROM extracted_edges
ON CONFLICT (id) DO NOTHING;

-- Drop old table and its indexes
DROP INDEX IF EXISTS idx_edges_source;
DROP INDEX IF EXISTS idx_edges_group;
DROP INDEX IF EXISTS idx_edges_fts;
DROP INDEX IF EXISTS idx_edges_embedding;
DROP TABLE IF EXISTS extracted_edges;

-- Create indexes for the consolidated triples table
CREATE INDEX IF NOT EXISTS idx_triples_source ON extracted_triples(source_id);
CREATE INDEX IF NOT EXISTS idx_triples_group ON extracted_triples(group_id);
CREATE INDEX IF NOT EXISTS idx_triples_fts ON extracted_triples
    USING GIN (to_tsvector('english', COALESCE(subject, '') || ' ' || COALESCE(predicate, '') || ' ' || COALESCE(object, '') || ' ' || COALESCE(description, '')));

-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin

-- Recreate extracted_edges table
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
    embedding vector(1024),
    weight FLOAT,
    chunk_index INT,
    model VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_edges_source ON extracted_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_group ON extracted_edges(group_id);

-- +goose StatementEnd
