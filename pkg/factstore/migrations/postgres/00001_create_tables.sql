-- +goose Up
-- +goose StatementBegin

-- Predicato structured store schema for PostgreSQL (with VectorChord/pgvector)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sources (
    id VARCHAR(255) PRIMARY KEY,
    name TEXT,
    content TEXT,
    group_id VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS extracted_nodes (
    id VARCHAR(255) PRIMARY KEY,
    source_id VARCHAR(255) REFERENCES sources(id),
    group_id VARCHAR(255),
    name TEXT,
    normalized_name TEXT,
    type VARCHAR(50),
    description TEXT,
    embedding vector(1024),
    chunk_index INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS node_sources (
    node_id VARCHAR(255) REFERENCES extracted_nodes(id),
    source_id VARCHAR(255) REFERENCES sources(id),
    chunk_index INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (node_id, source_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS telemetry_logs (
    id VARCHAR(36) PRIMARY KEY,
    timestamp TIMESTAMP,
    level VARCHAR(10),
    message TEXT,
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    request_source VARCHAR(255),
    source_file VARCHAR(255),
    line_number INT,
    attributes JSONB
);

-- B-tree indexes
CREATE INDEX IF NOT EXISTS idx_sources_group ON sources(group_id);
CREATE INDEX IF NOT EXISTS idx_nodes_source ON extracted_nodes(source_id);
CREATE INDEX IF NOT EXISTS idx_nodes_group ON extracted_nodes(group_id);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON extracted_nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_dedup ON extracted_nodes(group_id, normalized_name, type);
CREATE INDEX IF NOT EXISTS idx_edges_source ON extracted_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_group ON extracted_edges(group_id);

-- GIN full-text search indexes
CREATE INDEX IF NOT EXISTS idx_nodes_fts ON extracted_nodes
    USING GIN (to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(description, '')));
CREATE INDEX IF NOT EXISTS idx_edges_fts ON extracted_edges
    USING GIN (to_tsvector('english', COALESCE(relation, '') || ' ' || COALESCE(description, '')));
CREATE INDEX IF NOT EXISTS idx_sources_fts ON sources
    USING GIN (to_tsvector('english', COALESCE(name, '') || ' ' || COALESCE(content, '')));

-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DROP TABLE IF EXISTS telemetry_logs;
DROP TABLE IF EXISTS node_sources;
DROP TABLE IF EXISTS extracted_edges;
DROP TABLE IF EXISTS extracted_nodes;
DROP TABLE IF EXISTS sources;
-- Do NOT drop the vector extension; other tables may depend on it.
-- +goose StatementEnd
