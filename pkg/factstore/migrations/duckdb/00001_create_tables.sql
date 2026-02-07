-- +goose Up
-- +goose StatementBegin

-- Predicato structured store schema for DuckDB
-- Embeddings stored as FLOAT arrays; DuckDB supports native array types
-- for efficient columnar storage. Vector similarity can use DuckDB's
-- list_cosine_similarity function.

CREATE TABLE IF NOT EXISTS sources (
    id VARCHAR PRIMARY KEY,
    name VARCHAR,
    content VARCHAR,
    group_id VARCHAR,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS extracted_nodes (
    id VARCHAR PRIMARY KEY,
    source_id VARCHAR REFERENCES sources(id),
    group_id VARCHAR,
    name VARCHAR,
    normalized_name VARCHAR,
    type VARCHAR,
    description VARCHAR,
    embedding FLOAT[1024],
    chunk_index INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS extracted_edges (
    id VARCHAR PRIMARY KEY,
    source_id VARCHAR REFERENCES sources(id),
    group_id VARCHAR,
    source_node_name VARCHAR,
    source_node_type VARCHAR,
    target_node_name VARCHAR,
    target_node_type VARCHAR,
    relation VARCHAR,
    description VARCHAR,
    embedding FLOAT[1024],
    weight FLOAT,
    chunk_index INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS node_sources (
    node_id VARCHAR REFERENCES extracted_nodes(id),
    source_id VARCHAR REFERENCES sources(id),
    chunk_index INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (node_id, source_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS telemetry_logs (
    id VARCHAR PRIMARY KEY,
    timestamp TIMESTAMP,
    level VARCHAR,
    message VARCHAR,
    user_id VARCHAR,
    session_id VARCHAR,
    request_source VARCHAR,
    source_file VARCHAR,
    line_number INTEGER,
    attributes JSON
);

CREATE INDEX IF NOT EXISTS idx_sources_group ON sources(group_id);
CREATE INDEX IF NOT EXISTS idx_nodes_source ON extracted_nodes(source_id);
CREATE INDEX IF NOT EXISTS idx_nodes_group ON extracted_nodes(group_id);
CREATE INDEX IF NOT EXISTS idx_nodes_type ON extracted_nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_dedup ON extracted_nodes(group_id, normalized_name, type);
CREATE INDEX IF NOT EXISTS idx_edges_source ON extracted_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_group ON extracted_edges(group_id);

-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DROP TABLE IF EXISTS telemetry_logs;
DROP TABLE IF EXISTS node_sources;
DROP TABLE IF EXISTS extracted_edges;
DROP TABLE IF EXISTS extracted_nodes;
DROP TABLE IF EXISTS sources;
-- +goose StatementEnd
