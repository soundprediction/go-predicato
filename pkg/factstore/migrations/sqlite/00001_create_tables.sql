-- +goose Up
-- +goose StatementBegin

-- Predicato structured store schema for SQLite
-- Embeddings stored as TEXT (JSON-encoded arrays); vector similarity
-- is computed in-memory.

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    name TEXT,
    content TEXT,
    group_id TEXT,
    metadata TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS extracted_nodes (
    id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES sources(id),
    group_id TEXT,
    name TEXT,
    normalized_name TEXT,
    type TEXT,
    description TEXT,
    embedding TEXT,
    chunk_index INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS extracted_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES sources(id),
    group_id TEXT,
    source_node_name TEXT,
    source_node_type TEXT,
    target_node_name TEXT,
    target_node_type TEXT,
    relation TEXT,
    description TEXT,
    embedding TEXT,
    weight REAL,
    chunk_index INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS node_sources (
    node_id TEXT REFERENCES extracted_nodes(id),
    source_id TEXT REFERENCES sources(id),
    chunk_index INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (node_id, source_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS telemetry_logs (
    id TEXT PRIMARY KEY,
    timestamp TEXT,
    level TEXT,
    message TEXT,
    user_id TEXT,
    session_id TEXT,
    request_source TEXT,
    source_file TEXT,
    line_number INTEGER,
    attributes TEXT
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
