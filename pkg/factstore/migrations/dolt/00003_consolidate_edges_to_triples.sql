-- +goose Up
-- +goose StatementBegin

CREATE TABLE IF NOT EXISTS extracted_triples (
    id VARCHAR(255) PRIMARY KEY,
    source_id VARCHAR(255),
    group_id VARCHAR(255),
    subject TEXT,
    subject_type VARCHAR(50),
    predicate TEXT,
    object TEXT,
    object_type VARCHAR(50),
    description TEXT,
    `condition` TEXT,
    temporal TEXT,
    location TEXT,
    certainty TEXT,
    scope TEXT,
    source_attribution TEXT,
    confidence FLOAT,
    embedding JSON,
    chunk_index INT,
    model VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES sources(id)
);

INSERT IGNORE INTO extracted_triples (id, source_id, group_id, subject, subject_type, predicate, object, object_type, description, confidence, embedding, chunk_index, model, created_at)
SELECT id, source_id, group_id, source_node_name, source_node_type, relation, target_node_name, target_node_type, description, weight, embedding, chunk_index, model, created_at
FROM extracted_edges;

DROP TABLE IF EXISTS extracted_edges;

-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
CREATE TABLE IF NOT EXISTS extracted_edges (
    id VARCHAR(255) PRIMARY KEY,
    source_id VARCHAR(255),
    group_id VARCHAR(255),
    source_node_name TEXT,
    source_node_type VARCHAR(50),
    target_node_name TEXT,
    target_node_type VARCHAR(50),
    relation TEXT,
    description TEXT,
    embedding JSON,
    weight FLOAT,
    chunk_index INT,
    model VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_id) REFERENCES sources(id)
);
-- +goose StatementEnd
