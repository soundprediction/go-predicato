-- +goose Up
-- +goose StatementBegin
ALTER TABLE extracted_nodes ADD COLUMN model TEXT;
ALTER TABLE extracted_edges ADD COLUMN model TEXT;
-- +goose StatementEnd

-- +goose Down
-- NOTE: SQLite doesn't support DROP COLUMN before 3.35.0
