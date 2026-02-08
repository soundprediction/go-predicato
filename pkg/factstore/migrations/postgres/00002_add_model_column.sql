-- +goose Up
-- +goose StatementBegin
ALTER TABLE extracted_nodes ADD COLUMN IF NOT EXISTS model VARCHAR(255);
ALTER TABLE extracted_edges ADD COLUMN IF NOT EXISTS model VARCHAR(255);
ALTER TABLE extracted_triples ADD COLUMN IF NOT EXISTS model VARCHAR(255);
ALTER TABLE extracted_rules ADD COLUMN IF NOT EXISTS model VARCHAR(255);
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
ALTER TABLE extracted_nodes DROP COLUMN IF EXISTS model;
ALTER TABLE extracted_edges DROP COLUMN IF EXISTS model;
ALTER TABLE extracted_triples DROP COLUMN IF EXISTS model;
ALTER TABLE extracted_rules DROP COLUMN IF EXISTS model;
-- +goose StatementEnd
