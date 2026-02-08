-- +goose Up
-- +goose StatementBegin
ALTER TABLE extracted_nodes ADD COLUMN model VARCHAR;
ALTER TABLE extracted_edges ADD COLUMN model VARCHAR;
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
ALTER TABLE extracted_nodes DROP COLUMN model;
ALTER TABLE extracted_edges DROP COLUMN model;
-- +goose StatementEnd
