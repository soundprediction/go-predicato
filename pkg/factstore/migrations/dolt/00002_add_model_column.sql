-- +goose Up
-- +goose StatementBegin
ALTER TABLE extracted_nodes ADD COLUMN model VARCHAR(255);
ALTER TABLE extracted_edges ADD COLUMN model VARCHAR(255);
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
ALTER TABLE extracted_nodes DROP COLUMN model;
ALTER TABLE extracted_edges DROP COLUMN model;
-- +goose StatementEnd
