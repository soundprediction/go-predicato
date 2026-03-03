// Package migrations provides embedded SQL migration files for the predicato
// structured store. Each subdirectory contains goose-format migrations for a
// specific database backend.
//
// Supported backends: postgres, duckdb.
//
// Usage with goose:
//
//	import "github.com/soundprediction/predicato/pkg/factstore/migrations"
//
//	fs := migrations.FS("postgres")
//	goose.SetBaseFS(fs)
//	goose.Up(db, ".")
package migrations

import (
	"embed"
	"fmt"
	"io/fs"
)

//go:embed postgres/*.sql
var postgres embed.FS

//go:embed duckdb/*.sql
var duckdb embed.FS

// FS returns the embedded filesystem for the given database dialect.
// Valid dialects: "postgres", "duckdb".
func FS(dialect string) (fs.FS, error) {
	switch dialect {
	case "postgres":
		return fs.Sub(postgres, "postgres")
	case "duckdb":
		return fs.Sub(duckdb, "duckdb")
	default:
		return nil, fmt.Errorf("unsupported dialect %q (supported: postgres, duckdb)", dialect)
	}
}
