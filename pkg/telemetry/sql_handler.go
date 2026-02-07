package telemetry

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log/slog"
	"runtime"

	"github.com/google/uuid"
	"github.com/soundprediction/predicato/pkg/types"

	_ "github.com/go-sql-driver/mysql" // Ensure mysql driver is available for Dolt
)

// SQLDialect controls placeholder style and column types for SQL telemetry.
type SQLDialect int

const (
	// DialectMySQL uses ? placeholders and JSON column type.
	DialectMySQL SQLDialect = iota
	// DialectPostgres uses $1 placeholders and JSONB column type.
	DialectPostgres
)

// SQLHandler is a slog.Handler that writes logs to a SQL database
type SQLHandler struct {
	next      slog.Handler
	db        *sql.DB
	tableName string
	dialect   SQLDialect
}

// NewSQLHandler creates a new SQLHandler using an existing DB connection.
// Defaults to MySQL dialect for backward compatibility.
func NewSQLHandler(next slog.Handler, db *sql.DB) (*SQLHandler, error) {
	return NewSQLHandlerWithDialect(next, db, DialectMySQL)
}

// NewSQLHandlerWithDialect creates a new SQLHandler with the specified SQL dialect.
func NewSQLHandlerWithDialect(next slog.Handler, db *sql.DB, dialect SQLDialect) (*SQLHandler, error) {
	h := &SQLHandler{
		next:      next,
		db:        db,
		tableName: "telemetry_logs",
		dialect:   dialect,
	}

	if err := h.ensureTable(); err != nil {
		return nil, fmt.Errorf("failed to ensure telemetry table: %w", err)
	}

	return h, nil
}

func (h *SQLHandler) ensureTable() error {
	attrType := "JSON"
	if h.dialect == DialectPostgres {
		attrType = "JSONB"
	}

	query := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s (
			id VARCHAR(36) PRIMARY KEY,
			timestamp TIMESTAMP,
			level VARCHAR(10),
			message TEXT,
			user_id VARCHAR(255),
			session_id VARCHAR(255),
			request_source VARCHAR(255),
			source_file VARCHAR(255),
			line_number INT,
			attributes %s
		)
	`, h.tableName, attrType)

	_, err := h.db.Exec(query)
	return err
}

// Enabled implements slog.Handler
func (h *SQLHandler) Enabled(ctx context.Context, level slog.Level) bool {
	return h.next.Enabled(ctx, level)
}

// Handle implements slog.Handler
func (h *SQLHandler) Handle(ctx context.Context, r slog.Record) error {
	// Always pass to next handler first
	if err := h.next.Handle(ctx, r); err != nil {
		return err
	}

	// Only log errors (and above) to DB, similar to ParquetHandler
	if r.Level < slog.LevelError {
		return nil
	}

	// Extract context info
	var userID, sessionID, requestSource string
	if v, ok := ctx.Value(types.ContextKeyUserID).(string); ok {
		userID = v
	}
	if v, ok := ctx.Value(types.ContextKeySessionID).(string); ok {
		sessionID = v
	}
	if v, ok := ctx.Value(types.ContextKeyRequestSource).(string); ok {
		requestSource = v
	}

	// Extract attributes
	attrs := make(map[string]interface{})
	r.Attrs(func(a slog.Attr) bool {
		attrs[a.Key] = a.Value.Any()
		return true
	})

	attrsJson, _ := json.Marshal(attrs)

	// Get source info
	fs := runtime.CallersFrames([]uintptr{r.PC})
	f, _ := fs.Next()
	sourceFile := f.File
	line := f.Line

	id := uuid.New().String()
	timestamp := r.Time.UTC()

	var query string
	if h.dialect == DialectPostgres {
		query = fmt.Sprintf(`
			INSERT INTO %s (id, timestamp, level, message, user_id, session_id, request_source, source_file, line_number, attributes)
			VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
		`, h.tableName)
	} else {
		query = fmt.Sprintf(`
			INSERT INTO %s (id, timestamp, level, message, user_id, session_id, request_source, source_file, line_number, attributes)
			VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		`, h.tableName)
	}

	_, err := h.db.Exec(query,
		id,
		timestamp,
		r.Level.String(),
		r.Message,
		userID,
		sessionID,
		requestSource,
		sourceFile,
		line,
		string(attrsJson),
	)

	if err != nil {
		// Log failure to fallback handler (e.g. stderr)
		fmt.Printf("Failed to write log to SQL: %v\n", err)
	}

	return nil // Don't block logging chain on database error
}

// WithAttrs implements slog.Handler
func (h *SQLHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return &SQLHandler{
		next:      h.next.WithAttrs(attrs),
		db:        h.db,
		tableName: h.tableName,
		dialect:   h.dialect,
	}
}

// WithGroup implements slog.Handler
func (h *SQLHandler) WithGroup(name string) slog.Handler {
	return &SQLHandler{
		next:      h.next.WithGroup(name),
		db:        h.db,
		tableName: h.tableName,
		dialect:   h.dialect,
	}
}
