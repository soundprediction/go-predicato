package telemetry

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/parquet-go/parquet-go"
	"github.com/soundprediction/predicato/pkg/types"
)

// LogRecord represents a single log entry for Parquet storage
type LogRecord struct {
	Timestamp     time.Time `parquet:"timestamp"`
	ID            string    `parquet:"id"`
	Level         string    `parquet:"level"`
	Message       string    `parquet:"message"`
	UserID        string    `parquet:"user_id"`
	SessionID     string    `parquet:"session_id"`
	RequestSource string    `parquet:"request_source"`
	SourceFile    string    `parquet:"source_file"`
	Attributes    string    `parquet:"attributes"` // JSON string
	LineNumber    int       `parquet:"line_number"`
}

// ParquetHandler is a slog.Handler that writes error logs to Parquet files
type ParquetHandler struct {
	next      slog.Handler
	outputDir string
	buffer    []LogRecord
	batchSize int
	mu        sync.Mutex
}

// NewParquetHandler creates a new ParquetHandler
func NewParquetHandler(next slog.Handler, outputDir string) (*ParquetHandler, error) {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create telemetry directory: %w", err)
	}

	h := &ParquetHandler{
		next:      next,
		outputDir: outputDir,
		batchSize: 100, // Flush every 100 logs or on close (not implemented yet for close)
		buffer:    make([]LogRecord, 0, 100),
	}

	return h, nil
}

// Enabled implements slog.Handler
func (h *ParquetHandler) Enabled(ctx context.Context, level slog.Level) bool {
	return h.next.Enabled(ctx, level)
}

// Handle implements slog.Handler
func (h *ParquetHandler) Handle(ctx context.Context, r slog.Record) error {
	// Always pass to next handler first
	if err := h.next.Handle(ctx, r); err != nil {
		return err
	}

	// Only log errors (and above) to DB
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

	record := LogRecord{
		ID:            uuid.New().String(),
		Timestamp:     r.Time.UTC(),
		Level:         r.Level.String(),
		Message:       r.Message,
		UserID:        userID,
		SessionID:     sessionID,
		RequestSource: requestSource,
		SourceFile:    sourceFile,
		LineNumber:    line,
		Attributes:    string(attrsJson),
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	h.buffer = append(h.buffer, record)

	if len(h.buffer) >= h.batchSize {
		return h.flush()
	}

	return nil
}

// flush writes the current buffer to a new Parquet file
// Caller must hold the lock
func (h *ParquetHandler) flush() error {
	if len(h.buffer) == 0 {
		return nil
	}

	filename := fmt.Sprintf("execution_errors_%s_%d.parquet", time.Now().Format("20060102_150405"), time.Now().UnixNano())
	filepath := filepath.Join(h.outputDir, filename)

	err := parquet.WriteFile(filepath, h.buffer)
	if err != nil {
		// Log to stderr if file write fails, but don't crash
		fmt.Printf("Failed to write telemetry parquet file: %v\n", err)
		return err
	}

	// Clear buffer
	h.buffer = h.buffer[:0]
	return nil
}

// WithAttrs implements slog.Handler
func (h *ParquetHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return &ParquetHandler{
		next:      h.next.WithAttrs(attrs),
		outputDir: h.outputDir,
		batchSize: h.batchSize,
		// Note: buffer is shared or new? Sharing buffer for clones is complex.
		// For simplicity in this refactor, we just create a new handler with empty buffer.
		// This means child loggers have their own batching.
		buffer: make([]LogRecord, 0, h.batchSize),
	}
}

// WithGroup implements slog.Handler
func (h *ParquetHandler) WithGroup(name string) slog.Handler {
	return &ParquetHandler{
		next:      h.next.WithGroup(name),
		outputDir: h.outputDir,
		batchSize: h.batchSize,
		buffer:    make([]LogRecord, 0, h.batchSize),
	}
}
