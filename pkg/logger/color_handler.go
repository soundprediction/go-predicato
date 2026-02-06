package logger

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"sync"
)

// ANSI color codes
const (
	colorReset  = "\033[0m"
	colorRed    = "\033[31m"
	colorYellow = "\033[33m"
	colorGreen  = "\033[32m"
)

// ColorHandler wraps an slog.Handler and adds color to messages based on level and content:
// - Error messages: Red
// - Warning messages: Yellow
// - Info messages containing "persist": Green (for database operations)
// - Other messages: Standard output
type ColorHandler struct {
	w      io.Writer
	attrs  []slog.Attr
	groups []string
	level  slog.Level
	mu     sync.Mutex
}

// NewColorHandler creates a new colored handler that writes directly to w
func NewColorHandler(w io.Writer, opts *slog.HandlerOptions) *ColorHandler {
	level := slog.LevelInfo
	if opts != nil && opts.Level != nil {
		level = opts.Level.Level()
	}
	return &ColorHandler{
		w:     w,
		level: level,
	}
}

// Enabled implements slog.Handler
func (h *ColorHandler) Enabled(ctx context.Context, level slog.Level) bool {
	return level >= h.level
}

// Handle implements slog.Handler and adds color based on log level
func (h *ColorHandler) Handle(ctx context.Context, r slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Determine color based on level and message content
	var color string
	switch r.Level {
	case slog.LevelError:
		color = colorRed
	case slog.LevelWarn:
		color = colorYellow
	case slog.LevelInfo:
		// Color persist messages green
		msgLower := strings.ToLower(r.Message)
		if strings.Contains(msgLower, "persist") {
			color = colorGreen
		}
	}

	// Build output string
	var buf strings.Builder

	// Write timestamp
	buf.WriteString(r.Time.Format("2006-01-02 15:04:05"))
	buf.WriteString(" ")

	// Write level
	buf.WriteString(r.Level.String())
	buf.WriteString(" ")

	// Write colored message
	if color != "" {
		buf.WriteString(color)
	}
	buf.WriteString(r.Message)
	if color != "" {
		buf.WriteString(colorReset)
	}

	// Write attributes
	r.Attrs(func(a slog.Attr) bool {
		buf.WriteString(" ")
		buf.WriteString(a.Key)
		buf.WriteString("=")
		buf.WriteString(a.Value.String())
		return true
	})

	// Write handler-level attributes
	for _, attr := range h.attrs {
		buf.WriteString(" ")
		buf.WriteString(attr.Key)
		buf.WriteString("=")
		buf.WriteString(attr.Value.String())
	}

	buf.WriteString("\n")

	_, err := fmt.Fprint(h.w, buf.String())
	return err
}

// WithAttrs implements slog.Handler
func (h *ColorHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	newAttrs := make([]slog.Attr, len(h.attrs)+len(attrs))
	copy(newAttrs, h.attrs)
	copy(newAttrs[len(h.attrs):], attrs)

	return &ColorHandler{
		w:      h.w,
		level:  h.level,
		attrs:  newAttrs,
		groups: h.groups,
	}
}

// WithGroup implements slog.Handler
func (h *ColorHandler) WithGroup(name string) slog.Handler {
	newGroups := make([]string, len(h.groups)+1)
	copy(newGroups, h.groups)
	newGroups[len(h.groups)] = name

	return &ColorHandler{
		w:      h.w,
		level:  h.level,
		attrs:  h.attrs,
		groups: newGroups,
	}
}
