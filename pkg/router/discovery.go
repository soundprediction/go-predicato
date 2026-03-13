package router

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// DiscoverTopicGraphs scans a directory for topic graph databases paired with
// routing descriptors. For each <slug>.md file with a matching database file,
// it creates a ClientConfig.
//
// Supported database formats:
//   - <slug>.duckdb  → DuckPGQ
//   - <slug>.lbug    → LadybugDB
//   - <slug>.ladybug → LadybugDB
//
// Returns the discovered client configs and a map of slug → routing text
// (description + routing signals from the .md file) for use by the cross-encoder classifier.
func DiscoverTopicGraphs(dir string) ([]ClientConfig, map[string]string, error) {
	if dir == "" {
		return nil, nil, fmt.Errorf("topic graphs directory is empty")
	}

	absDir, err := filepath.Abs(dir)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to resolve directory path: %w", err)
	}

	mdFiles, err := filepath.Glob(filepath.Join(absDir, "*.md"))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to glob md files: %w", err)
	}

	var configs []ClientConfig
	routingTexts := make(map[string]string)
	firstClient := true

	for _, mdPath := range mdFiles {
		slug := strings.TrimSuffix(filepath.Base(mdPath), ".md")

		dbPath, dbType := findTopicDatabase(absDir, slug)
		if dbPath == "" {
			continue
		}

		mdContent, err := os.ReadFile(mdPath)
		if err != nil {
			log.Printf("WARNING: failed to read topic descriptor %s: %v", mdPath, err)
			continue
		}

		keywords, routingText := parseTopicDescriptor(string(mdContent))

		// Try to read group_id from companion .yaml file
		groupID := "default"
		yamlPath := filepath.Join(absDir, slug+".yaml")
		if yamlContent, err := os.ReadFile(yamlPath); err == nil {
			if gid := parseYAMLGroupID(string(yamlContent)); gid != "" {
				groupID = gid
			}
		}

		graphDbMap := map[string]any{
			"type":    string(dbType),
			"db_path": dbPath,
		}

		cfg := ClientConfig{
			Name:     slug,
			GroupID:  groupID,
			Topics:   keywords,
			GraphDb:  graphDbMap,
			Default:  firstClient,
			ReadOnly: true,
		}

		configs = append(configs, cfg)
		if routingText != "" {
			routingTexts[slug] = routingText
		}
		firstClient = false
	}

	return configs, routingTexts, nil
}

// findTopicDatabase looks for a database file matching the slug.
func findTopicDatabase(dir, slug string) (string, GraphDbType) {
	// Check for DuckPGQ database
	duckdbPath := filepath.Join(dir, slug+".duckdb")
	if _, err := os.Stat(duckdbPath); err == nil {
		return duckdbPath, GraphDbTypeDuckPGQ
	}

	// Check for Ladybug database (.lbug directory)
	lbugPath := filepath.Join(dir, slug+".lbug")
	if info, err := os.Stat(lbugPath); err == nil && info.IsDir() {
		return lbugPath, GraphDbTypeLadybug
	}

	// Check for Ladybug database (.ladybug file)
	ladybugPath := filepath.Join(dir, slug+".ladybug")
	if _, err := os.Stat(ladybugPath); err == nil {
		return ladybugPath, GraphDbTypeLadybug
	}

	return "", ""
}

// parseTopicDescriptor parses a topic routing descriptor markdown file.
// Returns extracted keywords and the full routing text (description + routing signals).
func parseTopicDescriptor(content string) (keywords []string, routingText string) {
	lines := strings.Split(content, "\n")

	var (
		currentSection string
		description    strings.Builder
		routeSignals   strings.Builder
		keywordsRaw    strings.Builder
	)

	for _, line := range lines {
		trimmed := strings.TrimSpace(line)

		if after, ok := strings.CutPrefix(trimmed, "## "); ok {
			headerLower := strings.ToLower(after)

			switch {
			case strings.Contains(headerLower, "description"):
				currentSection = "description"
			case strings.Contains(headerLower, "route") || strings.Contains(headerLower, "routing"):
				currentSection = "routing"
			case strings.Contains(headerLower, "keyword"):
				currentSection = "keywords"
			default:
				currentSection = "other"
			}
			continue
		}

		if strings.HasPrefix(trimmed, "# ") || strings.HasPrefix(trimmed, "**Category") {
			continue
		}

		switch currentSection {
		case "description":
			if trimmed != "" {
				description.WriteString(trimmed)
				description.WriteString(" ")
			}
		case "routing":
			if trimmed != "" {
				routeSignals.WriteString(trimmed)
				routeSignals.WriteString(" ")
			}
		case "keywords":
			if trimmed != "" {
				keywordsRaw.WriteString(trimmed + " ")
			}
		}
	}

	keywords = parseKeywords(keywordsRaw.String())

	var parts []string
	if desc := strings.TrimSpace(description.String()); desc != "" {
		parts = append(parts, desc)
	}
	if signals := strings.TrimSpace(routeSignals.String()); signals != "" {
		parts = append(parts, signals)
	}
	routingText = strings.Join(parts, " ")

	return keywords, routingText
}

// parseYAMLGroupID extracts group_id from a simple YAML file without a full YAML parser.
func parseYAMLGroupID(content string) string {
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "group_id:") {
			val := strings.TrimPrefix(line, "group_id:")
			return strings.TrimSpace(val)
		}
	}
	return ""
}

// parseKeywords extracts keywords from a raw string.
// Supports backtick-delimited (`keyword`) and plain comma-separated formats.
func parseKeywords(raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}

	var keywords []string

	// Try backtick-delimited format first
	parts := strings.Split(raw, "`")
	for i, part := range parts {
		if i%2 == 1 {
			kw := strings.TrimSpace(part)
			if kw != "" {
				keywords = append(keywords, kw)
			}
		}
	}

	if len(keywords) > 0 {
		return keywords
	}

	// Fall back to comma-separated
	for _, kw := range strings.Split(raw, ",") {
		kw = strings.TrimSpace(kw)
		if kw != "" {
			keywords = append(keywords, kw)
		}
	}

	return keywords
}
