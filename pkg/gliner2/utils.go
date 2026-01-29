package gliner2

import (
	"encoding/json"
	"regexp"
	"strings"
)

// parseSection extracts content between <TAG> and </TAG>
func parseSection(text, tag string) string {
	re := regexp.MustCompile(regexp.QuoteMeta(`<` + tag + `>\s*([\s\S]*?)\s*</` + tag + `>`))
	match := re.FindStringSubmatch(text)
	if len(match) > 1 {
		return match[1]
	}
	return ""
}

// parseTSV parses simple TSV string into slice of records
func parseTSV(tsv string) [][]string {
	lines := strings.Split(tsv, "\n")
	var records [][]string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.Split(line, "\t")
		// clean quotes?
		for i := range parts {
			parts[i] = strings.Trim(parts[i], "\"")
		}
		records = append(records, parts)
	}
	return records
}

// extractClassificationSchema extracts classification labels from the user message
func extractClassificationSchema(userMsg string) []string {
	// Try to extract from LABELS section
	labelsSection := parseSection(userMsg, "LABELS")
	if labelsSection != "" {
		var labels []string
		for _, line := range strings.Split(labelsSection, "\n") {
			line = strings.TrimSpace(line)
			if line != "" {
				labels = append(labels, line)
			}
		}
		if len(labels) > 0 {
			return labels
		}
	}

	// Try to extract from CATEGORIES section
	categoriesSection := parseSection(userMsg, "CATEGORIES")
	if categoriesSection != "" {
		var labels []string
		for _, line := range strings.Split(categoriesSection, "\n") {
			line = strings.TrimSpace(line)
			if line != "" {
				labels = append(labels, line)
			}
		}
		if len(labels) > 0 {
			return labels
		}
	}

	return nil
}

// extractTextContent extracts the text to classify from the user message
func extractTextContent(userMsg string) string {
	// Try common section names
	text := parseSection(userMsg, "TEXT")
	if text != "" {
		return text
	}
	text = parseSection(userMsg, "CONTENT")
	if text != "" {
		return text
	}
	text = parseSection(userMsg, "INPUT")
	if text != "" {
		return text
	}
	// Fallback to the entire message
	return userMsg
}

// formatClassificationResult formats a ClassificationResult as JSON
func formatClassificationResult(result *ClassificationResult) (string, error) {
	if result == nil {
		return "{}", nil
	}
	data, err := json.Marshal(result)
	if err != nil {
		return "", err
	}
	return string(data), nil
}
