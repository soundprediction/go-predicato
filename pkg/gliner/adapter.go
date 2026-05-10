package gliner

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"
	"strings"

	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/types"
)

// LLMAdapter wraps a gliner.Client to implement nlp.Client interface
// It intercepts extraction prompts and uses GLiNER models
// It delegates other prompts to a base LLM client
type LLMAdapter struct {
	glinerClient *Client
	baseClient   nlp.Client
	logger       *slog.Logger
}

func NewLLMAdapter(glinerClient *Client, baseClient nlp.Client) *LLMAdapter {
	return &LLMAdapter{
		glinerClient: glinerClient,
		baseClient:   baseClient,
		logger:       slog.Default(),
	}
}

func (a *LLMAdapter) SetLogger(l *slog.Logger) {
	a.logger = l
}

// Close implements Client
func (a *LLMAdapter) Close() error {
	if a.glinerClient != nil {
		a.glinerClient.Close()
	}
	if a.baseClient != nil {
		return a.baseClient.Close()
	}
	return nil
}

// GetCapabilities returns the list of capabilities supported by this client.
func (a *LLMAdapter) GetCapabilities() []nlp.TaskCapability {
	// Start with NER and Relation Extraction capabilities from GLiNER
	caps := []nlp.TaskCapability{nlp.TaskNamedEntityRecognition, nlp.TaskRelationExtraction}

	// Delegate to base client for other capabilities if available
	if a.baseClient != nil {
		baseCaps := a.baseClient.GetCapabilities()
		for _, cap := range baseCaps {
			// Avoid duplicates
			found := false
			for _, existing := range caps {
				if existing == cap {
					found = true
					break
				}
			}
			if !found {
				caps = append(caps, cap)
			}
		}
	}
	return caps
}

// GetModel returns the model identifier.
func (a *LLMAdapter) GetModel() string {
	if a.baseClient != nil {
		return "gliner-adapter(" + a.baseClient.GetModel() + ")"
	}
	return "gliner-adapter"
}

func (a *LLMAdapter) Chat(ctx context.Context, messages []types.Message) (*types.Response, error) {
	// 1. Inspect messages to detect extraction pattern
	if len(messages) == 0 {
		return &types.Response{Content: ""}, nil
	}

	systemMsg := ""
	lastUserMsg := ""
	for _, m := range messages {
		if m.Role == "system" { // nlp.RoleSystem is constrained const, string comparison safe
			systemMsg = m.Content
		}
		if m.Role == "user" {
			lastUserMsg = m.Content
		}
	}

	// NODE EXTRACTION DETECTION
	if strings.Contains(systemMsg, "extracts entity nodes") || strings.Contains(systemMsg, "extracts entity nodes from") {
		a.logger.Info("GLiNER Adapter: Detected Node Extraction request")
		return a.handleNodeExtraction(lastUserMsg)
	}

	// EDGE EXTRACTION DETECTION
	if strings.Contains(systemMsg, "expert fact extractor") || strings.Contains(systemMsg, "extracts fact triples") {
		a.logger.Info("GLiNER Adapter: Detected Edge Extraction request")
		return a.handleEdgeExtraction(lastUserMsg)
	}

	// Fallback to base client
	if a.baseClient != nil {
		return a.baseClient.Chat(ctx, messages)
	}

	return nil, fmt.Errorf("no base client configured and prompt not handled by GLiNER")
}

func (a *LLMAdapter) ChatWithStructuredOutput(ctx context.Context, messages []types.Message, schema any) (*types.Response, error) {
	// GLiNER does not support structured output directly mapping to arbitrary schemas easily yet.
	// For now, fallback to base client.
	if a.baseClient != nil {
		return a.baseClient.ChatWithStructuredOutput(ctx, messages, schema)
	}
	return nil, fmt.Errorf("ChatWithStructuredOutput not supported by GLiNER adapter without base client")
}

// ExtractEntities implements Client
func (a *LLMAdapter) ExtractEntities(ctx context.Context, text string, entityTypes []string) ([]nlp.ExtractedEntity, error) {
	// Use GLiNER client directly
	entities, err := a.glinerClient.ExtractEntities(text, entityTypes)
	if err != nil {
		return nil, err
	}

	var extracted []nlp.ExtractedEntity
	for _, e := range entities {
		extracted = append(extracted, nlp.ExtractedEntity{
			Text:       e.Text,
			Label:      e.Label,
			Confidence: float64(e.Score),
			Start:      0, // Legacy GLiNER client doesn't expose offsets
			End:        0,
		})
	}
	return extracted, nil
}

// ExtractRelations implements Client
func (a *LLMAdapter) ExtractRelations(ctx context.Context, text string, relationTypes []string) ([]nlp.ExtractedRelation, error) {
	// GLiNER ExtractRelations needs a Schema (map[string][2][]string).
	// But the interface only provides relationTypes (names).
	// Legacy GLiNER adapter parsed schema from prompt content (fact types signature).
	// If consumer calls ExtractRelations directly, they might expect us to know the schema or infer it?
	// Or maybe pass it? The interface doesn't support complex schema passing yet.
	// However, if we don't have schema, GLiNER might fail or we might need to assume all-to-all?
	// For now, let's error if we can't do it, or try to build a default schema?
	// Since usage in node_operations/ExtractEdges now calls client.ExtractRelations, we need to support it.
	// But newer clients (GLiNER2, OpenAI) handle it differently.
	// Legacy GLiNER implementation might be strict.

	// Hack: if baseClient exists, use it? No, we want GLiNER extraction.
	// Let's defer to baseClient if relationTypes is simple list, OR error.
	// Actually, let's try to construct a schema where all relations apply to all entities?
	// But we need entities to build schema? No.

	return nil, fmt.Errorf("ExtractRelations with simple signature not supported by legacy GLiNER adapter; use Chat or upgrade")
}

// Summarize implements Client
func (a *LLMAdapter) Summarize(ctx context.Context, text string) (string, error) {
	if a.baseClient != nil {
		return a.baseClient.Summarize(ctx, text)
	}
	return "", fmt.Errorf("Summarize not supported: no base client")
}

// GenerateText implements Client
func (a *LLMAdapter) GenerateText(ctx context.Context, prompt string) (string, error) {
	if a.baseClient != nil {
		return a.baseClient.GenerateText(ctx, prompt)
	}
	return "", fmt.Errorf("GenerateText not supported: no base client")
}

// ExtractExtended implements Client — not supported by legacy GLiNER v1 adapter.
func (a *LLMAdapter) ExtractExtended(ctx context.Context, text string, entityTypes, relationTypes []string) (*nlp.ExtendedExtractionResult, error) {
	if a.baseClient != nil {
		return a.baseClient.ExtractExtended(ctx, text, entityTypes, relationTypes)
	}
	return nil, fmt.Errorf("ExtractExtended not supported: upgrade to GLiNER2 or provide a base LLM client")
}

// ---- Extraction Handlers ----

// parseSection extracts content between <TAG> and </TAG>
func parseSection(text, tag string) string {
	open := "<" + tag + ">"
	close := "</" + tag + ">"
	start := strings.Index(text, open)
	if start < 0 {
		return ""
	}
	start += len(open)
	end := strings.Index(text[start:], close)
	if end < 0 {
		return ""
	}
	return strings.TrimSpace(text[start : start+end])
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

func (a *LLMAdapter) handleNodeExtraction(userMsg string) (*types.Response, error) {
	// Parse input
	entityTypesTSV := parseSection(userMsg, "ENTITY TYPES")
	text := parseSection(userMsg, "TEXT")
	if text == "" {
		text = parseSection(userMsg, "CURRENT MESSAGE")
	}
	if text == "" {
		text = parseSection(userMsg, "JSON")
	}

	// Parse Entity Types to map Name -> ID
	// Format: entity_type_id\tentity_type_name...
	typesRecords := parseTSV(entityTypesTSV)

	labelToID := make(map[string]string)
	var labels []string

	// Skip header if present (check if first row has "entity_type_id" string)
	startIndex := 0
	if len(typesRecords) > 0 && (strings.Contains(typesRecords[0][0], "entity_type_id") || strings.Contains(typesRecords[0][1], "entity_type_name")) {
		startIndex = 1
	}

	for i := startIndex; i < len(typesRecords); i++ {
		row := typesRecords[i]
		if len(row) < 2 {
			continue
		}
		// Assuming standard prompts: entity_type_id is col 0, entity_type_name is col 1
		id := row[0]
		name := row[1]
		// Verify
		if _, err := strconv.Atoi(id); err != nil {
			// maybe swapped?
			if _, err2 := strconv.Atoi(row[1]); err2 == nil {
				id = row[1]
				name = row[0]
			}
		}

		labelToID[name] = id
		labels = append(labels, name)
	}

	// Run Extraction
	entities, err := a.glinerClient.ExtractEntities(text, labels)
	if err != nil {
		return nil, fmt.Errorf("GLiNER node extraction failed: %w", err)
	}

	// Format Output as TSV
	// entity\tentity_type_id
	var sb strings.Builder
	sb.WriteString("entity\tentity_type_id\n")

	for _, e := range entities {
		id, ok := labelToID[e.Label]
		if !ok {
			// Should not happen if labels matched
			id = "-1"
		}
		sb.WriteString(fmt.Sprintf("%s\t%s\n", e.Text, id))
	}

	return &types.Response{
		Content: sb.String(),
	}, nil
}

func (a *LLMAdapter) handleEdgeExtraction(userMsg string) (*types.Response, error) {
	// Parse input
	factTypesTSV := parseSection(userMsg, "FACT TYPES")
	extractedEntitiesTSV := parseSection(userMsg, "ENTITIES")
	text := parseSection(userMsg, "CURRENT_MESSAGE") // in edgePrompt it's CURRENT_MESSAGE
	if text == "" {
		text = parseSection(userMsg, "CURRENT MESSAGE")
	}

	// Parse Entities to map Text -> ID (as string)
	entitiesRecords := parseTSV(extractedEntitiesTSV)
	nameToID := make(map[string]string)

	// Skip header
	startIndex := 0
	if len(entitiesRecords) > 0 && (strings.Contains(entitiesRecords[0][0], "id") || strings.Contains(entitiesRecords[0][1], "name")) {
		startIndex = 1
	}

	// Need to identify columns dynamically
	idCol := -1
	nameCol := -1
	if len(entitiesRecords) > 0 {
		header := entitiesRecords[0]
		for i, h := range header {
			if h == "id" || h == "node_id" {
				idCol = i
			}
			if h == "name" || h == "entity" {
				nameCol = i
			}
		}
	}
	// Fallback default
	if idCol == -1 {
		idCol = 0
	}
	if nameCol == -1 {
		nameCol = 1
	}

	// Collect labels for GLiNER Relation (Entity Labels)
	allLabelsMap := make(map[string]bool)

	// Parse Fact Types (Schema)
	factTypesRecords := parseTSV(factTypesTSV)
	schema := make(map[string][2][]string) // Rel -> [Heads, Tails]

	startFact := 0
	if len(factTypesRecords) > 0 && strings.Contains(factTypesRecords[0][0], "relation_type") {
		startFact = 1
	}

	// Find columns (sorted keys)
	sigCol := -1
	relCol := -1

	if len(factTypesRecords) > 0 {
		header := factTypesRecords[0]
		for i, h := range header {
			if h == "fact_type_signature" {
				sigCol = i
			}
			if h == "relation_type" {
				relCol = i
			}
		}
	}
	if sigCol == -1 {
		sigCol = 0
	}
	if relCol == -1 {
		relCol = 1
	}

	for i := startFact; i < len(factTypesRecords); i++ {
		row := factTypesRecords[i]
		if len(row) <= max(sigCol, relCol) {
			continue
		}

		sig := row[sigCol]
		rel := row[relCol]

		// Parse matches "Head->REL->Tail"
		parts := strings.Split(sig, "->")
		if len(parts) == 3 {
			head := parts[0]
			tail := parts[2]

			// Add to schema
			schema[rel] = [2][]string{{head}, {tail}}

			allLabelsMap[head] = true
			allLabelsMap[tail] = true
		}
	}

	var allLabels []string
	for l := range allLabelsMap {
		allLabels = append(allLabels, l)
	}

	// Populate nameToID mapping
	for i := startIndex; i < len(entitiesRecords); i++ {
		row := entitiesRecords[i]
		if len(row) <= max(idCol, nameCol) {
			continue
		}
		id := row[idCol]
		name := row[nameCol]
		nameToID[name] = id
	}

	// Run Extraction
	relations, err := a.glinerClient.ExtractRelations(text, allLabels, schema)
	if err != nil {
		return nil, fmt.Errorf("GLiNER relation extraction failed: %w", err)
	}

	// Format Output as TSV
	var sb strings.Builder
	sb.WriteString("source_id\trelation_type\ttarget_id\tfact\tsummary\tvalid_at\tinvalid_at\n")

	for _, r := range relations {
		srcID, okSrc := nameToID[r.Source]
		tgtID, okTgt := nameToID[r.Target]

		// Only emit if we can map back to IDs
		if okSrc && okTgt {
			fact := fmt.Sprintf("%s %s %s", r.Source, r.Type, r.Target)
			sb.WriteString(fmt.Sprintf("%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
				srcID, r.Type, tgtID, fact, "", "null", "null"))
		}
	}

	return &types.Response{
		Content: sb.String(),
	}, nil
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
