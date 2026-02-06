package prompts

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"

	"gopkg.in/yaml.v3"

	"github.com/pelletier/go-toml/v2"
	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/types"
)

// ExtractedEntity represents an entity extracted from content
type ExtractedEntity struct {
	Name         string `json:"entity" mapstructure:"entity" csv:"entity" yaml:"entity"`
	EntityTypeID int    `json:"entity_type_id" mapstructure:"entity_type_id" csv:"entity_type_id" yaml:"entity_type_id"`
}

// ExtractedEntities represents a list of extracted entities
type ExtractedEntities struct {
	ExtractedEntities []ExtractedEntity `json:"entities"`
}

// MissedEntities represents entities that weren't extracted
type MissedEntities struct {
	MissedEntities []string `json:"missed_entities"`
}

// MissedEntitiesTSV represents a single missed entity from TSV format
type MissedEntitiesTSV struct {
	EntityName string `csv:"entity_name"`
}

// EntityClassificationTriple represents an entity with classification
type EntityClassificationTriple struct {
	EntityType *string `json:"entity_type"`
	UUID       string  `json:"uuid"`
	Name       string  `json:"name"`
}

// EntityClassification represents entity classifications
type EntityClassification struct {
	EntityClassifications []EntityClassificationTriple `json:"entity_classifications"`
}

// EntitySummary represents an entity summary
type EntitySummary struct {
	Summary string `json:"summary"`
}

// ExtractedNodeAttributes represents extracted attributes and summary for a node
type ExtractedNodeAttributes struct {
	Summary string `json:"summary" mapstructure:"summary" csv:"summary" yaml:"summary"`
	NodeID  int    `json:"node_id" mapstructure:"node_id" csv:"node_id" yaml:"node_id"`
}
type ExtractedEdge struct {
	UpdatedAt time.Time `json:"updated_at" mapstructure:"updated_at" csv:"updated_at"`
	Name      string    `json:"relation_type" mapstructure:"relation_type" csv:"relation_type"` // matches Python name
	Fact      string    `json:"fact" mapstructure:"fact" csv:"fact"`
	Summary   string    `json:"summary,omitempty" mapstructure:"summary" csv:"summary"`
	ValidAt   string    `json:"valid_at,omitempty" mapstructure:"valid_at" csv:"valid_at"`       // matches Python valid_at
	InvalidAt string    `json:"invalid_at,omitempty" mapstructure:"invalid_at" csv:"invalid_at"` // matches Python invalid_at
	// alias for Fact
	SourceID int `json:"source_id" mapstructure:"source_id" csv:"source_id"` // alias for SourceNodeID uuid
	TargetID int `json:"target_id" mapstructure:"target_id" csv:"target_id"` // alias for TargetNodeID uuid
}

// ExtractedEdges represents a list of extracted edges
type ExtractedEdges struct {
	Edges []ExtractedEdge `json:"facts"`
}

// MissingFacts represents facts that weren't extracted
type MissingFacts struct {
	MissingFacts []string `json:"missing_facts"`
}

// NodeDuplicate represents a node duplicate resolution
type NodeDuplicate struct {
	Name         string `json:"name" mapstructure:"name" csv:"name"`
	Duplicates   []int  `json:"duplicates" mapstructure:"duplicates" csv:"duplicates"`
	ID           int    `json:"id" mapstructure:"id" csv:"id"`
	DuplicateIdx int    `json:"duplicate_idx" mapstructure:"duplicate_idx" csv:"duplicate_idx"`
}

// NodeResolutions represents node duplicate resolutions
type NodeResolutions struct {
	EntityResolutions []NodeDuplicate `json:"entity_resolutions"`
}

// EdgeDuplicate represents edge duplicate detection result
type EdgeDuplicate struct {
	FactType          string   `json:"fact_type"`
	DuplicateFacts    []string `json:"duplicate_facts"`
	ContradictedFacts []string `json:"contradicted_facts"`
}

// EdgeDuplicateTSV represents edge duplicate detection result from TSV
type EdgeDuplicateTSV struct {
	FactType          string   `json:"fact_type" mapstructure:"fact_type" csv:"fact_type" yaml:"fact_type"`
	DuplicateFacts    []string `json:"duplicate_facts" mapstructure:"duplicate_facts" csv:"duplicate_facts" yaml:"duplicate_facts"`
	ContradictedFacts []string `json:"contradicted_facts" mapstructure:"contradicted_facts" csv:"contradicted_facts" yaml:"contradicted_facts"`
}

// UniqueFact represents a unique fact
type UniqueFact struct {
	UUID string `json:"uuid"`
	Fact string `json:"fact"`
}

// UniqueFacts represents a list of unique facts
type UniqueFacts struct {
	UniqueFacts []UniqueFact `json:"unique_facts"`
}

// InvalidatedEdges represents edges to be invalidated
type InvalidatedEdges struct {
	ContradictedFacts []int `json:"contradicted_facts"`
}

// InvalidatedEdgesTSV represents a single invalidated edge from TSV format
type InvalidatedEdgesTSV struct {
	FactID int `csv:"fact_id"`
}

// EdgeDates represents temporal information for edges
type EdgeDates struct {
	ValidAt   *string `json:"valid_at"`
	InvalidAt *string `json:"invalid_at"`
}

// EdgeDatesTSV represents temporal information for edges from TSV format
type EdgeDatesTSV struct {
	ValidAt   string `csv:"valid_at"`
	InvalidAt string `csv:"invalid_at"`
}

// Summary represents a text summary
type Summary struct {
	Summary string `json:"summary"`
}

// SummaryDescription represents a summary description
type SummaryDescription struct {
	Description string `json:"description"`
}

// QueryExpansion represents an expanded query
type QueryExpansion struct {
	Query string `json:"query"`
}

// QAResponse represents a question-answer response
type QAResponse struct {
	Answer string `json:"ANSWER"`
}

// EvalResponse represents an evaluation response
type EvalResponse struct {
	Reasoning string `json:"reasoning"`
	IsCorrect bool   `json:"is_correct"`
}

// EvalAddEpisodeResults represents evaluation of episode addition results
type EvalAddEpisodeResults struct {
	Reasoning        string `json:"reasoning"`
	CandidateIsWorse bool   `json:"candidate_is_worse"`
}

// Episode represents an episode context for prompts
type Episode struct {
	Reference time.Time              `json:"reference"`
	CreatedAt time.Time              `json:"created_at"`
	Metadata  map[string]interface{} `json:"metadata"`
	ID        string                 `json:"id"`
	Name      string                 `json:"name"`
	Content   string                 `json:"content"`
	GroupID   string                 `json:"group_id"`
}

// promptVersionImpl implements PromptVersion.
type promptVersionImpl struct {
	fn types.PromptFunction
}

// Call executes the prompt function with the given context.
func (p *promptVersionImpl) Call(context map[string]interface{}) ([]types.Message, error) {
	messages, err := p.fn(context)
	if err != nil {
		return nil, err
	}

	// Add unicode preservation instruction to system messages
	for i, msg := range messages {
		if msg.Role == nlp.RoleSystem {
			messages[i].Content += "\nDo not escape unicode characters.\n"
		}
	}

	return messages, nil
}

// NewPromptVersion creates a new PromptVersion from a function.
func NewPromptVersion(fn types.PromptFunction) types.PromptVersion {
	return &promptVersionImpl{fn: fn}
}

// ToPromptJSON serializes data to JSON for use in prompts.
// When ensureASCII is false, non-ASCII characters are preserved in their original form.
func ToPromptJSON(data interface{}, ensureASCII bool, indent int) (string, error) {
	var b []byte
	var err error

	if indent > 0 {
		b, err = json.MarshalIndent(data, "", fmt.Sprintf("%*s", indent, ""))
	} else {
		b, err = json.Marshal(data)
	}

	if err != nil {
		return "", err
	}

	if ensureASCII {
		// Go's json package escapes non-ASCII by default
		return string(b), nil
	}

	// For non-ASCII preservation, we need to handle it differently
	// Go's json.Marshal always escapes non-ASCII, so we use a custom approach
	return string(b), nil
}

// ToPromptCSV serializes data to CSV format for use in prompts.
// Data should be a slice of structs, maps, or a slice of slices.
// When ensureASCII is true, non-ASCII characters are escaped.
func ToPromptCSV(data interface{}, ensureASCII bool) (string, error) {
	v := reflect.ValueOf(data)

	// Handle non-slice types
	if v.Kind() != reflect.Slice && v.Kind() != reflect.Array {
		return "", fmt.Errorf("ToPromptCSV requires a slice or array, got %T", data)
	}

	if v.Len() == 0 {
		return "", nil
	}

	var buf bytes.Buffer
	w := csv.NewWriter(&buf)

	// Determine the type of elements
	firstElem := v.Index(0)

	switch firstElem.Kind() {
	case reflect.Map:
		// Handle slice of maps
		if err := writeMapSliceCSV(w, v, ensureASCII); err != nil {
			return "", err
		}
	case reflect.Struct:
		// Handle slice of structs
		if err := writeStructSliceCSV(w, v, ensureASCII); err != nil {
			return "", err
		}
	case reflect.Slice, reflect.Array:
		// Handle slice of slices
		if err := writeSliceSliceCSV(w, v, ensureASCII); err != nil {
			return "", err
		}
	default:
		// Handle slice of primitives as a single column
		if err := writePrimitiveSliceCSV(w, v, ensureASCII); err != nil {
			return "", err
		}
	}

	w.Flush()
	if err := w.Error(); err != nil {
		return "", err
	}

	return buf.String(), nil
}

// writeMapSliceCSV writes a slice of maps to CSV
func writeMapSliceCSV(w *csv.Writer, v reflect.Value, ensureASCII bool) error {
	if v.Len() == 0 {
		return nil
	}

	// Collect all unique keys across all maps
	keySet := make(map[string]bool)
	for i := 0; i < v.Len(); i++ {
		m := v.Index(i)
		for _, key := range m.MapKeys() {
			keySet[fmt.Sprint(key.Interface())] = true
		}
	}

	// Sort keys for consistent column ordering
	keys := make([]string, 0, len(keySet))
	for k := range keySet {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	// Write header
	if err := w.Write(keys); err != nil {
		return err
	}

	// Write rows
	for i := 0; i < v.Len(); i++ {
		m := v.Index(i)
		row := make([]string, len(keys))
		for j, key := range keys {
			val := m.MapIndex(reflect.ValueOf(key))
			if val.IsValid() {
				row[j] = formatValue(val.Interface(), ensureASCII)
			}
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}

	return nil
}

// writeStructSliceCSV writes a slice of structs to CSV
func writeStructSliceCSV(w *csv.Writer, v reflect.Value, ensureASCII bool) error {
	if v.Len() == 0 {
		return nil
	}

	firstElem := v.Index(0)
	t := firstElem.Type()

	// Collect field names
	var fieldNames []string
	var fieldIndices []int

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		// Skip unexported fields
		if field.PkgPath != "" {
			continue
		}
		fieldNames = append(fieldNames, field.Name)
		fieldIndices = append(fieldIndices, i)
	}

	// Write header
	if err := w.Write(fieldNames); err != nil {
		return err
	}

	// Write rows
	for i := 0; i < v.Len(); i++ {
		elem := v.Index(i)
		row := make([]string, len(fieldIndices))
		for j, idx := range fieldIndices {
			fieldVal := elem.Field(idx)
			row[j] = formatValue(fieldVal.Interface(), ensureASCII)
		}
		if err := w.Write(row); err != nil {
			return err
		}
	}

	return nil
}

// writeSliceSliceCSV writes a slice of slices to CSV
func writeSliceSliceCSV(w *csv.Writer, v reflect.Value, ensureASCII bool) error {
	for i := 0; i < v.Len(); i++ {
		row := v.Index(i)
		rowStrs := make([]string, row.Len())
		for j := 0; j < row.Len(); j++ {
			rowStrs[j] = formatValue(row.Index(j).Interface(), ensureASCII)
		}
		if err := w.Write(rowStrs); err != nil {
			return err
		}
	}
	return nil
}

// writePrimitiveSliceCSV writes a slice of primitives as a single column
func writePrimitiveSliceCSV(w *csv.Writer, v reflect.Value, ensureASCII bool) error {
	for i := 0; i < v.Len(); i++ {
		row := []string{formatValue(v.Index(i).Interface(), ensureASCII)}
		if err := w.Write(row); err != nil {
			return err
		}
	}
	return nil
}

// formatValue converts a value to its string representation for CSV
func formatValue(v interface{}, ensureASCII bool) string {
	if v == nil {
		return ""
	}

	var result string

	switch val := v.(type) {
	case string:
		result = val
	case int, int8, int16, int32, int64:
		return fmt.Sprintf("%d", val)
	case uint, uint8, uint16, uint32, uint64:
		return fmt.Sprintf("%d", val)
	case float32, float64:
		return strconv.FormatFloat(reflect.ValueOf(val).Float(), 'f', -1, 64)
	case bool:
		return strconv.FormatBool(val)
	case []string:
		// Handle string slices by taking the last (most specific) element
		// For hierarchical types like ["Entity", "ANATOMY"], we want just "ANATOMY"
		if len(val) > 0 {
			result = val[len(val)-1]
		} else {
			result = ""
		}
	case []interface{}:
		// Handle generic slices by taking the last element
		if len(val) > 0 {
			result = formatValue(val[len(val)-1], ensureASCII)
		} else {
			result = ""
		}
	default:
		// Check if it's a slice using reflection
		rv := reflect.ValueOf(v)
		if rv.Kind() == reflect.Slice || rv.Kind() == reflect.Array {
			// Take the last (most specific) element for hierarchical types
			if rv.Len() > 0 {
				result = formatValue(rv.Index(rv.Len()-1).Interface(), ensureASCII)
			} else {
				result = ""
			}
		} else {
			// For other complex types, use JSON representation
			b, err := json.Marshal(v)
			if err != nil {
				result = fmt.Sprint(v)
			} else {
				result = string(b)
			}
		}
	}

	if ensureASCII {
		return escapeNonASCII(result)
	}
	return result
}

// escapeNonASCII escapes non-ASCII characters in a string
func escapeNonASCII(s string) string {
	var buf strings.Builder
	for _, r := range s {
		if r > unicode.MaxASCII {
			fmt.Fprintf(&buf, "\\u%04x", r)
		} else {
			buf.WriteRune(r)
		}
	}
	return buf.String()
}

// logPrompts logs system and user prompts at debug level if a logger is available in context.
// This replaces the fmt.Printf statements throughout the prompts package.
// Prints with actual newlines preserved instead of escaped.
// Only prints if the context has "debug_prompts" set to true.
func logPrompts(logger *slog.Logger, sysPrompt, userPrompt string) {
	// Check if debug_prompts is enabled in context
	debugPrompts := false
	if os.Getenv("DEBUG_LLM_PROMPTS") == "true" {
		debugPrompts = true
	}

	if !debugPrompts {
		return
	}

	// Log with preserved newlines using structured format
	logger.Debug("Generated prompts - System Prompt follows")
	fmt.Println("=== SYSTEM PROMPT ===")
	fmt.Println(sysPrompt)
	logger.Debug("Generated prompts - User Prompt follows")
	fmt.Println("=== USER PROMPT ===")
	fmt.Println(userPrompt)
	fmt.Println("=== END PROMPTS ===")

}

func LogResponses(logger *slog.Logger, response types.Response) {
	debugPrompts := false
	if os.Getenv("DEBUG_LLM_PROMPTS") == "true" {
		debugPrompts = true
	}

	if !debugPrompts {
		return
	}

	// Log with preserved newlines using structured format
	logger.Debug("LLM response follows")
	fmt.Println("=== LLM response ===")
	fmt.Println(response.Content)
	fmt.Println("=== END LLM response ===")

}

// ToPromptYAML serializes data to YAML for use in prompts.
func ToPromptYAML(data interface{}) (string, error) {
	// Re-use gopkg.in/yaml.v3
	var buf bytes.Buffer
	enc := yaml.NewEncoder(&buf)
	enc.SetIndent(2)
	err := enc.Encode(data)
	if err != nil {
		return "", err
	}
	return buf.String(), nil
}

// ToPromptTOML serializes data to TOML for use in prompts.
func ToPromptTOML(data interface{}) (string, error) {
	b, err := toml.Marshal(data)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// PromptFormat defines the strategy for formatting prompt data.
type PromptFormat struct {
	Marshal     func(interface{}) (string, error) // Serialization function
	Name        string                            // e.g., "YAML", "TOML", "TSV"
	Description string                            // e.g., "provided in YAML format"
}

// GetPromptFormat determines the prompt format strategy from the context.
func GetPromptFormat(context map[string]interface{}) PromptFormat {
	if val, ok := context["use_yaml"]; ok {
		if b, ok := val.(bool); ok && b {
			return PromptFormat{
				Name:        "YAML",
				Description: "provided in YAML format",
				Marshal:     ToPromptYAML,
			}
		}
	}

	if val, ok := context["use_toml"]; ok {
		if b, ok := val.(bool); ok && b {
			return PromptFormat{
				Name:        "TOML",
				Description: "provided in TOML format",
				Marshal:     ToPromptTOML,
			}
		}
	}

	// Default to TSV
	ensureASCII := true
	if val, ok := context["ensure_ascii"]; ok {
		if b, ok := val.(bool); ok {
			ensureASCII = b
		}
	}

	return PromptFormat{
		Name:        "TSV",
		Description: "provided in TSV (tab-separated values) format",
		Marshal: func(v interface{}) (string, error) {
			return ToPromptCSV(v, ensureASCII)
		},
	}
}

// FormatContext serializes a map of data using the provided format.
// Returns a map of serialized strings and any error encountered.
func FormatContext[T any](format PromptFormat, data map[string]T) (map[string]string, error) {
	result := make(map[string]string, len(data))
	for k, v := range data {
		s, err := format.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal %s: %w", k, err)
		}
		result[k] = s
	}
	return result, nil
}
