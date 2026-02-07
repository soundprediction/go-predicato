package utils

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/soundprediction/predicato/pkg/driver"
	"gopkg.in/yaml.v3"
)

const (
	DefaultPageLimit              = 20
	DefaultSemaphoreLimit         = 20
	DefaultMaxReflexionIterations = 0
)

var (
	// ErrInvalidGroupID is returned when a group ID contains invalid characters
	ErrInvalidGroupID = errors.New("group ID contains invalid characters")
	// ErrInvalidEntityType is returned when an entity type is invalid
	ErrInvalidEntityType = errors.New("invalid entity type")
)

// CleanString removes extraneous spaces and line breaks from a string.
// It trims whitespace from the ends and replaces multiple whitespace/newline sequences with a single space.
func CleanString(s string) string {
	// Replace all newlines and tabs with spaces
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\t", " ")
	s = strings.ReplaceAll(s, "\r", " ")
	
	// Collapse multiple spaces into one
	// Using a loop is simple and effective for this without regex overhead
	for strings.Contains(s, "  ") {
		s = strings.ReplaceAll(s, "  ", " ")
	}
	
	return strings.TrimSpace(s)
}

// GetUseParallelRuntime returns whether to use parallel runtime based on environment variable
func GetUseParallelRuntime() bool {
	val := os.Getenv("USE_PARALLEL_RUNTIME")
	if val == "" {
		return false
	}
	useParallel, _ := strconv.ParseBool(val)
	return useParallel
}

// GetSemaphoreLimit returns the semaphore limit from environment variable or default
func GetSemaphoreLimit() int {
	val := os.Getenv("SEMAPHORE_LIMIT")
	if val == "" {
		return DefaultSemaphoreLimit
	}
	limit, err := strconv.Atoi(val)
	if err != nil {
		return DefaultSemaphoreLimit
	}
	return limit
}

// GetMaxReflexionIterations returns the max reflexion iterations from environment variable or default
func GetMaxReflexionIterations() int {
	val := os.Getenv("MAX_REFLEXION_ITERATIONS")
	if val == "" {
		return DefaultMaxReflexionIterations
	}
	iterations, err := strconv.Atoi(val)
	if err != nil {
		return DefaultMaxReflexionIterations
	}
	return iterations
}

// ParseDBDate parses various date formats from database responses
func ParseDBDate(inputDate interface{}) (*time.Time, error) {
	switch v := inputDate.(type) {
	case time.Time:
		return &v, nil
	case string:
		if v == "" {
			return nil, nil
		}
		parsed, err := time.Parse(time.RFC3339, v)
		if err != nil {
			// Try parsing ISO format without timezone
			parsed, err = time.Parse("2006-01-02T15:04:05", v)
			if err != nil {
				return nil, fmt.Errorf("failed to parse date string %q: %w", v, err)
			}
		}
		return &parsed, nil
	case nil:
		return nil, nil
	default:
		return nil, fmt.Errorf("unsupported date type: %T", v)
	}
}

// GetDefaultGroupID differentiates the default group id based on the database type
func GetDefaultGroupID(provider driver.GraphProvider) string {
	if provider == driver.GraphProviderFalkorDB {
		return "_"
	}
	return ""
}

// LuceneSanitize escapes special characters from a query before passing into Lucene
func LuceneSanitize(query string) string {
	// Escape special characters: + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
	replacer := strings.NewReplacer(
		"+", `\+`,
		"-", `\-`,
		"&", `\&`,
		"|", `\|`,
		"!", `\!`,
		"(", `\(`,
		")", `\)`,
		"{", `\{`,
		"}", `\}`,
		"[", `\[`,
		"]", `\]`,
		"^", `\^`,
		"\"", `\"`,
		"~", `\~`,
		"*", `\*`,
		"?", `\?`,
		":", `\:`,
		"\\", `\\`,
		"/", `\/`,
		"O", `\O`,
		"R", `\R`,
		"N", `\N`,
		"T", `\T`,
		"A", `\A`,
		"D", `\D`,
	)
	return replacer.Replace(query)
}

// NormalizeL2 normalizes a vector using L2 normalization
func NormalizeL2(embedding []float64) []float64 {
	if len(embedding) == 0 {
		return embedding
	}

	// Calculate the L2 norm
	var norm float64
	for _, val := range embedding {
		norm += val * val
	}
	norm = math.Sqrt(norm)

	// Avoid division by zero
	if norm == 0 {
		return embedding
	}

	// Normalize the vector
	normalized := make([]float64, len(embedding))
	for i, val := range embedding {
		normalized[i] = val / norm
	}

	return normalized
}

// NormalizeL2Float32 normalizes a float32 vector using L2 normalization
func NormalizeL2Float32(embedding []float32) []float32 {
	if len(embedding) == 0 {
		return embedding
	}

	// Calculate the L2 norm
	var norm float32
	for _, val := range embedding {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))

	// Avoid division by zero
	if norm == 0 {
		return embedding
	}

	// Normalize the vector
	normalized := make([]float32, len(embedding))
	for i, val := range embedding {
		normalized[i] = val / norm
	}

	return normalized
}

// ValidateGroupID validates that a group_id contains only ASCII alphanumeric characters, dashes, and underscores
func ValidateGroupID(groupID string) error {
	// Allow empty string (default case)
	if groupID == "" {
		return nil
	}

	// Check if string contains only ASCII alphanumeric characters, dashes, or underscores
	// Pattern matches: letters (a-z, A-Z), digits (0-9), hyphens (-), and underscores (_)
	matched, err := regexp.MatchString(`^[a-zA-Z0-9_-]+$`, groupID)
	if err != nil {
		return fmt.Errorf("failed to validate group ID: %w", err)
	}

	if !matched {
		return fmt.Errorf("%w: group ID %q contains invalid characters", ErrInvalidGroupID, groupID)
	}

	return nil
}

// ValidateExcludedEntityTypes validates that excluded entity types are valid type names
func ValidateExcludedEntityTypes(excludedEntityTypes []string, availableTypes []string) error {
	if len(excludedEntityTypes) == 0 {
		return nil
	}

	// Build set of available type names
	availableSet := make(map[string]bool)
	availableSet["Entity"] = true // Default type is always available
	for _, t := range availableTypes {
		availableSet[t] = true
	}

	// Check for invalid type names
	var invalidTypes []string
	for _, excludedType := range excludedEntityTypes {
		if !availableSet[excludedType] {
			invalidTypes = append(invalidTypes, excludedType)
		}
	}

	if len(invalidTypes) > 0 {
		availableList := make([]string, 0, len(availableSet))
		for t := range availableSet {
			availableList = append(availableList, t)
		}
		return fmt.Errorf("%w: invalid excluded entity types: %v, available types: %v",
			ErrInvalidEntityType, invalidTypes, availableList)
	}

	return nil
}

// GenerateUUID generates a new UUID7 string
func GenerateUUID() string {
	return uuid.Must(uuid.NewV7()).String()
}

// removeLastLine takes a string and returns a new string with the
// last line of text removed.
func RemoveLastLine(s string) string {
	// Find the index of the last newline character.
	lastNewline := strings.LastIndex(s, "\n")

	// If no newline is found, the string is either a single line or empty.
	// In either case, removing the last line results in an empty string.
	if lastNewline == -1 {
		return ""
	}

	// Return the substring from the beginning up to the last newline.
	// This effectively cuts off the text that follows it.
	return s[:lastNewline]
}

// UnmarshalCSV parses a CSV string and unmarshals it into a slice of structs.
// It uses standard encoding/csv with error recovery.
func UnmarshalCSV[T any](csvString string, delimiter rune) ([]*T, error) {
	reader := csv.NewReader(strings.NewReader(csvString))
	reader.Comma = delimiter
	reader.LazyQuotes = true

	// Read header
	header, err := reader.Read()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV header: %w", err)
	}

	results := make([]*T, 0)
	structType := reflect.TypeOf(new(T)).Elem()

	// Map headers to fields
	fieldMap := make(map[string]int)
	for i := 0; i < structType.NumField(); i++ {
		field := structType.Field(i)
		csvTag := field.Tag.Get("csv")
		if csvTag != "" && csvTag != "-" {
			fieldMap[csvTag] = i
		} else {
			fieldMap[strings.ToLower(field.Name)] = i
		}
	}

	// Read all records
	records, err := reader.ReadAll()
	if err != nil {
		// encoding/csv often returns partial results and an error with ParseError
		// But if we want "ignore_errors" behavior like DuckDB, we might need to read row by row.
		// For now, let's assume we want to fail on bad CSVs unless we implement row-by-row skipping.
		// Let's retry row-by-row to skip bad ones.
		reader = csv.NewReader(strings.NewReader(csvString))
		reader.Comma = delimiter
		reader.LazyQuotes = true
		_, _ = reader.Read() // skip header again
	} else {
		// Process pre-read records with pre-allocated capacity
		results = make([]*T, 0, len(records))
		for _, record := range records {
			if len(record) != len(header) {
				continue
			}
			newStruct, err := mapRowToStruct[T](record, header, fieldMap, structType)
			if err != nil {
				fmt.Printf("Warning: failed to map row: %v\n", err)
				continue
			}
			results = append(results, newStruct)
		}
		return results, nil
	}

	// Row-by-row fallback
	reader = csv.NewReader(strings.NewReader(csvString))
	reader.Comma = delimiter
	reader.LazyQuotes = true
	_, _ = reader.Read() // skip header

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			fmt.Printf("Warning: skipping bad CSV row: %v\n", err)
			continue
		}

		newStruct, err := mapRowToStruct[T](record, header, fieldMap, structType)
		if err != nil {
			fmt.Printf("Warning: failed to map row: %v\n", err)
			continue
		}
		results = append(results, newStruct)
	}

	return results, nil
}

func mapRowToStruct[T any](record []string, header []string, fieldMap map[string]int, structType reflect.Type) (*T, error) {
	newStructPtr := reflect.New(structType)
	newStruct := newStructPtr.Elem()

	for i, colName := range header {
		if i >= len(record) {
			break
		}
		val := record[i]

		if fieldIdx, ok := fieldMap[colName]; ok {
			if err := setField(newStruct.Field(fieldIdx), val); err != nil {
				return nil, err
			}
			continue
		}
		if fieldIdx, ok := fieldMap[strings.ToLower(colName)]; ok {
			if err := setField(newStruct.Field(fieldIdx), val); err != nil {
				return nil, err
			}
		}
	}
	return newStructPtr.Interface().(*T), nil
}

// setField is a helper that converts a string value and sets it on a reflect.Value field.
func setField(field reflect.Value, value string) error {
	if !field.CanSet() {
		return errors.New("field cannot be set")
	}

	// Handle pointers by dereferencing to the underlying type
	if field.Kind() == reflect.Ptr {
		if field.IsNil() {
			field.Set(reflect.New(field.Type().Elem()))
		}
		field = field.Elem()
	}

	switch field.Kind() {
	case reflect.String:
		field.SetString(value)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		i, err := strconv.ParseInt(value, 10, 64)
		if err != nil {
			return err
		}
		if field.OverflowInt(i) {
			return fmt.Errorf("int overflow for value %s", value)
		}
		field.SetInt(i)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		u, err := strconv.ParseUint(value, 10, 64)
		if err != nil {
			return err
		}
		if field.OverflowUint(u) {
			return fmt.Errorf("uint overflow for value %s", value)
		}
		field.SetUint(u)
	case reflect.Float32, reflect.Float64:
		f, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return err
		}
		if field.OverflowFloat(f) {
			return fmt.Errorf("float overflow for value %s", value)
		}
		field.SetFloat(f)
	case reflect.Bool:
		b, err := strconv.ParseBool(strings.ToLower(value))
		if err != nil {
			return err
		}
		field.SetBool(b)
	case reflect.Slice:
		// Handle slice types (e.g., []string)
		// Check for empty array notation
		trimmed := strings.TrimSpace(value)
		if trimmed == "[]" || trimmed == "" {
			// Set to empty slice
			field.Set(reflect.MakeSlice(field.Type(), 0, 0))
			return nil
		}

		// For non-empty arrays, parse based on element type
		elemType := field.Type().Elem()

		// Remove brackets if present
		if strings.HasPrefix(trimmed, "[") && strings.HasSuffix(trimmed, "]") {
			trimmed = trimmed[1 : len(trimmed)-1]
		}

		// Split by comma
		if trimmed == "" {
			field.Set(reflect.MakeSlice(field.Type(), 0, 0))
			return nil
		}

		parts := strings.Split(trimmed, ",")
		slice := reflect.MakeSlice(field.Type(), len(parts), len(parts))

		for i, part := range parts {
			part = strings.TrimSpace(part)
			// Remove quotes if present
			part = strings.Trim(part, "\"'")

			elem := slice.Index(i)
			if err := setSliceElement(elem, part, elemType); err != nil {
				return fmt.Errorf("failed to set slice element %d: %w", i, err)
			}
		}

		field.Set(slice)
	default:
		return fmt.Errorf("unsupported field type: %s", field.Kind())
	}
	return nil
}

// setSliceElement sets a single element in a slice based on its type
func setSliceElement(elem reflect.Value, value string, elemType reflect.Type) error {
	switch elemType.Kind() {
	case reflect.String:
		elem.SetString(value)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		i, err := strconv.ParseInt(value, 10, 64)
		if err != nil {
			return err
		}
		elem.SetInt(i)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		u, err := strconv.ParseUint(value, 10, 64)
		if err != nil {
			return err
		}
		elem.SetUint(u)
	case reflect.Float32, reflect.Float64:
		f, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return err
		}
		elem.SetFloat(f)
	case reflect.Bool:
		b, err := strconv.ParseBool(strings.ToLower(value))
		if err != nil {
			return err
		}
		elem.SetBool(b)
	default:
		return fmt.Errorf("unsupported slice element type: %s", elemType.Kind())
	}
	return nil
}

func IsLastLineEmpty(text string) bool {
	lines := strings.Split(text, "\n")

	// Handle empty text
	if len(lines) == 0 {
		return true // Or false, depending on your definition of "empty last line" for empty text
	}

	// Get the last line (which might be an empty string if the text ends with a newline)
	lastLine := lines[len(lines)-1]

	// Trim whitespace and check if it's empty
	return strings.TrimSpace(lastLine) == ""
}

// UnmarshalYAML parses a YAML string and unmarshals it into a slice of structs.
// It uses gopkg.in/yaml.v3 and handles partial failures by skipping invalid items.
func UnmarshalYAML[T any](yamlString string) ([]*T, error) {
	// First, try to unmarshal as a slice of yaml.Nodes to access individual items
	var nodes []yaml.Node
	err := yaml.Unmarshal([]byte(yamlString), &nodes)
	if err != nil {
		// Fallback: If it's not a list, try generic unmarshal to see if it's a single item not wrapped in list ??
		// Or if the outer structure is fundamentally broken, we can't do much.
		return nil, fmt.Errorf("failed to parse YAML structure: %w", err)
	}

	results := make([]*T, 0, len(nodes))
	var errors []error

	for i, node := range nodes {
		var item T
		// Decode individual node
		if err := node.Decode(&item); err != nil {
			// Log error but continue
			errors = append(errors, fmt.Errorf("failed to unmarshal item %d: %v", i, err))
			continue
		}
		results = append(results, &item)
	}

	if len(results) == 0 && len(errors) > 0 {
		// If ALL items failed, return error
		return nil, fmt.Errorf("failed to unmarshal any items: %v", errors[0])
	}

	if len(errors) > 0 {
		// Log errors for partial failures (using standard log or just printing since we don't have logger here)
		// Ideally we'd accept a logger, but for a utility helper, fmt.Printf to stderr or just ignoring is common.
		// Given this is for LLM resilience, silent partial success is often desired, but let's print a warning.
		fmt.Fprintf(os.Stderr, "Warning: %d YAML items failed to parse and were skipped\n", len(errors))
	}

	return results, nil
}
