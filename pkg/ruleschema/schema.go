package ruleschema

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const (
	// CurrentSchemaVersion is the rule contract version emitted by predicato.
	CurrentSchemaVersion = 1

	// StructureStatusParsed means a structured rule was present and validated.
	StructureStatusParsed = "parsed"
	// StructureStatusUnparsed means the text rule had no structured patterns.
	StructureStatusUnparsed = "unparsed"
	// StructureStatusInvalid means structured patterns were present but invalid.
	StructureStatusInvalid = "invalid"
)

// Term is either a variable or a constant entity reference.
type Term struct {
	Var        string `json:"var,omitempty"`
	Entity     string `json:"entity,omitempty"`
	EntityUUID string `json:"entity_uuid,omitempty"`
	EntityType string `json:"entity_type,omitempty"`
}

// Pattern is a subject-predicate-object clause in a conditional rule.
type Pattern struct {
	ID                 string `json:"id"`
	Subject            Term   `json:"subject"`
	Predicate          string `json:"predicate"`
	PredicateCanonical string `json:"predicate_canonical,omitempty"`
	Object             Term   `json:"object"`
	Negated            bool   `json:"negated,omitempty"`
}

// Provenance captures source metadata for an extracted structured rule.
type Provenance struct {
	SourceID          string `json:"source_id"`
	ChunkIndex        int    `json:"chunk_index"`
	Model             string `json:"model"`
	SourceAttribution string `json:"source_attribution"`
}

// Rule is the versioned, domain-agnostic JSON contract for conditional rules.
type Rule struct {
	SchemaVersion int        `json:"schema_version"`
	ID            string     `json:"id"`
	RuleType      string     `json:"rule_type"`
	Scope         string     `json:"scope"`
	Confidence    float64    `json:"confidence"`
	Conditions    []Pattern  `json:"conditions"`
	Consequent    Pattern    `json:"consequent"`
	Exceptions    []Pattern  `json:"exceptions"`
	Provenance    Provenance `json:"provenance"`
}

// MarshalJSON emits the current schema version when SchemaVersion is unset.
func (r Rule) MarshalJSON() ([]byte, error) {
	type alias Rule
	Normalize(&r)
	return json.Marshal(alias(r))
}

// UnmarshalJSON accepts missing schema_version and defaults it to the current version.
func (r *Rule) UnmarshalJSON(data []byte) error {
	type alias Rule
	var decoded alias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*r = Rule(decoded)
	Normalize(r)
	return nil
}

// Normalize fills default fields and canonicalizes whitespace/variable spelling.
func Normalize(r *Rule) {
	if r == nil {
		return
	}
	if r.SchemaVersion == 0 {
		r.SchemaVersion = CurrentSchemaVersion
	}
	r.ID = strings.TrimSpace(r.ID)
	r.RuleType = strings.TrimSpace(r.RuleType)
	r.Scope = strings.TrimSpace(r.Scope)
	r.Provenance.SourceID = strings.TrimSpace(r.Provenance.SourceID)
	r.Provenance.Model = strings.TrimSpace(r.Provenance.Model)
	r.Provenance.SourceAttribution = strings.TrimSpace(r.Provenance.SourceAttribution)
	for i := range r.Conditions {
		normalizePattern(&r.Conditions[i])
	}
	normalizePattern(&r.Consequent)
	for i := range r.Exceptions {
		normalizePattern(&r.Exceptions[i])
	}
}

// Validate enforces the structural invariants for a rule.
func Validate(r Rule) error {
	Normalize(&r)
	if r.SchemaVersion != CurrentSchemaVersion {
		return fmt.Errorf("unsupported schema_version %d", r.SchemaVersion)
	}
	if len(r.Conditions) == 0 {
		return errors.New("rule must include at least one condition")
	}

	conditionVars := make(map[string]struct{})
	for i, condition := range r.Conditions {
		path := fmt.Sprintf("conditions[%d]", i)
		if err := validatePattern(condition, path); err != nil {
			return err
		}
		addPatternVars(condition, conditionVars)
	}

	if err := validatePattern(r.Consequent, "consequent"); err != nil {
		return err
	}
	if err := validateRangeRestricted(r.Consequent, conditionVars, "consequent"); err != nil {
		return err
	}

	for i, exception := range r.Exceptions {
		path := fmt.Sprintf("exceptions[%d]", i)
		if err := validatePattern(exception, path); err != nil {
			return err
		}
		if err := validateRangeRestricted(exception, conditionVars, path); err != nil {
			return err
		}
	}

	return nil
}

func normalizePattern(p *Pattern) {
	p.ID = strings.TrimSpace(p.ID)
	p.Predicate = strings.TrimSpace(p.Predicate)
	p.PredicateCanonical = strings.TrimSpace(p.PredicateCanonical)
	normalizeTerm(&p.Subject)
	normalizeTerm(&p.Object)
}

func normalizeTerm(t *Term) {
	t.Var = normalizeVariableName(t.Var)
	t.Entity = strings.TrimSpace(t.Entity)
	t.EntityUUID = strings.TrimSpace(t.EntityUUID)
	t.EntityType = strings.TrimSpace(t.EntityType)
}

func normalizeVariableName(v string) string {
	v = strings.TrimSpace(v)
	v = strings.TrimPrefix(v, "?")
	return strings.TrimSpace(v)
}

func validatePattern(p Pattern, path string) error {
	if strings.TrimSpace(p.Predicate) == "" {
		return fmt.Errorf("%s.predicate must be non-empty", path)
	}
	if err := validateTerm(p.Subject, path+".subject"); err != nil {
		return err
	}
	if err := validateTerm(p.Object, path+".object"); err != nil {
		return err
	}
	return nil
}

func validateTerm(t Term, path string) error {
	hasVar := strings.TrimSpace(t.Var) != ""
	hasEntity := strings.TrimSpace(t.Entity) != ""
	switch {
	case hasVar && hasEntity:
		return fmt.Errorf("%s must set exactly one of var or entity", path)
	case !hasVar && !hasEntity:
		return fmt.Errorf("%s must set exactly one of var or entity", path)
	case hasVar && (strings.TrimSpace(t.EntityUUID) != "" || strings.TrimSpace(t.EntityType) != ""):
		return fmt.Errorf("%s entity_uuid/entity_type require entity", path)
	default:
		return nil
	}
}

func addPatternVars(p Pattern, vars map[string]struct{}) {
	if v, ok := variableName(p.Subject); ok {
		vars[v] = struct{}{}
	}
	if v, ok := variableName(p.Object); ok {
		vars[v] = struct{}{}
	}
}

func validateRangeRestricted(p Pattern, conditionVars map[string]struct{}, path string) error {
	for termPath, term := range map[string]Term{
		path + ".subject": p.Subject,
		path + ".object":  p.Object,
	} {
		v, ok := variableName(term)
		if !ok {
			continue
		}
		if _, found := conditionVars[v]; !found {
			return fmt.Errorf("%s variable %q must appear in at least one condition", termPath, v)
		}
	}
	return nil
}

func variableName(t Term) (string, bool) {
	v := normalizeVariableName(t.Var)
	return v, v != ""
}
