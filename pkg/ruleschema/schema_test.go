package ruleschema

import (
	"encoding/json"
	"testing"
)

func validRule() Rule {
	return Rule{
		ID:         "r1",
		RuleType:   "policy",
		Scope:      "general",
		Confidence: 0.9,
		Conditions: []Pattern{
			{
				ID:        "c1",
				Subject:   Term{Var: "x"},
				Predicate: "has_status",
				Object:    Term{Entity: "active", EntityType: "status"},
			},
		},
		Consequent: Pattern{
			ID:        "k1",
			Subject:   Term{Var: "x"},
			Predicate: "receives",
			Object:    Term{Entity: "standard review"},
		},
		Exceptions: []Pattern{
			{
				ID:        "e1",
				Subject:   Term{Var: "x"},
				Predicate: "blocked_by",
				Object:    Term{Entity: "manual hold"},
				Negated:   true,
			},
		},
		Provenance: Provenance{
			SourceID:          "source-1",
			ChunkIndex:        2,
			Model:             "model-a",
			SourceAttribution: "section 1",
		},
	}
}

func TestValidateRangeRestrictionPass(t *testing.T) {
	rule := validRule()
	if err := Validate(rule); err != nil {
		t.Fatalf("Validate() returned error: %v", err)
	}
}

func TestValidateRangeRestrictionFail(t *testing.T) {
	rule := validRule()
	rule.Consequent.Subject = Term{Var: "y"}
	if err := Validate(rule); err == nil {
		t.Fatal("Validate() succeeded for consequent variable absent from conditions")
	}
}

func TestValidateExceptionRangeRestrictionFail(t *testing.T) {
	rule := validRule()
	rule.Exceptions[0].Object = Term{Var: "missing"}
	if err := Validate(rule); err == nil {
		t.Fatal("Validate() succeeded for exception variable absent from conditions")
	}
}

func TestValidateTermExclusivity(t *testing.T) {
	tests := []struct {
		name string
		term Term
	}{
		{name: "both var and entity", term: Term{Var: "x", Entity: "thing"}},
		{name: "neither var nor entity", term: Term{}},
		{name: "var with entity metadata", term: Term{Var: "x", EntityUUID: "uuid-1"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rule := validRule()
			rule.Conditions[0].Subject = tt.term
			if err := Validate(rule); err == nil {
				t.Fatal("Validate() succeeded for invalid term")
			}
		})
	}
}

func TestJSONRoundTripDefaultsSchemaVersion(t *testing.T) {
	rule := validRule()
	rule.SchemaVersion = 0
	rule.Conditions[0].Subject.Var = "?x"

	data, err := json.Marshal(rule)
	if err != nil {
		t.Fatalf("Marshal() error: %v", err)
	}

	var decoded Rule
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal() error: %v", err)
	}
	if decoded.SchemaVersion != CurrentSchemaVersion {
		t.Fatalf("schema version = %d, want %d", decoded.SchemaVersion, CurrentSchemaVersion)
	}
	if decoded.Conditions[0].Subject.Var != "x" {
		t.Fatalf("variable = %q, want %q", decoded.Conditions[0].Subject.Var, "x")
	}
	if err := Validate(decoded); err != nil {
		t.Fatalf("decoded rule did not validate: %v", err)
	}
}

func TestStructuredRuleAttributeRoundTrip(t *testing.T) {
	rule := validRule()
	structured, err := json.Marshal(rule)
	if err != nil {
		t.Fatalf("Marshal(rule) error: %v", err)
	}

	attributes := map[string]any{
		"structured_rule":  string(structured),
		"structure_status": StructureStatusParsed,
	}
	encodedAttributes, err := json.Marshal(attributes)
	if err != nil {
		t.Fatalf("Marshal(attributes) error: %v", err)
	}

	var decodedAttributes map[string]string
	if err := json.Unmarshal(encodedAttributes, &decodedAttributes); err != nil {
		t.Fatalf("Unmarshal(attributes) error: %v", err)
	}
	if decodedAttributes["structure_status"] != StructureStatusParsed {
		t.Fatalf("structure_status = %q, want %q", decodedAttributes["structure_status"], StructureStatusParsed)
	}

	var decodedRule Rule
	if err := json.Unmarshal([]byte(decodedAttributes["structured_rule"]), &decodedRule); err != nil {
		t.Fatalf("Unmarshal(structured_rule) error: %v", err)
	}
	if err := Validate(decodedRule); err != nil {
		t.Fatalf("decoded structured_rule did not validate: %v", err)
	}
}
