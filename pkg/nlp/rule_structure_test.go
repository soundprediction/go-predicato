package nlp_test

import (
	"testing"

	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/ruleschema"
)

func structuredRule() *ruleschema.Rule {
	return &ruleschema.Rule{
		Conditions: []ruleschema.Pattern{
			{
				ID:        "c1",
				Subject:   ruleschema.Term{Var: "?x"},
				Predicate: "has_state",
				Object:    ruleschema.Term{Entity: "ready"},
			},
		},
		Consequent: ruleschema.Pattern{
			ID:        "k1",
			Subject:   ruleschema.Term{Var: "x"},
			Predicate: "can_start",
			Object:    ruleschema.Term{Entity: "workflow"},
		},
	}
}

func TestCompileRuleStructureParsed(t *testing.T) {
	rule := nlp.Rule{
		Antecedent:        "item is ready",
		Consequent:        "workflow can start",
		RuleType:          "policy",
		Scope:             "operations",
		SourceAttribution: "paragraph 2",
		Confidence:        0.8,
		Structured:        structuredRule(),
	}

	nlp.CompileRuleStructure(&rule, nlp.RuleStructureMetadata{
		ID:            "rule-1",
		SourceID:      "source-1",
		Model:         "model-a",
		ChunkIndex:    3,
		HasChunkIndex: true,
	})

	if rule.StructureStatus != ruleschema.StructureStatusParsed {
		t.Fatalf("StructureStatus = %q, want %q", rule.StructureStatus, ruleschema.StructureStatusParsed)
	}
	if rule.StructureError != "" {
		t.Fatalf("StructureError = %q, want empty", rule.StructureError)
	}
	if rule.Structured == nil {
		t.Fatal("Structured was cleared")
	}
	if rule.Structured.ID != "rule-1" {
		t.Fatalf("structured ID = %q, want rule-1", rule.Structured.ID)
	}
	if rule.Structured.Conditions[0].Subject.Var != "x" {
		t.Fatalf("condition variable = %q, want x", rule.Structured.Conditions[0].Subject.Var)
	}
	if rule.Structured.RuleType != "policy" {
		t.Fatalf("rule_type = %q, want policy", rule.Structured.RuleType)
	}
	if rule.Structured.Provenance.SourceID != "source-1" {
		t.Fatalf("source_id = %q, want source-1", rule.Structured.Provenance.SourceID)
	}
	if err := ruleschema.Validate(*rule.Structured); err != nil {
		t.Fatalf("structured rule did not validate: %v", err)
	}
}

func TestCompileRuleStructureTextOnlyFallback(t *testing.T) {
	rule := nlp.Rule{
		Antecedent: "item is ready",
		Consequent: "workflow can start",
	}

	nlp.CompileRuleStructure(&rule, nlp.RuleStructureMetadata{})

	if rule.StructureStatus != ruleschema.StructureStatusUnparsed {
		t.Fatalf("StructureStatus = %q, want %q", rule.StructureStatus, ruleschema.StructureStatusUnparsed)
	}
	if rule.Structured != nil {
		t.Fatal("Structured set for text-only rule")
	}
	if rule.Antecedent == "" || rule.Consequent == "" {
		t.Fatal("text rule fields were dropped")
	}
}

func TestCompileRuleStructureInvalid(t *testing.T) {
	invalid := structuredRule()
	invalid.Consequent.Subject = ruleschema.Term{Var: "missing"}
	rule := nlp.Rule{
		Antecedent: "item is ready",
		Consequent: "workflow can start",
		Structured: invalid,
	}

	nlp.CompileRuleStructure(&rule, nlp.RuleStructureMetadata{})

	if rule.StructureStatus != ruleschema.StructureStatusInvalid {
		t.Fatalf("StructureStatus = %q, want %q", rule.StructureStatus, ruleschema.StructureStatusInvalid)
	}
	if rule.StructureError == "" {
		t.Fatal("StructureError is empty for invalid structure")
	}
	if rule.Structured != nil {
		t.Fatal("Structured was not cleared for invalid structure")
	}
	if rule.Antecedent == "" || rule.Consequent == "" {
		t.Fatal("text rule fields were dropped")
	}
}

func TestCompileExtendedExtractionRuleStructures(t *testing.T) {
	result := &nlp.ExtendedExtractionResult{
		Rules: []nlp.Rule{
			{Antecedent: "if A", Consequent: "then B", Structured: structuredRule()},
			{Antecedent: "if C", Consequent: "then D"},
		},
	}

	nlp.CompileExtendedExtractionRuleStructures(result)

	if result.Rules[0].StructureStatus != ruleschema.StructureStatusParsed {
		t.Fatalf("structured rule status = %q", result.Rules[0].StructureStatus)
	}
	if result.Rules[1].StructureStatus != ruleschema.StructureStatusUnparsed {
		t.Fatalf("text-only rule status = %q", result.Rules[1].StructureStatus)
	}
}
