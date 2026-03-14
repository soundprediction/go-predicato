package types

import (
	"testing"
)

func TestIsTautological(t *testing.T) {
	tests := []struct {
		src, tgt string
		want     bool
	}{
		{"Kidney failure", "Kidney", true},
		{"Allergy medication drugs", "Allergy medication", true},
		{"Poor drug metabolism", "Drug Metabolism", true},
		{"heart attack", "myocardial infarction", false},
		{"aspirin", "headache", false},
		{"", "something", false},
		{"something", "", false},
	}

	for _, tt := range tests {
		got := IsTautological(tt.src, tt.tgt)
		if got != tt.want {
			t.Errorf("IsTautological(%q, %q) = %v, want %v", tt.src, tt.tgt, got, tt.want)
		}
	}
}

func TestIsGenericTerm(t *testing.T) {
	tests := []struct {
		s    string
		want bool
	}{
		{"Homo Sapiens", true},
		{"human", true},
		{"Disease", true},
		{"clinical trial", true},
		{"aspirin", false},
		{"myocardial infarction", false},
		{"", false},
	}

	for _, tt := range tests {
		got := IsGenericTerm(tt.s)
		if got != tt.want {
			t.Errorf("IsGenericTerm(%q) = %v, want %v", tt.s, got, tt.want)
		}
	}
}

func TestFilterLowValueTriples(t *testing.T) {
	triples := []*ExtractedTriple{
		{Subject: "aspirin", Predicate: "TREATS", Object: "headache"},
		{Subject: "Kidney failure", Predicate: "AFFECTS", Object: "Kidney"},           // tautological
		{Subject: "cancer", Predicate: "AFFECTS", Object: "Homo Sapiens"},             // generic target
		{Subject: "aspirin", Predicate: "TREATS", Object: "headache"},                 // duplicate
		{Subject: "ibuprofen", Predicate: "IS_A", Object: "anti-inflammatory drug"},   // valid IS_A
		{Subject: "Drug Metabolism", Predicate: "AFFECTS", Object: "Drug metabolism"}, // tautological
	}

	kept, removed := FilterLowValueTriples(triples)

	if len(kept) != 2 {
		t.Errorf("expected 2 kept triples, got %d", len(kept))
		for _, k := range kept {
			t.Logf("  kept: %s %s %s", k.Subject, k.Predicate, k.Object)
		}
	}
	if len(removed) != 4 {
		t.Errorf("expected 4 removed triples, got %d", len(removed))
		for _, r := range removed {
			t.Logf("  removed: %s %s %s", r.Subject, r.Predicate, r.Object)
		}
	}
}

func TestIsClinicalMetadataEdge(t *testing.T) {
	tests := []struct {
		name string
		edge string
		src  string
		tgt  string
		want bool
	}{
		{"HAS_CONDITION from clinical trial", "HAS_CONDITION", "Clinical Trial NCT001234", "diabetes", true},
		{"HAS_CONDITION from clinical study", "HAS_CONDITION", "Phase 3 Clinical Study", "hypertension", true},
		{"HAS_CONDITION non-trial source", "HAS_CONDITION", "metformin", "diabetes", false},
		{"Different predicate from trial", "TREATS", "clinical trial NCT999", "cancer", false},
	}

	for _, tt := range tests {
		got := IsClinicalMetadataEdge(tt.edge, tt.src, tt.tgt)
		if got != tt.want {
			t.Errorf("IsClinicalMetadataEdge(%q, %q, %q) = %v, want %v",
				tt.edge, tt.src, tt.tgt, got, tt.want)
		}
	}
}

func TestFilterLowValueTriples_ClinicalMetadata(t *testing.T) {
	triples := []*ExtractedTriple{
		{Subject: "aspirin", Predicate: "TREATS", Object: "headache"},                               // keep
		{Subject: "Clinical Trial NCT001234", Predicate: "HAS_CONDITION", Object: "diabetes"},       // clinical metadata
		{Subject: "Phase 2 Clinical Study XYZ", Predicate: "HAS_CONDITION", Object: "hypertension"}, // clinical metadata
	}

	kept, removed := FilterLowValueTriples(triples)

	if len(kept) != 1 {
		t.Errorf("expected 1 kept, got %d", len(kept))
		for _, k := range kept {
			t.Logf("  kept: %s %s %s", k.Subject, k.Predicate, k.Object)
		}
	}
	if len(removed) != 2 {
		t.Errorf("expected 2 removed, got %d", len(removed))
	}
}

func TestFilterLowValueEdges(t *testing.T) {
	edges := []*Edge{
		{Name: "TREATS", Fact: "aspirin is used to treat headache"},
		{Name: "AFFECTS", Fact: "kidney failure affects Kidney"},    // tautological
		{Name: "AFFECTS", Fact: "cancer affects Homo Sapiens"},      // generic
		{Name: "IS_A", Fact: "ibuprofen is a type of NSAID"},        // valid
		{Name: "TREATS", Fact: "aspirin is used to treat headache"}, // duplicate
	}

	result := FilterLowValueEdges(edges)

	if len(result) != 2 {
		t.Errorf("expected 2 edges after filtering, got %d", len(result))
		for _, e := range result {
			t.Logf("  kept: [%s] %s", e.Name, e.Fact)
		}
	}
}
