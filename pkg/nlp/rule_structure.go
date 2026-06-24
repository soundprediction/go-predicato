package nlp

import "github.com/soundprediction/predicato/pkg/ruleschema"

// RuleStructureMetadata contains provenance known outside the LLM response.
type RuleStructureMetadata struct {
	ID            string
	SourceID      string
	Model         string
	ChunkIndex    int
	HasChunkIndex bool
}

// CompileRuleStructure validates a rule's optional structured form.
//
// Text-only rules are preserved with structure_status="unparsed". If structured
// patterns are present but invalid, Structured is cleared and the text rule is
// preserved with structure_status="invalid".
func CompileRuleStructure(rule *Rule, meta RuleStructureMetadata) {
	if rule == nil {
		return
	}
	if rule.Structured == nil {
		rule.StructureStatus = ruleschema.StructureStatusUnparsed
		rule.StructureError = ""
		return
	}

	structured := *rule.Structured
	applyRuleStructureMetadata(&structured, *rule, meta)
	ruleschema.Normalize(&structured)
	if err := ruleschema.Validate(structured); err != nil {
		rule.Structured = nil
		rule.StructureStatus = ruleschema.StructureStatusInvalid
		rule.StructureError = err.Error()
		return
	}

	rule.Structured = &structured
	rule.StructureStatus = ruleschema.StructureStatusParsed
	rule.StructureError = ""
}

// CompileExtendedExtractionRuleStructures validates all optional structured
// rules in an extraction result.
func CompileExtendedExtractionRuleStructures(result *ExtendedExtractionResult) {
	if result == nil {
		return
	}
	for i := range result.Rules {
		CompileRuleStructure(&result.Rules[i], RuleStructureMetadata{})
	}
	for i := range result.InferredRules {
		CompileRuleStructure(&result.InferredRules[i], RuleStructureMetadata{})
	}
}

func applyRuleStructureMetadata(dst *ruleschema.Rule, src Rule, meta RuleStructureMetadata) {
	if meta.ID != "" {
		dst.ID = meta.ID
	}
	if dst.RuleType == "" {
		dst.RuleType = src.RuleType
	}
	if dst.Scope == "" {
		dst.Scope = src.Scope
	}
	if dst.Confidence == 0 && src.Confidence != 0 {
		dst.Confidence = src.Confidence
	}
	if meta.SourceID != "" {
		dst.Provenance.SourceID = meta.SourceID
	}
	if meta.Model != "" {
		dst.Provenance.Model = meta.Model
	}
	if meta.HasChunkIndex {
		dst.Provenance.ChunkIndex = meta.ChunkIndex
	}
	if dst.Provenance.SourceAttribution == "" {
		dst.Provenance.SourceAttribution = src.SourceAttribution
	}
}
