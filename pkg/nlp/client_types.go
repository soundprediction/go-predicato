package nlp

// ExtractedEntity represents an entity extracted from text by an NLP model.
type ExtractedEntity struct {
	Text       string  `json:"text"`
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence"`
	Start      int     `json:"start"`
	End        int     `json:"end"`
}

// ExtractedRelation represents a relationship extracted from text between two entities.
type ExtractedRelation struct {
	Source     string  `json:"source"`
	Target     string  `json:"target"`
	Type       string  `json:"type"`
	Confidence float64 `json:"confidence"`
}

// ExtendedTriple represents a predicate triple with contextual extensions.
type ExtendedTriple struct {
	Subject           string  `json:"subject"`
	Predicate         string  `json:"predicate"`
	Object            string  `json:"object"`
	Condition         string  `json:"condition,omitempty"`
	Temporal          string  `json:"temporal,omitempty"`
	Location          string  `json:"location,omitempty"`
	Certainty         string  `json:"certainty,omitempty"`
	Scope             string  `json:"scope,omitempty"`
	SourceAttribution string  `json:"source_attribution,omitempty"`
	Confidence        float64 `json:"confidence,omitempty"`
}

// Rule represents a conditional rule: IF antecedent THEN consequent [UNLESS exception].
type Rule struct {
	Antecedent        string  `json:"antecedent"`
	Consequent        string  `json:"consequent"`
	Exception         string  `json:"exception,omitempty"`
	RuleType          string  `json:"rule_type,omitempty"`
	Scope             string  `json:"scope,omitempty"`
	SourceAttribution string  `json:"source_attribution,omitempty"`
	Confidence        float64 `json:"confidence,omitempty"`
}

// ExtendedExtractionResult represents the full extraction output including extended triples and rules.
type ExtendedExtractionResult struct {
	SourceText string              `json:"source_text"`
	Entities   map[string][]string `json:"entities"`
	Relations  []ExtractedRelation `json:"relations"`
	Triples    []ExtendedTriple    `json:"triples"`
	Rules      []Rule              `json:"rules"`
}
