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
