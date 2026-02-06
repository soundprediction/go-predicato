package gliner2

import (
	"encoding/json"
	"time"
)

type Entity struct {
	Text       string  `json:"text"`
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence,omitempty"`
	Start      int     `json:"start,omitempty"`
	End        int     `json:"end,omitempty"`
}

type Relation struct {
	HeadSpan   *Span        `json:"head_span,omitempty"`
	TailSpan   *Span        `json:"tail_span,omitempty"`
	Relation   string       `json:"relation"`
	Head       SpanOrString `json:"head"`
	Tail       SpanOrString `json:"tail"`
	Confidence float64      `json:"confidence,omitempty"`
}

type Span struct {
	Text  string `json:"text"`
	Start int    `json:"start"`
	End   int    `json:"end"`
}

type SpanOrString struct {
	Text  string `json:"text"`
	Start int    `json:"start"`
	End   int    `json:"end"`
}

func (s *SpanOrString) UnmarshalJSON(data []byte) error {
	// Try unmarshaling as a string first
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		s.Text = str
		return nil
	}
	// If it's not a string, try unmarshaling as a struct
	type alias SpanOrString
	var aux alias
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	*s = SpanOrString(aux)
	return nil
}

type Fact struct {
	SourceSpan *Span   `json:"source_span,omitempty"`
	TargetSpan *Span   `json:"target_span,omitempty"`
	Source     string  `json:"source"`
	Target     string  `json:"target"`
	Type       string  `json:"type"`
	Confidence float64 `json:"confidence,omitempty"`
}

type Classification struct {
	Task       string   `json:"task"`
	Label      string   `json:"label"`
	Labels     []string `json:"labels,omitempty"` // For multi-label
	Confidence float64  `json:"confidence,omitempty"`
}

// GLInER2 API request/response types
type ExtractRequest struct {
	Schema    interface{} `json:"schema"`
	Task      string      `json:"task"`
	Text      string      `json:"text"`
	Threshold float64     `json:"threshold,omitempty"`
}

type ExtractResponse struct {
	Result interface{} `json:"result"`
}

type EntityResult struct {
	Entities map[string][]Entity `json:"entities"`
}

type RelationResult struct {
	RelationExtraction map[string][]RelationTuple `json:"relation_extraction"`
}

type RelationTuple struct {
	Head SpanOrString `json:"head"`
	Tail SpanOrString `json:"tail"`
}

type ClassificationResult struct {
	Classifications map[string]Classification `json:"classifications"`
}

type StructuredResult struct {
	Structured map[string][]map[string]interface{} `json:"structured"`
}

// Health check
type HealthResponse struct {
	Timestamp time.Time `json:"timestamp"`
	Status    string    `json:"status"`
	Models    []string  `json:"models,omitempty"`
}
