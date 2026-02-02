package gliner2

import (
	"encoding/json"
	"testing"
)

func TestSpanOrString_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantText string
		wantStart int
		wantEnd   int
		wantErr  bool
	}{
		{
			name:     "string input",
			input:    `"Aspirin"`,
			wantText: "Aspirin",
			wantStart: 0,
			wantEnd:   0,
			wantErr:  false,
		},
		{
			name:     "object input",
			input:    `{"text": "Aspirin", "start": 0, "end": 7}`,
			wantText: "Aspirin",
			wantStart: 0,
			wantEnd:   7,
			wantErr:  false,
		},
		{
			name:    "invalid input",
			input:   `123`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var s SpanOrString
			err := json.Unmarshal([]byte(tt.input), &s)
			if (err != nil) != tt.wantErr {
				t.Errorf("UnmarshalJSON() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				if s.Text != tt.wantText {
					t.Errorf("Text = %v, want %v", s.Text, tt.wantText)
				}
				if s.Start != tt.wantStart {
					t.Errorf("Start = %v, want %v", s.Start, tt.wantStart)
				}
				if s.End != tt.wantEnd {
					t.Errorf("End = %v, want %v", s.End, tt.wantEnd)
				}
			}
		})
	}
}
