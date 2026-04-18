package verification

import (
	"testing"
)

func TestSplitClaims(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantLen  int
		wantMeta []bool
	}{
		{
			name:    "single sentence",
			input:   "Pregnancy typically lasts about 40 weeks.",
			wantLen: 1,
		},
		{
			name:    "multiple sentences",
			input:   "Pregnancy typically lasts about 40 weeks. The first trimester covers weeks 1 through 12. Morning sickness is common during this period.",
			wantLen: 3,
		},
		{
			name:    "filters short fragments",
			input:   "Yes. No. Pregnancy typically lasts about 40 weeks.",
			wantLen: 1,
		},
		{
			name:     "detects meta-statements",
			input:    "I don't have specific information about that topic. Please consult with your healthcare provider.",
			wantLen:  2,
			wantMeta: []bool{true, true},
		},
		{
			name:    "preserves decimal numbers",
			input:   "The normal body temperature is 98.6 degrees Fahrenheit. Blood pressure should be below 120 over 80.",
			wantLen: 2,
		},
		{
			name:    "preserves abbreviations",
			input:   "Dr. Smith recommends regular exercise. The study by Johnson et al. found similar results.",
			wantLen: 2,
		},
		{
			name:    "empty input",
			input:   "",
			wantLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			claims := SplitClaims(tt.input)
			if len(claims) != tt.wantLen {
				t.Errorf("got %d claims, want %d", len(claims), tt.wantLen)
				for i, c := range claims {
					t.Logf("  claim[%d]: %q", i, c.Text)
				}
			}

			if tt.wantMeta != nil {
				for i, wantMeta := range tt.wantMeta {
					if i < len(claims) && claims[i].IsMetaStatement != wantMeta {
						t.Errorf("claim[%d] IsMetaStatement = %v, want %v (text: %q)",
							i, claims[i].IsMetaStatement, wantMeta, claims[i].Text)
					}
				}
			}
		})
	}
}

func TestHallucinationPatterns(t *testing.T) {
	tests := []struct {
		text string
		want bool
	}{
		{"Based on my knowledge, this is true.", true},
		{"It is well-known that exercise is beneficial.", true},
		{"According to the provided context, the result was positive.", false},
		{"The study found a correlation between the two factors.", false},
	}

	for _, tt := range tests {
		got := hasHallucinationPattern(tt.text)
		if got != tt.want {
			t.Errorf("hasHallucinationPattern(%q) = %v, want %v", tt.text, got, tt.want)
		}
	}
}
