package verification

import (
	"regexp"
	"strings"
)

const minClaimLength = 15

// metaPatterns match statements where the LLM honestly admits uncertainty.
// These should never be flagged as hallucinations.
var metaPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)(?:the |my )?(?:documents?|sources?|context|knowledge base|information)(?: provided)? (?:do|does) not (?:contain|include|mention|have|provide|specify)`),
	regexp.MustCompile(`(?i)i (?:don'?t|do not) have (?:specific |enough )?information`),
	regexp.MustCompile(`(?i)(?:there is|there'?s) no (?:specific |available )?(?:information|data|evidence)`),
	regexp.MustCompile(`(?i)i (?:cannot|can'?t|could not|couldn'?t) find`),
	regexp.MustCompile(`(?i)(?:based on|according to) the (?:provided |available )?(?:context|documents?|sources?|information)`),
	regexp.MustCompile(`(?i)the (?:provided |available )?(?:context|documents?|sources?) (?:do|does) not`),
	regexp.MustCompile(`(?i)i (?:am|'m) not (?:sure|certain|able)`),
	regexp.MustCompile(`(?i)please (?:consult|verify|check) with`),
}

// hallucinationPatterns match signals that the LLM is fabricating outside source context.
var hallucinationPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)(?:based on|from) my (?:knowledge|training|understanding)`),
	regexp.MustCompile(`(?i)it is (?:well[ -])?known that`),
	regexp.MustCompile(`(?i)(?:as |it is )?(?:generally|commonly|widely) (?:known|accepted|understood|believed)`),
	regexp.MustCompile(`(?i)in (?:my|general) (?:experience|understanding)`),
}

// protectedAbbreviations are patterns that should not be treated as sentence boundaries.
var protectedAbbreviations = regexp.MustCompile(`(?i)\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|etc|vs|e\.g|i\.e|al|approx|dept|est|fig|inc|ltd|no|vol)\.\s`)

// sentenceBoundary matches sentence-ending punctuation followed by whitespace and uppercase.
var sentenceBoundary = regexp.MustCompile(`([.!?])(\s+)([A-Z])`)

// SplitClaims splits an LLM response into individual claim sentences,
// classifying each as a regular claim or a meta-statement.
func SplitClaims(text string) []Claim {
	sentences := splitSentences(text)

	claims := make([]Claim, 0, len(sentences))
	for _, s := range sentences {
		s = strings.TrimSpace(s)
		if len(s) < minClaimLength {
			continue
		}

		claim := Claim{
			Text:            s,
			IsMetaStatement: isMetaStatement(s),
			HasHallucinationPattern: hasHallucinationPattern(s),
		}
		claims = append(claims, claim)
	}
	return claims
}

// Claim represents a single extracted claim from the LLM response.
type Claim struct {
	Text                    string
	IsMetaStatement         bool
	HasHallucinationPattern bool
}

func splitSentences(text string) []string {
	// Protect abbreviations by replacing their periods with a placeholder
	protected := protectedAbbreviations.ReplaceAllStringFunc(text, func(m string) string {
		return strings.Replace(m, ".", "\x00", 1)
	})

	// Protect decimal numbers (e.g., "3.14")
	decimalRe := regexp.MustCompile(`(\d)\.(\d)`)
	protected = decimalRe.ReplaceAllString(protected, "${1}\x01${2}")

	// Split on sentence boundaries
	var sentences []string
	remaining := protected
	for {
		loc := sentenceBoundary.FindStringSubmatchIndex(remaining)
		if loc == nil {
			if s := strings.TrimSpace(remaining); s != "" {
				sentences = append(sentences, s)
			}
			break
		}
		// Include the punctuation in the sentence
		sentence := remaining[:loc[3]]
		if s := strings.TrimSpace(sentence); s != "" {
			sentences = append(sentences, s)
		}
		remaining = remaining[loc[6]:]
	}

	// Restore protected characters
	for i, s := range sentences {
		s = strings.ReplaceAll(s, "\x00", ".")
		s = strings.ReplaceAll(s, "\x01", ".")
		sentences[i] = s
	}

	return sentences
}

func isMetaStatement(text string) bool {
	for _, p := range metaPatterns {
		if p.MatchString(text) {
			return true
		}
	}
	return false
}

func hasHallucinationPattern(text string) bool {
	for _, p := range hallucinationPatterns {
		if p.MatchString(text) {
			return true
		}
	}
	return false
}
