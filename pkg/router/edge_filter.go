//go:build cgo

package router

import (
	"strings"
	"unicode"

	"github.com/soundprediction/predicato/pkg/types"
)

// genericTerms are target/source concepts too vague to be useful in an edge.
var genericTerms = map[string]bool{
	"homo sapiens":  true,
	"human":         true,
	"disease":       true,
	"protein":       true,
	"metabolism":    true,
	"organism":      true,
	"cell":          true,
	"tissue":        true,
	"gene":          true,
	"medical":       true,
	"treatment":     true,
	"patient":       true,
	"clinical trial": true,
}

// filterLowValueEdges removes edges that are tautological, overly generic,
// or duplicate by fact content. Returns the filtered list.
func filterLowValueEdges(edges []*types.Edge) []*types.Edge {
	if len(edges) == 0 {
		return edges
	}

	seenFacts := make(map[string]bool, len(edges))
	result := make([]*types.Edge, 0, len(edges))

	for _, e := range edges {
		// Deduplicate by normalized fact text
		normFact := normalizeText(e.Fact)
		if seenFacts[normFact] {
			continue
		}
		seenFacts[normFact] = true

		// Extract source/target concepts from the fact text
		src, tgt := extractFactConcepts(e.Fact, e.Name)

		// Filter overly generic targets/sources
		if isGenericTerm(tgt) || isGenericTerm(src) {
			continue
		}

		// Filter tautological edges (high word overlap between src and tgt)
		// Skip IS_A edges since taxonomy is usually valid even with overlap
		if e.Name != "IS_A" && isTautological(src, tgt) {
			continue
		}

		result = append(result, e)
	}

	return result
}

// extractFactConcepts pulls the source and target concept strings from a fact.
// Facts follow patterns like "X is used to treat Y", "X affects Y", etc.
func extractFactConcepts(fact, predicate string) (source, target string) {
	lower := strings.ToLower(fact)

	// Try common fact patterns
	separators := []string{
		" is used to treat ",
		" treats ",
		" affects ",
		" is a symptom of ",
		" can cause ",
		" is associated with ",
		" is relevant to ",
		" is related to ",
		" is a type of ",
		" is a kind of ",
		" is classified under ",
		" is classified as ",
		" contains ",
		" is administered via ",
		" is located in ",
		" is located at ",
		" is located on ",
		" is part of ",
	}

	for _, sep := range separators {
		idx := strings.Index(lower, sep)
		if idx >= 0 {
			return strings.TrimSpace(fact[:idx]), strings.TrimSpace(fact[idx+len(sep):])
		}
	}

	return "", ""
}

// isTautological returns true when source and target concepts share enough
// word overlap that the edge is uninformative.
func isTautological(src, tgt string) bool {
	if src == "" || tgt == "" {
		return false
	}

	srcTokens := tokenize(src)
	tgtTokens := tokenize(tgt)

	if len(srcTokens) == 0 || len(tgtTokens) == 0 {
		return false
	}

	// Count overlapping tokens
	overlap := 0
	for tok := range srcTokens {
		if tgtTokens[tok] {
			overlap++
		}
	}

	// Use Jaccard similarity
	union := len(srcTokens) + len(tgtTokens) - overlap
	if union == 0 {
		return false
	}

	jaccard := float64(overlap) / float64(union)

	// Also check containment: if one is a subset of the other
	smaller, larger := len(srcTokens), len(tgtTokens)
	if smaller > larger {
		smaller, larger = larger, smaller
	}
	containment := float64(overlap) / float64(smaller)

	// Tautological if high jaccard OR if one concept is contained in the other
	return jaccard >= 0.5 || (containment >= 0.8 && smaller >= 2)
}

// isGenericTerm checks if a concept is too vague to be useful.
func isGenericTerm(s string) bool {
	if s == "" {
		return false
	}
	lower := strings.ToLower(strings.TrimSpace(s))

	// Direct lookup
	if genericTerms[lower] {
		return true
	}

	// Single-word generic terms
	tokens := tokenize(lower)
	if len(tokens) == 1 {
		for tok := range tokens {
			if genericTerms[tok] {
				return true
			}
		}
	}

	return false
}

// tokenize splits text into a set of lowercase word tokens, filtering stopwords.
func tokenize(s string) map[string]bool {
	words := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})

	tokens := make(map[string]bool, len(words))
	for _, w := range words {
		if len(w) > 1 && !stopwords[w] {
			tokens[w] = true
		}
	}
	return tokens
}

// normalizeText lowercases and collapses whitespace for dedup comparison.
func normalizeText(s string) string {
	return strings.Join(strings.Fields(strings.ToLower(s)), " ")
}

var stopwords = map[string]bool{
	"a": true, "an": true, "the": true, "is": true, "are": true,
	"was": true, "were": true, "be": true, "been": true, "being": true,
	"have": true, "has": true, "had": true, "do": true, "does": true,
	"did": true, "will": true, "would": true, "could": true, "should": true,
	"may": true, "might": true, "shall": true, "can": true,
	"of": true, "in": true, "to": true, "for": true, "with": true,
	"on": true, "at": true, "by": true, "from": true, "as": true,
	"into": true, "about": true, "between": true, "through": true,
	"and": true, "or": true, "but": true, "not": true, "no": true,
	"it": true, "its": true, "this": true, "that": true, "these": true,
	"those": true, "used": true,
}
