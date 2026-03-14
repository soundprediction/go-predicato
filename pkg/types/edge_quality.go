package types

import (
	"strings"
	"unicode"
)

// genericTerms are target/source concepts too vague to be useful in an edge.
var genericTerms = map[string]bool{
	"homo sapiens":   true,
	"human":          true,
	"disease":        true,
	"protein":        true,
	"metabolism":     true,
	"organism":       true,
	"cell":           true,
	"tissue":         true,
	"gene":           true,
	"medical":        true,
	"treatment":      true,
	"patient":        true,
	"clinical trial": true,
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

// factSeparators are phrases used to split fact text into source/target concepts.
var factSeparators = []string{
	" is used to treat ",
	" treats ",
	" affects ",
	" is a symptom of ",
	" can cause ",
	" causes ",
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
	" involves ",
	" participates in ",
	" is made up of ",
}

// FilterLowValueEdges removes edges that are tautological, overly generic,
// or duplicate by fact content. Returns the filtered list.
func FilterLowValueEdges(edges []*Edge) []*Edge {
	if len(edges) == 0 {
		return edges
	}

	seenFacts := make(map[string]bool, len(edges))
	result := make([]*Edge, 0, len(edges))

	for _, e := range edges {
		normFact := normalizeEdgeText(e.Fact)
		if seenFacts[normFact] {
			continue
		}
		seenFacts[normFact] = true

		src, tgt := ExtractFactConcepts(e.Fact, e.Name)

		if IsGenericTerm(tgt) || IsGenericTerm(src) {
			continue
		}

		if e.Name != "IS_A" && IsTautological(src, tgt) {
			continue
		}

		result = append(result, e)
	}

	return result
}

// FilterLowValueTriples removes triples that are tautological, overly generic,
// or duplicate by content. Operates on ExtractedTriple for use at ingestion time.
func FilterLowValueTriples(triples []*ExtractedTriple) (kept []*ExtractedTriple, removed []*ExtractedTriple) {
	if len(triples) == 0 {
		return triples, nil
	}

	seenFacts := make(map[string]bool, len(triples))
	kept = make([]*ExtractedTriple, 0, len(triples))

	for _, t := range triples {
		// Deduplicate by normalized content
		normFact := normalizeEdgeText(t.Subject + " " + t.Predicate + " " + t.Object)
		if seenFacts[normFact] {
			removed = append(removed, t)
			continue
		}
		seenFacts[normFact] = true

		// Filter overly generic subjects/objects
		if IsGenericTerm(t.Subject) || IsGenericTerm(t.Object) {
			removed = append(removed, t)
			continue
		}

		// Filter tautological triples (skip IS_A — taxonomy overlap is expected)
		if t.Predicate != "IS_A" && IsTautological(t.Subject, t.Object) {
			removed = append(removed, t)
			continue
		}

		kept = append(kept, t)
	}

	return kept, removed
}

// ExtractFactConcepts pulls source and target concept strings from a fact.
// Facts follow patterns like "X is used to treat Y", "X affects Y", etc.
func ExtractFactConcepts(fact, predicate string) (source, target string) {
	lower := strings.ToLower(fact)

	for _, sep := range factSeparators {
		idx := strings.Index(lower, sep)
		if idx >= 0 {
			return strings.TrimSpace(fact[:idx]), strings.TrimSpace(fact[idx+len(sep):])
		}
	}

	return "", ""
}

// IsTautological returns true when source and target concepts share enough
// word overlap that the edge is uninformative.
func IsTautological(src, tgt string) bool {
	if src == "" || tgt == "" {
		return false
	}

	srcTokens := Tokenize(src)
	tgtTokens := Tokenize(tgt)

	if len(srcTokens) == 0 || len(tgtTokens) == 0 {
		return false
	}

	overlap := 0
	for tok := range srcTokens {
		if tgtTokens[tok] {
			overlap++
		}
	}

	union := len(srcTokens) + len(tgtTokens) - overlap
	if union == 0 {
		return false
	}

	jaccard := float64(overlap) / float64(union)

	smaller, larger := len(srcTokens), len(tgtTokens)
	if smaller > larger {
		smaller, larger = larger, smaller
	}
	containment := float64(overlap) / float64(smaller)

	return jaccard >= 0.5 || (containment >= 0.8 && smaller >= 2)
}

// IsGenericTerm checks if a concept is too vague to be useful.
func IsGenericTerm(s string) bool {
	if s == "" {
		return false
	}
	lower := strings.ToLower(strings.TrimSpace(s))

	if genericTerms[lower] {
		return true
	}

	tokens := Tokenize(lower)
	if len(tokens) == 1 {
		for tok := range tokens {
			if genericTerms[tok] {
				return true
			}
		}
	}

	return false
}

// Tokenize splits text into a set of lowercase word tokens, filtering stopwords.
func Tokenize(s string) map[string]bool {
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

// normalizeEdgeText lowercases and collapses whitespace for dedup comparison.
func normalizeEdgeText(s string) string {
	return strings.Join(strings.Fields(strings.ToLower(s)), " ")
}
