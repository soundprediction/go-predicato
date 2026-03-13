//go:build cgo

package router

import "strings"

// queryPredicateRules maps query keyword patterns to preferred edge predicates.
// Multiple rules can match; all matched predicates are collected in priority order.
var queryPredicateRules = []struct {
	keywords   []string // any of these in the query triggers this rule
	predicates []string // edge names to boost
}{
	{
		keywords:   []string{"treat", "medication", "drug", "therapy", "cure", "prescri"},
		predicates: []string{"TREATS", "ADMINISTERED_VIA"},
	},
	{
		keywords:   []string{"cause", "why", "risk factor", "etiology", "pathogen"},
		predicates: []string{"CAUSES", "ASSOCIATED_WITH"},
	},
	{
		keywords:   []string{"symptom", "sign", "diagnos", "present", "manifest"},
		predicates: []string{"SYMPTOM_OF", "HAS_CONDITION"},
	},
	{
		keywords:   []string{"side effect", "adverse", "toxicity", "complication"},
		predicates: []string{"SIDE_EFFECT_OF", "AFFECTS", "CAUSES"},
	},
	{
		keywords:   []string{"what is", "type of", "classif", "kind of", "category"},
		predicates: []string{"IS_A", "COMPRISES"},
	},
	{
		keywords:   []string{"how does", "mechanism", "pathway", "work", "action", "affect", "influence"},
		predicates: []string{"AFFECTS", "CAUSES", "ASSOCIATED_WITH"},
	},
	{
		keywords:   []string{"locat", "where", "found in", "expressed"},
		predicates: []string{"LOCATED_IN", "LOCATED_AT", "LOCATED_ON", "PART_OF"},
	},
	{
		keywords:   []string{"mutat", "gene", "genetic", "variant", "polymorphism", "heredit"},
		predicates: []string{"ASSOCIATED_WITH", "CAUSES", "IS_A"},
	},
	{
		keywords:   []string{"vaccin", "immuniz", "prevent", "prophyla", "recommend"},
		predicates: []string{"TREATS", "ASSOCIATED_WITH"},
	},
	{
		keywords:   []string{"lower", "reduce", "inhibit", "block", "suppress", "decreas"},
		predicates: []string{"AFFECTS", "TREATS", "CAUSES"},
	},
}

// InferPredicatesFromQuery detects query intent and returns edge predicate
// names that should be boosted in result ranking. Returns nil if no intent
// is detected (all predicates treated equally).
func InferPredicatesFromQuery(query string) []string {
	lower := strings.ToLower(query)

	seen := make(map[string]bool)
	var result []string

	for _, rule := range queryPredicateRules {
		matched := false
		for _, kw := range rule.keywords {
			if strings.Contains(lower, kw) {
				matched = true
				break
			}
		}
		if matched {
			for _, p := range rule.predicates {
				if !seen[p] {
					seen[p] = true
					result = append(result, p)
				}
			}
		}
	}

	return result
}
