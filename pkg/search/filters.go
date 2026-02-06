package search

import (
	"fmt"
	"strings"
	"time"

	"github.com/soundprediction/predicato/pkg/types"
)

// ComparisonOperator defines comparison operators for filtering
type ComparisonOperator string

const (
	Equals           ComparisonOperator = "="
	NotEquals        ComparisonOperator = "<>"
	GreaterThan      ComparisonOperator = ">"
	LessThan         ComparisonOperator = "<"
	GreaterThanEqual ComparisonOperator = ">="
	LessThanEqual    ComparisonOperator = "<="
	IsNull           ComparisonOperator = "IS NULL"
	IsNotNull        ComparisonOperator = "IS NOT NULL"
)

// DateFilter represents a date-based filter with comparison operator
type DateFilter struct {
	Date               *time.Time         `json:"date,omitempty"`
	ComparisonOperator ComparisonOperator `json:"comparison_operator"`
}

// EnhancedSearchFilters provides more sophisticated filtering capabilities
type EnhancedSearchFilters struct {
	GroupIDs    []string         `json:"group_ids,omitempty"`
	NodeTypes   []types.NodeType `json:"node_types,omitempty"`
	EdgeTypes   []types.EdgeType `json:"edge_types,omitempty"`
	EntityTypes []string         `json:"entity_types,omitempty"`
	TimeRange   *types.TimeRange `json:"time_range,omitempty"`

	// Date filters with comparison operators
	ValidFrom [][]DateFilter `json:"valid_from,omitempty"`
	ValidTo   [][]DateFilter `json:"valid_to,omitempty"`
	CreatedAt [][]DateFilter `json:"created_at,omitempty"`
	UpdatedAt [][]DateFilter `json:"updated_at,omitempty"`
}

// FilterQueryResult contains the constructed query parts and parameters
type FilterQueryResult struct {
	Parameters map[string]interface{} `json:"parameters"`
	Queries    []string               `json:"queries"`
}

// NodeSearchFilterQueryConstructor constructs filter queries for node searches
func NodeSearchFilterQueryConstructor(filters *EnhancedSearchFilters) *FilterQueryResult {
	filterQueries := []string{}
	filterParams := map[string]interface{}{}

	if filters == nil {
		return &FilterQueryResult{
			Queries:    filterQueries,
			Parameters: filterParams,
		}
	}

	// Handle node type filtering
	if len(filters.NodeTypes) > 0 {
		var nodeTypeStrs []string
		for _, nt := range filters.NodeTypes {
			nodeTypeStrs = append(nodeTypeStrs, string(nt))
		}
		filterQueries = append(filterQueries, "n.type IN $node_types")
		filterParams["node_types"] = nodeTypeStrs
	}

	// Handle entity type filtering
	if len(filters.EntityTypes) > 0 {
		filterQueries = append(filterQueries, "n.entity_type IN $entity_types")
		filterParams["entity_types"] = filters.EntityTypes
	}

	// Handle group ID filtering
	if len(filters.GroupIDs) > 0 {
		filterQueries = append(filterQueries, "n.group_id IN $group_ids")
		filterParams["group_ids"] = filters.GroupIDs
	}

	return &FilterQueryResult{
		Queries:    filterQueries,
		Parameters: filterParams,
	}
}

// EdgeSearchFilterQueryConstructor constructs filter queries for edge searches
func EdgeSearchFilterQueryConstructor(filters *EnhancedSearchFilters) *FilterQueryResult {
	filterQueries := []string{}
	filterParams := map[string]interface{}{}

	if filters == nil {
		return &FilterQueryResult{
			Queries:    filterQueries,
			Parameters: filterParams,
		}
	}

	// Handle edge type filtering
	if len(filters.EdgeTypes) > 0 {
		var edgeTypeStrs []string
		for _, et := range filters.EdgeTypes {
			edgeTypeStrs = append(edgeTypeStrs, string(et))
		}
		filterQueries = append(filterQueries, "e.type IN $edge_types")
		filterParams["edge_types"] = edgeTypeStrs
	}

	// Handle node type filtering for connected nodes
	if len(filters.NodeTypes) > 0 {
		var nodeTypeStrs []string
		for _, nt := range filters.NodeTypes {
			nodeTypeStrs = append(nodeTypeStrs, string(nt))
		}
		filterQueries = append(filterQueries, "n.type IN $node_types AND m.type IN $node_types")
		filterParams["node_types"] = nodeTypeStrs
	}

	// Handle group ID filtering
	if len(filters.GroupIDs) > 0 {
		filterQueries = append(filterQueries, "e.group_id IN $group_ids")
		filterParams["group_ids"] = filters.GroupIDs
	}

	// Handle ValidFrom date filtering
	if len(filters.ValidFrom) > 0 {
		dateQuery, dateParams := constructDateFilterQuery("e.valid_from", "valid_from", filters.ValidFrom)
		if dateQuery != "" {
			filterQueries = append(filterQueries, dateQuery)
			for k, v := range dateParams {
				filterParams[k] = v
			}
		}
	}

	// Handle ValidTo date filtering
	if len(filters.ValidTo) > 0 {
		dateQuery, dateParams := constructDateFilterQuery("e.valid_to", "valid_to", filters.ValidTo)
		if dateQuery != "" {
			filterQueries = append(filterQueries, dateQuery)
			for k, v := range dateParams {
				filterParams[k] = v
			}
		}
	}

	// Handle CreatedAt date filtering
	if len(filters.CreatedAt) > 0 {
		dateQuery, dateParams := constructDateFilterQuery("e.created_at", "created_at", filters.CreatedAt)
		if dateQuery != "" {
			filterQueries = append(filterQueries, dateQuery)
			for k, v := range dateParams {
				filterParams[k] = v
			}
		}
	}

	// Handle UpdatedAt date filtering
	if len(filters.UpdatedAt) > 0 {
		dateQuery, dateParams := constructDateFilterQuery("e.updated_at", "updated_at", filters.UpdatedAt)
		if dateQuery != "" {
			filterQueries = append(filterQueries, dateQuery)
			for k, v := range dateParams {
				filterParams[k] = v
			}
		}
	}

	return &FilterQueryResult{
		Queries:    filterQueries,
		Parameters: filterParams,
	}
}

// constructDateFilterQuery constructs date filter queries with proper parameter handling
func constructDateFilterQuery(fieldName, paramPrefix string, dateFilters [][]DateFilter) (string, map[string]interface{}) {
	if len(dateFilters) == 0 {
		return "", nil
	}

	filterParams := map[string]interface{}{}
	var orClauses []string

	for i, orList := range dateFilters {
		var andClauses []string

		for j, dateFilter := range orList {
			paramName := fmt.Sprintf("%s_%d_%d", paramPrefix, i, j)
			clause := constructSingleDateFilterQuery(fieldName, paramName, dateFilter.ComparisonOperator)
			andClauses = append(andClauses, clause)

			// Only add parameter if it's not a NULL check
			if dateFilter.ComparisonOperator != IsNull && dateFilter.ComparisonOperator != IsNotNull {
				if dateFilter.Date != nil {
					filterParams[paramName] = *dateFilter.Date
				}
			}
		}

		if len(andClauses) > 0 {
			orClause := strings.Join(andClauses, " AND ")
			if len(andClauses) > 1 {
				orClause = "(" + orClause + ")"
			}
			orClauses = append(orClauses, orClause)
		}
	}

	if len(orClauses) == 0 {
		return "", nil
	}

	finalQuery := strings.Join(orClauses, " OR ")
	if len(orClauses) > 1 {
		finalQuery = "(" + finalQuery + ")"
	}

	return finalQuery, filterParams
}

// constructSingleDateFilterQuery constructs a single date filter query clause
func constructSingleDateFilterQuery(fieldName, paramName string, operator ComparisonOperator) string {
	if operator == IsNull || operator == IsNotNull {
		return fmt.Sprintf("(%s %s)", fieldName, string(operator))
	}
	return fmt.Sprintf("(%s %s $%s)", fieldName, string(operator), paramName)
}

// ConvertToBasicFilters converts EnhancedSearchFilters to basic SearchFilters for backward compatibility
func (esf *EnhancedSearchFilters) ConvertToBasicFilters() *SearchFilters {
	if esf == nil {
		return nil
	}

	return &SearchFilters{
		GroupIDs:    esf.GroupIDs,
		NodeTypes:   esf.NodeTypes,
		EdgeTypes:   esf.EdgeTypes,
		EntityTypes: esf.EntityTypes,
		TimeRange:   esf.TimeRange,
	}
}
