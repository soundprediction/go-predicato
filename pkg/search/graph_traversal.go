package search

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/types"
)

// BFSSearchOptions holds options for BFS search operations
type BFSSearchOptions struct {
	MaxDepth      int
	Limit         int
	SearchFilters *SearchFilters
	GroupIDs      []string
}

// NodeBFSSearch performs breadth-first search to find nodes connected to origin nodes
// This implements the Python node_bfs_search function from search_utils.py
func (su *SearchUtilities) NodeBFSSearch(ctx context.Context, originNodeUUIDs []string, options *BFSSearchOptions) ([]*types.Node, error) {
	if len(originNodeUUIDs) == 0 || options.MaxDepth < 1 {
		return []*types.Node{}, nil
	}

	// Set default options
	if options.Limit <= 0 {
		options.Limit = RelevantSchemaLimit
	}

	// Get the driver provider
	provider := su.driver.Provider()

	// Build filter queries and parameters
	filterQueries := []string{}
	filterParams := make(map[string]interface{})

	if options.SearchFilters != nil {
		// Add node type filters
		if len(options.SearchFilters.NodeTypes) > 0 {
			typeStrs := make([]string, len(options.SearchFilters.NodeTypes))
			for i, nt := range options.SearchFilters.NodeTypes {
				typeStrs[i] = string(nt)
			}
			filterQueries = append(filterQueries, "n.node_type IN $node_types")
			filterParams["node_types"] = typeStrs
		}

		// Add entity type filters
		if len(options.SearchFilters.EntityTypes) > 0 {
			filterQueries = append(filterQueries, "n.entity_type IN $entity_types")
			filterParams["entity_types"] = options.SearchFilters.EntityTypes
		}

		// TODO: Add time range filters if specified
		// if options.SearchFilters.TimeRange != nil {
		// 	// Would add temporal filtering here
		// }
	}

	if len(options.GroupIDs) > 0 {
		filterQueries = append(filterQueries, "n.group_id IN $group_ids")
		filterQueries = append(filterQueries, "origin.group_id IN $group_ids")
		filterParams["group_ids"] = options.GroupIDs
	}

	filterQuery := ""
	if len(filterQueries) > 0 {
		filterQuery = " AND " + strings.Join(filterQueries, " AND ")
	}

	// Build match queries based on provider
	matchQueries := []string{}

	if provider == driver.GraphProviderLadybug {
		// For Ladybug, we need to handle the RelatesToNode_ intermediate nodes
		// Depth is multiplied by 2 because of the intermediate nodes
		depth := options.MaxDepth * 2

		// Match from Episodic nodes to Entity nodes via MENTIONS
		matchQueries = append(matchQueries, `
			UNWIND $bfs_origin_node_uuids AS origin_uuid
			MATCH (origin:Episodic {uuid: origin_uuid})-[:MENTIONS]->(n:Entity)
			WHERE n.group_id = origin.group_id
		`)

		// Match from Entity nodes to other Entity nodes via RELATES_TO
		matchQueries = append(matchQueries, fmt.Sprintf(`
			UNWIND $bfs_origin_node_uuids AS origin_uuid
			MATCH (origin:Entity {uuid: origin_uuid})-[:RELATES_TO*2..%d]->(n:Entity)
			WHERE n.group_id = origin.group_id
		`, depth))

		// If depth > 1, also match from Episodic through Entity nodes
		if options.MaxDepth > 1 {
			innerDepth := (options.MaxDepth - 1) * 2
			matchQueries = append(matchQueries, fmt.Sprintf(`
				UNWIND $bfs_origin_node_uuids AS origin_uuid
				MATCH (origin:Episodic {uuid: origin_uuid})-[:MENTIONS]->(:Entity)-[:RELATES_TO*2..%d]->(n:Entity)
				WHERE n.group_id = origin.group_id
			`, innerDepth))
		}
	} else if provider == driver.GraphProviderNeptune {
		matchQueries = append(matchQueries, fmt.Sprintf(`
			UNWIND $bfs_origin_node_uuids AS origin_uuid
			MATCH (origin {uuid: origin_uuid})-[e:RELATES_TO|MENTIONS*1..%d]->(n:Entity)
			WHERE origin:Entity OR origin:Episodic
			AND n.group_id = origin.group_id
		`, options.MaxDepth))
	} else {
		// Default for Neo4j and FalkorDB
		matchQueries = append(matchQueries, fmt.Sprintf(`
			UNWIND $bfs_origin_node_uuids AS origin_uuid
			MATCH (origin {uuid: origin_uuid})-[:RELATES_TO|MENTIONS*1..%d]->(n:Entity)
			WHERE n.group_id = origin.group_id
		`, options.MaxDepth))
	}

	// Execute queries and collect results
	allRecords := []map[string]interface{}{}

	for _, matchQuery := range matchQueries {
		query := matchQuery + filterQuery + `
			RETURN
				n.uuid AS uuid,
				n.group_id AS group_id,
				n.name AS name,
				n.name_embedding AS name_embedding,
				n.labels AS labels,
				n.created_at AS created_at,
				n.summary AS summary,
				n.embedding AS embedding
			LIMIT $limit
		`

		params := map[string]interface{}{
			"bfs_origin_node_uuids": originNodeUUIDs,
			"limit":                 options.Limit,
		}
		for k, v := range filterParams {
			params[k] = v
		}

		records, _, _, err := su.driver.ExecuteQuery(ctx, query, params)
		if err != nil {
			return nil, fmt.Errorf("BFS node search query failed: %w", err)
		}

		if recordList, ok := records.([]map[string]interface{}); ok {
			allRecords = append(allRecords, recordList...)
		}
	}

	// Convert records to Node objects
	return convertRecordsToNodes(allRecords), nil
}

// EdgeBFSSearch performs breadth-first search to find edges connected to origin nodes
// This implements the Python edge_bfs_search function from search_utils.py
func (su *SearchUtilities) EdgeBFSSearch(ctx context.Context, originNodeUUIDs []string, options *BFSSearchOptions) ([]*types.Edge, error) {
	if len(originNodeUUIDs) == 0 {
		return []*types.Edge{}, nil
	}

	// Set default options
	if options.Limit <= 0 {
		options.Limit = RelevantSchemaLimit
	}

	// Get the driver provider
	provider := su.driver.Provider()

	// Build filter queries and parameters
	filterQueries := []string{}
	filterParams := make(map[string]interface{})

	if options.SearchFilters != nil {
		// Add edge type filters
		if len(options.SearchFilters.EdgeTypes) > 0 {
			typeStrs := make([]string, len(options.SearchFilters.EdgeTypes))
			for i, et := range options.SearchFilters.EdgeTypes {
				typeStrs[i] = string(et)
			}
			filterQueries = append(filterQueries, "e.edge_type IN $edge_types")
			filterParams["edge_types"] = typeStrs
		}
	}

	if len(options.GroupIDs) > 0 {
		filterQueries = append(filterQueries, "e.group_id IN $group_ids")
		filterParams["group_ids"] = options.GroupIDs
	}

	filterQuery := ""
	if len(filterQueries) > 0 {
		filterQuery = " WHERE " + strings.Join(filterQueries, " AND ")
	}

	allRecords := []map[string]interface{}{}

	if provider == driver.GraphProviderLadybug {
		// Ladybug stores entity edges with intermediate RelatesToNode_ nodes
		depth := options.MaxDepth*2 - 1
		matchQueries := []string{
			fmt.Sprintf(`
				UNWIND $bfs_origin_node_uuids AS origin_uuid
				MATCH path = (origin:Entity {uuid: origin_uuid})-[:RELATES_TO*1..%d]->(:RelatesToNode_)
				UNWIND nodes(path) AS relNode
				MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: relNode.uuid})-[:RELATES_TO]->(m:Entity)
			`, depth),
		}

		if options.MaxDepth > 1 {
			innerDepth := (options.MaxDepth-1)*2 - 1
			matchQueries = append(matchQueries, fmt.Sprintf(`
				UNWIND $bfs_origin_node_uuids AS origin_uuid
				MATCH path = (origin:Episodic {uuid: origin_uuid})-[:MENTIONS]->(:Entity)-[:RELATES_TO*1..%d]->(:RelatesToNode_)
				UNWIND nodes(path) AS relNode
				MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: relNode.uuid})-[:RELATES_TO]->(m:Entity)
			`, innerDepth))
		}

		for _, matchQuery := range matchQueries {
			query := matchQuery + filterQuery + `
				RETURN DISTINCT
					e.uuid AS uuid,
					e.group_id AS group_id,
					n.uuid AS source_node_uuid,
					m.uuid AS target_node_uuid,
					e.created_at AS created_at,
					e.name AS name,
					e.fact AS fact,
					e.fact_embedding AS fact_embedding,
					e.episodes AS episodes,
					e.expired_at AS expired_at,
					e.valid_at AS valid_at,
					e.invalid_at AS invalid_at
				LIMIT $limit
			`

			params := map[string]interface{}{
				"bfs_origin_node_uuids": originNodeUUIDs,
				"limit":                 options.Limit,
			}
			for k, v := range filterParams {
				params[k] = v
			}

			records, _, _, err := su.driver.ExecuteQuery(ctx, query, params)
			if err != nil {
				return nil, fmt.Errorf("BFS edge search query failed: %w", err)
			}

			if recordList, ok := records.([]map[string]interface{}); ok {
				allRecords = append(allRecords, recordList...)
			}
		}
	} else {
		// For Neo4j, FalkorDB, and Neptune
		var query string
		if provider == driver.GraphProviderNeptune {
			query = fmt.Sprintf(`
				UNWIND $bfs_origin_node_uuids AS origin_uuid
				MATCH path = (origin {uuid: origin_uuid})-[:RELATES_TO|MENTIONS *1..%d]->(n:Entity)
				WHERE origin:Entity OR origin:Episodic
				UNWIND relationships(path) AS rel
				MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]-(m:Entity)
			`, options.MaxDepth) + filterQuery + `
				RETURN DISTINCT
					e.uuid AS uuid,
					e.group_id AS group_id,
					startNode(e).uuid AS source_node_uuid,
					endNode(e).uuid AS target_node_uuid,
					e.created_at AS created_at,
					e.name AS name,
					e.fact AS fact,
					e.fact_embedding AS fact_embedding,
					split(e.episodes, ',') AS episodes,
					e.expired_at AS expired_at,
					e.valid_at AS valid_at,
					e.invalid_at AS invalid_at
				LIMIT $limit
			`
		} else {
			query = fmt.Sprintf(`
				UNWIND $bfs_origin_node_uuids AS origin_uuid
				MATCH path = (origin {uuid: origin_uuid})-[:RELATES_TO|MENTIONS*1..%d]->(:Entity)
				UNWIND relationships(path) AS rel
				MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]-(m:Entity)
			`, options.MaxDepth) + filterQuery + `
				RETURN DISTINCT
					e.uuid AS uuid,
					e.group_id AS group_id,
					startNode(e).uuid AS source_node_uuid,
					endNode(e).uuid AS target_node_uuid,
					e.created_at AS created_at,
					e.name AS name,
					e.fact AS fact,
					e.fact_embedding AS fact_embedding,
					e.episodes AS episodes,
					e.expired_at AS expired_at,
					e.valid_at AS valid_at,
					e.invalid_at AS invalid_at
				LIMIT $limit
			`
		}

		params := map[string]interface{}{
			"bfs_origin_node_uuids": originNodeUUIDs,
			"limit":                 options.Limit,
		}
		for k, v := range filterParams {
			params[k] = v
		}

		records, _, _, err := su.driver.ExecuteQuery(ctx, query, params)
		if err != nil {
			return nil, fmt.Errorf("BFS edge search query failed: %w", err)
		}

		if recordList, ok := records.([]map[string]interface{}); ok {
			allRecords = recordList
		}
	}

	// Convert records to Edge objects
	return convertRecordsToEdges(allRecords), nil
}

// Helper functions for BFS implementation

// convertRecordsToNodes converts database records to Node objects
func convertRecordsToNodes(records []map[string]interface{}) []*types.Node {
	nodes := make([]*types.Node, 0, len(records))
	seen := make(map[string]bool)

	for _, record := range records {
		uuid, _ := record["uuid"].(string)
		if uuid == "" || seen[uuid] {
			continue
		}
		seen[uuid] = true

		node := &types.Node{
			Uuid:    uuid,
			GroupID: stringValue(record["group_id"]),
			Name:    stringValue(record["name"]),
			Summary: stringValue(record["summary"]),
		}

		if labels := record["labels"]; labels != nil {
			node.EntityType = stringValue(labels)
		}

		if nameEmbedding := record["name_embedding"]; nameEmbedding != nil {
			node.NameEmbedding = toFloat32Slice(nameEmbedding)
		}

		if embedding := record["embedding"]; embedding != nil {
			node.Embedding = toFloat32Slice(embedding)
		}

		if createdAt := record["created_at"]; createdAt != nil {
			if t, ok := createdAt.(time.Time); ok {
				node.CreatedAt = t
			}
		}

		nodes = append(nodes, node)
	}

	return nodes
}

// convertRecordsToEdges converts database records to Edge objects
func convertRecordsToEdges(records []map[string]interface{}) []*types.Edge {
	edges := make([]*types.Edge, 0, len(records))
	seen := make(map[string]bool)

	for _, record := range records {
		uuid, _ := record["uuid"].(string)
		if uuid == "" || seen[uuid] {
			continue
		}
		seen[uuid] = true

		// Create EntityEdge instead of using BaseEdge directly
		edge := &types.EntityEdge{}
		edge.Uuid = uuid
		edge.SourceNodeID = stringValue(record["source_node_uuid"])
		edge.TargetNodeID = stringValue(record["target_node_uuid"])
		edge.GroupID = stringValue(record["group_id"])
		edge.Name = stringValue(record["name"])
		edge.Fact = stringValue(record["fact"])

		// Also set the backward compatibility fields
		edge.SourceID = edge.SourceNodeID
		edge.TargetID = edge.TargetNodeID

		if factEmbedding := record["fact_embedding"]; factEmbedding != nil {
			edge.FactEmbedding = toFloat32Slice(factEmbedding)
		}

		if episodes := record["episodes"]; episodes != nil {
			edge.Episodes = toStringSlice(episodes)
		}

		if createdAt := record["created_at"]; createdAt != nil {
			if t, ok := createdAt.(time.Time); ok {
				edge.CreatedAt = t
			}
		}

		// Convert EntityEdge to Edge (base type) for return
		baseEdge := &types.Edge{
			BaseEdge:      edge.BaseEdge,
			Fact:          edge.Fact,
			FactEmbedding: edge.FactEmbedding,
			Episodes:      edge.Episodes,
		}

		edges = append(edges, baseEdge)
	}

	return edges
}

// stringValue safely extracts string value from interface{}
func stringValue(v interface{}) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return fmt.Sprintf("%v", v)
}

// PathFinder provides utilities for finding paths between nodes
type PathFinder struct {
	driver driver.GraphDriver
}

// NewPathFinder creates a new PathFinder instance
func NewPathFinder(driver driver.GraphDriver) *PathFinder {
	return &PathFinder{
		driver: driver,
	}
}

// FindShortestPath finds the shortest path between two nodes
func (pf *PathFinder) FindShortestPath(ctx context.Context, sourceUUID, targetUUID string, maxDepth int) ([]*types.Node, []*types.Edge, error) {
	provider := pf.driver.Provider()

	var query string
	if provider == driver.GraphProviderLadybug {
		query = fmt.Sprintf(`
			MATCH path = shortestPath((source:Entity {uuid: $source_uuid})-[:RELATES_TO*1..%d]->(target:Entity {uuid: $target_uuid}))
			RETURN nodes(path) AS nodes, relationships(path) AS edges
		`, maxDepth*2)
	} else {
		query = fmt.Sprintf(`
			MATCH path = shortestPath((source:Entity {uuid: $source_uuid})-[:RELATES_TO*1..%d]->(target:Entity {uuid: $target_uuid}))
			RETURN nodes(path) AS nodes, relationships(path) AS edges
		`, maxDepth)
	}

	params := map[string]interface{}{
		"source_uuid": sourceUUID,
		"target_uuid": targetUUID,
	}

	_, _, _, err := pf.driver.ExecuteQuery(ctx, query, params)
	if err != nil {
		return nil, nil, fmt.Errorf("shortest path query failed: %w", err)
	}

	// Parse the path results
	// This would need proper implementation to extract nodes and edges from the path
	return []*types.Node{}, []*types.Edge{}, nil
}

// FindAllPaths finds all paths between two nodes up to a maximum depth
func (pf *PathFinder) FindAllPaths(ctx context.Context, sourceUUID, targetUUID string, maxDepth int) ([][]*types.Node, [][]*types.Edge, error) {
	provider := pf.driver.Provider()

	var query string
	if provider == driver.GraphProviderLadybug {
		query = fmt.Sprintf(`
			MATCH path = (source:Entity {uuid: $source_uuid})-[:RELATES_TO*1..%d]->(target:Entity {uuid: $target_uuid})
			RETURN nodes(path) AS nodes, relationships(path) AS edges
		`, maxDepth*2)
	} else {
		query = fmt.Sprintf(`
			MATCH path = (source:Entity {uuid: $source_uuid})-[:RELATES_TO*1..%d]->(target:Entity {uuid: $target_uuid})
			RETURN nodes(path) AS nodes, relationships(path) AS edges
		`, maxDepth)
	}

	params := map[string]interface{}{
		"source_uuid": sourceUUID,
		"target_uuid": targetUUID,
	}

	_, _, _, err := pf.driver.ExecuteQuery(ctx, query, params)
	if err != nil {
		return nil, nil, fmt.Errorf("all paths query failed: %w", err)
	}

	// Would need to parse multiple paths
	return [][]*types.Node{}, [][]*types.Edge{}, nil
}

// GetNeighbors gets direct neighbors of a node
func (pf *PathFinder) GetNeighbors(ctx context.Context, nodeUUID string, direction string) ([]*types.Node, error) {
	provider := pf.driver.Provider()

	var query string
	switch direction {
	case "out":
		if provider == driver.GraphProviderLadybug {
			query = `
				MATCH (n:Entity {uuid: $node_uuid})-[:RELATES_TO*2..2]->(neighbor:Entity)
				RETURN DISTINCT neighbor.uuid AS uuid, neighbor.name AS name, neighbor.group_id AS group_id
			`
		} else {
			query = `
				MATCH (n:Entity {uuid: $node_uuid})-[:RELATES_TO]->(neighbor:Entity)
				RETURN DISTINCT neighbor.uuid AS uuid, neighbor.name AS name, neighbor.group_id AS group_id
			`
		}
	case "in":
		if provider == driver.GraphProviderLadybug {
			query = `
				MATCH (neighbor:Entity)-[:RELATES_TO*2..2]->(n:Entity {uuid: $node_uuid})
				RETURN DISTINCT neighbor.uuid AS uuid, neighbor.name AS name, neighbor.group_id AS group_id
			`
		} else {
			query = `
				MATCH (neighbor:Entity)-[:RELATES_TO]->(n:Entity {uuid: $node_uuid})
				RETURN DISTINCT neighbor.uuid AS uuid, neighbor.name AS name, neighbor.group_id AS group_id
			`
		}
	default: // "both"
		if provider == driver.GraphProviderLadybug {
			query = `
				MATCH (n:Entity {uuid: $node_uuid})-[:RELATES_TO*2..2]-(neighbor:Entity)
				RETURN DISTINCT neighbor.uuid AS uuid, neighbor.name AS name, neighbor.group_id AS group_id
			`
		} else {
			query = `
				MATCH (n:Entity {uuid: $node_uuid})-[:RELATES_TO]-(neighbor:Entity)
				RETURN DISTINCT neighbor.uuid AS uuid, neighbor.name AS name, neighbor.group_id AS group_id
			`
		}
	}

	params := map[string]interface{}{
		"node_uuid": nodeUUID,
	}

	records, _, _, err := pf.driver.ExecuteQuery(ctx, query, params)
	if err != nil {
		return nil, fmt.Errorf("get neighbors query failed: %w", err)
	}

	if recordList, ok := records.([]map[string]interface{}); ok {
		return convertRecordsToNodes(recordList), nil
	}

	return []*types.Node{}, nil
}

// Community-related traversal functions

// CommunityTraversal provides utilities for community-based graph traversal
type CommunityTraversal struct {
	driver driver.GraphDriver
}

// NewCommunityTraversal creates a new CommunityTraversal instance
func NewCommunityTraversal(driver driver.GraphDriver) *CommunityTraversal {
	return &CommunityTraversal{
		driver: driver,
	}
}

// GetCommunityMembers retrieves all members of a community
func (ct *CommunityTraversal) GetCommunityMembers(ctx context.Context, communityUUID string) ([]*types.Node, error) {
	query := `
		MATCH (c:Community {uuid: $community_uuid})-[:HAS_MEMBER]->(member:Entity)
		RETURN
			member.uuid AS uuid,
			member.name AS name,
			member.group_id AS group_id,
			member.summary AS summary
	`

	params := map[string]interface{}{
		"community_uuid": communityUUID,
	}

	records, _, _, err := ct.driver.ExecuteQuery(ctx, query, params)
	if err != nil {
		return nil, fmt.Errorf("failed to get community members: %w", err)
	}

	if recordList, ok := records.([]map[string]interface{}); ok {
		return convertRecordsToNodes(recordList), nil
	}

	return []*types.Node{}, nil
}

// GetNodeCommunities retrieves all communities that a node belongs to
func (ct *CommunityTraversal) GetNodeCommunities(ctx context.Context, nodeUUID string) ([]*types.Node, error) {
	query := `
		MATCH (c:Community)-[:HAS_MEMBER]->(n:Entity {uuid: $node_uuid})
		RETURN
			c.uuid AS uuid,
			c.name AS name,
			c.group_id AS group_id,
			c.summary AS summary
	`

	params := map[string]interface{}{
		"node_uuid": nodeUUID,
	}

	records, _, _, err := ct.driver.ExecuteQuery(ctx, query, params)
	if err != nil {
		return nil, fmt.Errorf("failed to get node communities: %w", err)
	}

	if recordList, ok := records.([]map[string]interface{}); ok {
		return convertRecordsToNodes(recordList), nil
	}

	return []*types.Node{}, nil
}

// GetInterCommunityEdges retrieves edges between different communities
func (ct *CommunityTraversal) GetInterCommunityEdges(ctx context.Context, communityUUID1, communityUUID2 string) ([]*types.Edge, error) {
	provider := ct.driver.Provider()

	var query string
	if provider == driver.GraphProviderLadybug {
		query = `
			MATCH (c1:Community {uuid: $community_uuid1})-[:HAS_MEMBER]->(n1:Entity)
			MATCH (c2:Community {uuid: $community_uuid2})-[:HAS_MEMBER]->(n2:Entity)
			MATCH (n1)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(n2)
			RETURN DISTINCT
				e.uuid AS uuid,
				e.group_id AS group_id,
				n1.uuid AS source_node_uuid,
				n2.uuid AS target_node_uuid,
				e.fact AS fact
		`
	} else {
		query = `
			MATCH (c1:Community {uuid: $community_uuid1})-[:HAS_MEMBER]->(n1:Entity)
			MATCH (c2:Community {uuid: $community_uuid2})-[:HAS_MEMBER]->(n2:Entity)
			MATCH (n1)-[e:RELATES_TO]-(n2)
			RETURN DISTINCT
				e.uuid AS uuid,
				e.group_id AS group_id,
				startNode(e).uuid AS source_node_uuid,
				endNode(e).uuid AS target_node_uuid,
				e.fact AS fact
		`
	}

	params := map[string]interface{}{
		"community_uuid1": communityUUID1,
		"community_uuid2": communityUUID2,
	}

	records, _, _, err := ct.driver.ExecuteQuery(ctx, query, params)
	if err != nil {
		return nil, fmt.Errorf("failed to get inter-community edges: %w", err)
	}

	if recordList, ok := records.([]map[string]interface{}); ok {
		return convertRecordsToEdges(recordList), nil
	}

	return []*types.Edge{}, nil
}

// Temporal traversal functions

// TemporalTraversal provides utilities for time-aware graph traversal
type TemporalTraversal struct {
	driver driver.GraphDriver
}

// NewTemporalTraversal creates a new TemporalTraversal instance
func NewTemporalTraversal(driver driver.GraphDriver) *TemporalTraversal {
	return &TemporalTraversal{
		driver: driver,
	}
}

// GetNodesInTimeRange retrieves nodes created or valid within a time range
func (tt *TemporalTraversal) GetNodesInTimeRange(ctx context.Context, timeRange *types.TimeRange, groupID string) ([]*types.Node, error) {
	if timeRange == nil {
		return []*types.Node{}, fmt.Errorf("time range is required")
	}

	query := `
		MATCH (n:Entity)
		WHERE n.group_id = $group_id
		  AND n.created_at >= $start_time
		  AND n.created_at <= $end_time
		RETURN
			n.uuid AS uuid,
			n.name AS name,
			n.group_id AS group_id,
			n.created_at AS created_at,
			n.summary AS summary
	`

	params := map[string]interface{}{
		"group_id":   groupID,
		"start_time": timeRange.Start,
		"end_time":   timeRange.End,
	}

	records, _, _, err := tt.driver.ExecuteQuery(ctx, query, params)
	if err != nil {
		return nil, fmt.Errorf("failed to get nodes in time range: %w", err)
	}

	if recordList, ok := records.([]map[string]interface{}); ok {
		return convertRecordsToNodes(recordList), nil
	}

	return []*types.Node{}, nil
}

// GetEdgesInTimeRange retrieves edges created or valid within a time range
func (tt *TemporalTraversal) GetEdgesInTimeRange(ctx context.Context, timeRange *types.TimeRange, groupID string) ([]*types.Edge, error) {
	if timeRange == nil {
		return []*types.Edge{}, fmt.Errorf("time range is required")
	}

	provider := tt.driver.Provider()

	var query string
	if provider == driver.GraphProviderLadybug {
		query = `
			MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)
			WHERE e.group_id = $group_id
			  AND e.created_at >= $start_time
			  AND e.created_at <= $end_time
			RETURN DISTINCT
				e.uuid AS uuid,
				e.group_id AS group_id,
				n.uuid AS source_node_uuid,
				m.uuid AS target_node_uuid,
				e.created_at AS created_at,
				e.fact AS fact
		`
	} else {
		query = `
			MATCH (n:Entity)-[e:RELATES_TO]-(m:Entity)
			WHERE e.group_id = $group_id
			  AND e.created_at >= $start_time
			  AND e.created_at <= $end_time
			RETURN DISTINCT
				e.uuid AS uuid,
				e.group_id AS group_id,
				startNode(e).uuid AS source_node_uuid,
				endNode(e).uuid AS target_node_uuid,
				e.created_at AS created_at,
				e.fact AS fact
		`
	}

	params := map[string]interface{}{
		"group_id":   groupID,
		"start_time": timeRange.Start,
		"end_time":   timeRange.End,
	}

	records, _, _, err := tt.driver.ExecuteQuery(ctx, query, params)
	if err != nil {
		return nil, fmt.Errorf("failed to get edges in time range: %w", err)
	}

	if recordList, ok := records.([]map[string]interface{}); ok {
		return convertRecordsToEdges(recordList), nil
	}

	return []*types.Edge{}, nil
}

// GetTemporalNeighbors gets neighbors of a node at a specific point in time
func (tt *TemporalTraversal) GetTemporalNeighbors(ctx context.Context, nodeUUID string, timestamp int64, direction string) ([]*types.Node, error) {
	provider := tt.driver.Provider()
	targetTime := time.Unix(timestamp, 0)

	var query string
	switch direction {
	case "out":
		if provider == driver.GraphProviderLadybug {
			query = `
				MATCH (n:Entity {uuid: $node_uuid})-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(neighbor:Entity)
				WHERE e.created_at <= $target_time
				  AND (e.expired_at IS NULL OR e.expired_at > $target_time)
				RETURN DISTINCT neighbor.uuid AS uuid, neighbor.name AS name, neighbor.group_id AS group_id
			`
		} else {
			query = `
				MATCH (n:Entity {uuid: $node_uuid})-[e:RELATES_TO]->(neighbor:Entity)
				WHERE e.created_at <= $target_time
				  AND (e.expired_at IS NULL OR e.expired_at > $target_time)
				RETURN DISTINCT neighbor.uuid AS uuid, neighbor.name AS name, neighbor.group_id AS group_id
			`
		}
	case "in":
		if provider == driver.GraphProviderLadybug {
			query = `
				MATCH (neighbor:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(n:Entity {uuid: $node_uuid})
				WHERE e.created_at <= $target_time
				  AND (e.expired_at IS NULL OR e.expired_at > $target_time)
				RETURN DISTINCT neighbor.uuid AS uuid, neighbor.name AS name, neighbor.group_id AS group_id
			`
		} else {
			query = `
				MATCH (neighbor:Entity)-[e:RELATES_TO]->(n:Entity {uuid: $node_uuid})
				WHERE e.created_at <= $target_time
				  AND (e.expired_at IS NULL OR e.expired_at > $target_time)
				RETURN DISTINCT neighbor.uuid AS uuid, neighbor.name AS name, neighbor.group_id AS group_id
			`
		}
	default: // "both"
		if provider == driver.GraphProviderLadybug {
			query = `
				MATCH (n:Entity {uuid: $node_uuid})-[:RELATES_TO]-(e:RelatesToNode_)-[:RELATES_TO]-(neighbor:Entity)
				WHERE e.created_at <= $target_time
				  AND (e.expired_at IS NULL OR e.expired_at > $target_time)
				RETURN DISTINCT neighbor.uuid AS uuid, neighbor.name AS name, neighbor.group_id AS group_id
			`
		} else {
			query = `
				MATCH (n:Entity {uuid: $node_uuid})-[e:RELATES_TO]-(neighbor:Entity)
				WHERE e.created_at <= $target_time
				  AND (e.expired_at IS NULL OR e.expired_at > $target_time)
				RETURN DISTINCT neighbor.uuid AS uuid, neighbor.name AS name, neighbor.group_id AS group_id
			`
		}
	}

	params := map[string]interface{}{
		"node_uuid":   nodeUUID,
		"target_time": targetTime,
	}

	records, _, _, err := tt.driver.ExecuteQuery(ctx, query, params)
	if err != nil {
		return nil, fmt.Errorf("failed to get temporal neighbors: %w", err)
	}

	if recordList, ok := records.([]map[string]interface{}); ok {
		return convertRecordsToNodes(recordList), nil
	}

	return []*types.Node{}, nil
}
