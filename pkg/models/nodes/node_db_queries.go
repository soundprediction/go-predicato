package nodes

import (
	"fmt"
	"strings"

	"github.com/soundprediction/predicato/pkg/driver"
)

// GetEpisodeNodeSaveQuery returns the episode node save query based on provider
func GetEpisodeNodeSaveQuery(provider driver.GraphProvider) string {
	switch provider {
	case driver.GraphProviderNeptune:
		return `
                MERGE (n:Episodic {uuid: $uuid})
                SET n = {uuid: $uuid, name: $name, group_id: $group_id, source_description: $source_description, source: $source, content: $content,
                entity_edges: join([x IN coalesce($entity_edges, []) | toString(x) ], '|'), created_at: $created_at, valid_at: $valid_at}
                RETURN n.uuid AS uuid
            `
	case driver.GraphProviderLadybug:
		return `
                MERGE (n:Episodic {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.created_at = $created_at,
                    n.source = $source,
                    n.source_description = $source_description,
                    n.content = $content,
                    n.valid_at = $valid_at,
                    n.entity_edges = $entity_edges
                RETURN n.uuid AS uuid
            `
	case driver.GraphProviderFalkorDB:
		return `
                MERGE (n:Episodic {uuid: $uuid})
                SET n = {uuid: $uuid, name: $name, group_id: $group_id, source_description: $source_description, source: $source, content: $content,
                entity_edges: $entity_edges, created_at: $created_at, valid_at: $valid_at}
                RETURN n.uuid AS uuid
            `
	default: // Neo4j
		return `
                MERGE (n:Episodic {uuid: $uuid})
                SET n = {uuid: $uuid, name: $name, group_id: $group_id, source_description: $source_description, source: $source, content: $content,
                entity_edges: $entity_edges, created_at: $created_at, valid_at: $valid_at}
                RETURN n.uuid AS uuid
            `
	}
}

// GetEpisodeNodeSaveBulkQuery returns the bulk episode node save query based on provider
func GetEpisodeNodeSaveBulkQuery(provider driver.GraphProvider) string {
	switch provider {
	case driver.GraphProviderNeptune:
		return `
                UNWIND $episodes AS episode
                MERGE (n:Episodic {uuid: episode.uuid})
                SET n = {uuid: episode.uuid, name: episode.name, group_id: episode.group_id, source_description: episode.source_description,
                    source: episode.source, content: episode.content,
                entity_edges: join([x IN coalesce(episode.entity_edges, []) | toString(x) ], '|'), created_at: episode.created_at, valid_at: episode.valid_at}
                RETURN n.uuid AS uuid
            `
	case driver.GraphProviderLadybug:
		return `
                MERGE (n:Episodic {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.created_at = $created_at,
                    n.source = $source,
                    n.source_description = $source_description,
                    n.content = $content,
                    n.valid_at = $valid_at,
                    n.entity_edges = $entity_edges
                RETURN n.uuid AS uuid
            `
	case driver.GraphProviderFalkorDB:
		return `
                UNWIND $episodes AS episode
                MERGE (n:Episodic {uuid: episode.uuid})
                SET n = {uuid: episode.uuid, name: episode.name, group_id: episode.group_id, source_description: episode.source_description, source: episode.source, content: episode.content,
                entity_edges: episode.entity_edges, created_at: episode.created_at, valid_at: episode.valid_at}
                RETURN n.uuid AS uuid
            `
	default: // Neo4j
		return `
                UNWIND $episodes AS episode
                MERGE (n:Episodic {uuid: episode.uuid})
                SET n = {uuid: episode.uuid, name: episode.name, group_id: episode.group_id, source_description: episode.source_description, source: episode.source, content: episode.content,
                entity_edges: episode.entity_edges, created_at: episode.created_at, valid_at: episode.valid_at}
                RETURN n.uuid AS uuid
            `
	}
}

// EPISODIC_NODE_RETURN query constant
const EPISODIC_NODE_RETURN = `
    e.uuid AS uuid,
    e.name AS name,
    e.group_id AS group_id,
    e.created_at AS created_at,
    e.source AS source,
    e.source_description AS source_description,
    e.content AS content,
    e.valid_at AS valid_at,
    e.entity_edges AS entity_edges
`

// EPISODIC_NODE_RETURN_NEPTUNE query constant for Neptune provider
const EPISODIC_NODE_RETURN_NEPTUNE = `
    e.content AS content,
    e.created_at AS created_at,
    e.valid_at AS valid_at,
    e.uuid AS uuid,
    e.name AS name,
    e.group_id AS group_id,
    e.source_description AS source_description,
    e.source AS source,
    split(e.entity_edges, ",") AS entity_edges
`

// GetEntityNodeSaveQuery returns the entity node save query based on provider, labels, and AOSS configuration
func GetEntityNodeSaveQuery(provider driver.GraphProvider, labels string, hasAOSS bool) string {
	switch provider {
	case driver.GraphProviderFalkorDB:
		return fmt.Sprintf(`
                MERGE (n:Entity {uuid: $entity_data.uuid})
                SET n:%s
                SET n = $entity_data
                RETURN n.uuid AS uuid
            `, labels)
	case driver.GraphProviderLadybug:
		return `
                MERGE (n:Entity {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.labels = $labels,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.summary = $summary,
                    n.attributes = $attributes
                WITH n
                RETURN n.uuid AS uuid
            `
	case driver.GraphProviderNeptune:
		labelSubquery := ""
		for _, label := range strings.Split(labels, ":") {
			if label != "" {
				labelSubquery += fmt.Sprintf(" SET n:%s\n", label)
			}
		}
		return fmt.Sprintf(`
                MERGE (n:Entity {uuid: $entity_data.uuid})
                %s
                SET n = removeKeyFromMap(removeKeyFromMap($entity_data, "labels"), "name_embedding")
                SET n.name_embedding = join([x IN coalesce($entity_data.name_embedding, []) | toString(x) ], ",")
                RETURN n.uuid AS uuid
            `, labelSubquery)
	default: // Neo4j
		saveEmbeddingQuery := ""
		if !hasAOSS {
			saveEmbeddingQuery = `WITH n CALL db.create.setNodeVectorProperty(n, "name_embedding", $entity_data.name_embedding)`
		}

		return fmt.Sprintf(`
                MERGE (n:Entity {uuid: $entity_data.uuid})
                SET n:%s
                SET n = $entity_data
                %s
                RETURN n.uuid AS uuid
            `, labels, saveEmbeddingQuery)
	}
}

// QueryWithParams represents a query with its parameters for bulk operations
type QueryWithParams struct {
	Params map[string]interface{}
	Query  string
}

// GetEntityNodeSaveBulkQuery returns the bulk entity node save query based on provider and nodes
// For FalkorDB and Neptune, it returns []QueryWithParams, for others it returns a string
func GetEntityNodeSaveBulkQuery(provider driver.GraphProvider, nodes []map[string]interface{}, hasAOSS bool) interface{} {
	switch provider {
	case driver.GraphProviderFalkorDB:
		var queries []QueryWithParams
		for _, node := range nodes {
			if labelsInterface, ok := node["labels"]; ok {
				if labels, ok := labelsInterface.([]interface{}); ok {
					for _, labelInterface := range labels {
						if label, ok := labelInterface.(string); ok {
							query := fmt.Sprintf(`
                            UNWIND $nodes AS node
                            MERGE (n:Entity {uuid: node.uuid})
                            SET n:%s
                            SET n = node
                            WITH n, node
                            SET n.name_embedding = vecf32(node.name_embedding)
                            RETURN n.uuid AS uuid
                            `, label)
							params := map[string]interface{}{
								"nodes": []map[string]interface{}{node},
							}
							queries = append(queries, QueryWithParams{
								Query:  query,
								Params: params,
							})
						}
					}
				}
			}
		}
		return queries
	case driver.GraphProviderNeptune:
		var queries []string
		for _, node := range nodes {
			labelsSubquery := ""
			if labelsInterface, ok := node["labels"]; ok {
				if labels, ok := labelsInterface.([]interface{}); ok {
					for _, labelInterface := range labels {
						if label, ok := labelInterface.(string); ok {
							labelsSubquery += fmt.Sprintf(" SET n:%s\n", label)
						}
					}
				}
			}
			query := fmt.Sprintf(`
                        UNWIND $nodes AS node
                        MERGE (n:Entity {uuid: node.uuid})
                        %s
                        SET n = removeKeyFromMap(removeKeyFromMap(node, "labels"), "name_embedding")
                        SET n.name_embedding = join([x IN coalesce(node.name_embedding, []) | toString(x) ], ",")
                        RETURN n.uuid AS uuid
                    `, labelsSubquery)
			queries = append(queries, query)
		}
		return queries
	case driver.GraphProviderLadybug:
		return `
                MERGE (n:Entity {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.labels = $labels,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.summary = $summary,
                    n.attributes = $attributes
                RETURN n.uuid AS uuid
            `
	default: // Neo4j
		saveEmbeddingQuery := ""
		if !hasAOSS {
			saveEmbeddingQuery = `WITH n, node CALL db.create.setNodeVectorProperty(n, "name_embedding", node.name_embedding)`
		}

		return fmt.Sprintf(`
                    UNWIND $nodes AS node
                    MERGE (n:Entity {uuid: node.uuid})
                    SET n:$(node.labels)
                    SET n = node
                    %s
                RETURN n.uuid AS uuid
            `, saveEmbeddingQuery)
	}
}

// GetEntityNodeReturnQuery returns the entity node return query based on provider
// Note: name_embedding is not returned by default and must be loaded manually using load_name_embedding().
func GetEntityNodeReturnQuery(provider driver.GraphProvider) string {
	if provider == driver.GraphProviderLadybug {
		return `
            n.uuid AS uuid,
            n.name AS name,
            n.group_id AS group_id,
            n.labels AS labels,
            n.created_at AS created_at,
            n.summary AS summary,
            n.attributes AS attributes
        `
	}

	return `
        n.uuid AS uuid,
        n.name AS name,
        n.group_id AS group_id,
        n.created_at AS created_at,
        n.summary AS summary,
        labels(n) AS labels,
        properties(n) AS attributes
    `
}

// GetCommunityNodeSaveQuery returns the community node save query based on provider
func GetCommunityNodeSaveQuery(provider driver.GraphProvider) string {
	switch provider {
	case driver.GraphProviderFalkorDB:
		return `
                MERGE (n:Community {uuid: $uuid})
                SET n = {uuid: $uuid, name: $name, group_id: $group_id, summary: $summary, created_at: $created_at, name_embedding: vecf32($name_embedding)}
                RETURN n.uuid AS uuid
            `
	case driver.GraphProviderNeptune:
		return `
                MERGE (n:Community {uuid: $uuid})
                SET n = {uuid: $uuid, name: $name, group_id: $group_id, summary: $summary, created_at: $created_at}
                SET n.name_embedding = join([x IN coalesce($name_embedding, []) | toString(x) ], ",")
                RETURN n.uuid AS uuid
            `
	case driver.GraphProviderLadybug:
		return `
                MERGE (n:Community {uuid: $uuid})
                SET
                    n.name = $name,
                    n.group_id = $group_id,
                    n.created_at = $created_at,
                    n.name_embedding = $name_embedding,
                    n.summary = $summary
                RETURN n.uuid AS uuid
            `
	default: // Neo4j
		return `
                MERGE (n:Community {uuid: $uuid})
                SET n = {uuid: $uuid, name: $name, group_id: $group_id, summary: $summary, created_at: $created_at}
                WITH n CALL db.create.setNodeVectorProperty(n, "name_embedding", $name_embedding)
                RETURN n.uuid AS uuid
            `
	}
}

// COMMUNITY_NODE_RETURN query constant
const COMMUNITY_NODE_RETURN = `
    c.uuid AS uuid,
    c.name AS name,
    c.group_id AS group_id,
    c.created_at AS created_at,
    c.name_embedding AS name_embedding,
    c.summary AS summary
`

// COMMUNITY_NODE_RETURN_NEPTUNE query constant for Neptune provider
const COMMUNITY_NODE_RETURN_NEPTUNE = `
    n.uuid AS uuid,
    n.name AS name,
    [x IN split(n.name_embedding, ",") | toFloat(x)] AS name_embedding,
    n.group_id AS group_id,
    n.summary AS summary,
    n.created_at AS created_at
`
