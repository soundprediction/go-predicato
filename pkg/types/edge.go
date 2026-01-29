package types

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

// GraphProvider represents the type of graph database provider.
// This is the canonical definition. pkg/driver exports this type via type alias
// (driver.GraphProvider = types.GraphProvider) for backward compatibility.
type GraphProvider string

const (
	GraphProviderNeo4j    GraphProvider = "neo4j"
	GraphProviderMemgraph GraphProvider = "memgraph"
	GraphProviderFalkorDB GraphProvider = "falkordb"
	GraphProviderLadybug  GraphProvider = "ladybug"
	GraphProviderNeptune  GraphProvider = "neptune"
)

// EdgeOperations provides methods for edge-related database operations
type EdgeOperations interface {
	ExecuteQuery(ctx context.Context, query string, params map[string]interface{}) (interface{}, interface{}, interface{}, error)
	Provider() GraphProvider
	GetAossClient() interface{}
}

// BaseEdge represents the abstract base class for all edges (equivalent to Python Edge class)
type BaseEdge struct {
	Uuid         string    `json:"uuid"`             // matches Python uuid field
	GroupID      string    `json:"group_id"`         // matches Python group_id
	SourceNodeID string    `json:"source_node_uuid"` // matches Python source_node_uuid
	TargetNodeID string    `json:"target_node_uuid"` // matches Python target_node_uuid
	CreatedAt    time.Time `json:"created_at"`       // matches Python created_at

	// Metadata and common fields
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// EdgeInterface defines the interface that all edge types must implement (equivalent to Python ABC methods)
type EdgeInterface interface {
	Save(ctx context.Context, driver EdgeOperations) error
	GetUUID() string
	GetGroupID() string
	GetSourceNodeUUID() string
	GetTargetNodeUUID() string
	GetCreatedAt() time.Time
}

// Implement EdgeInterface for BaseEdge
func (e *BaseEdge) GetUUID() string           { return e.Uuid }
func (e *BaseEdge) GetGroupID() string        { return e.GroupID }
func (e *BaseEdge) GetSourceNodeUUID() string { return e.SourceNodeID }
func (e *BaseEdge) GetTargetNodeUUID() string { return e.TargetNodeID }
func (e *BaseEdge) GetCreatedAt() time.Time   { return e.CreatedAt }

// Delete replicates the Python Edge.delete() method
func (e *BaseEdge) Delete(ctx context.Context, driver EdgeOperations) error {
	if driver.Provider() == GraphProviderLadybug {
		// ladybug provider logic (lines 56-70 in Python)
		_, _, _, err := driver.ExecuteQuery(ctx, `
			MATCH (n)-[e:MENTIONS|HAS_MEMBER {uuid: $uuid}]->(m)
			DELETE e
		`, map[string]interface{}{
			"uuid": e.Uuid,
		})
		if err != nil {
			return err
		}

		_, _, _, err = driver.ExecuteQuery(ctx, `
			MATCH (e:RelatesToNode_ {uuid: $uuid})
			DETACH DELETE e
		`, map[string]interface{}{
			"uuid": e.Uuid,
		})
		return err
	} else {
		// Non-ladybug provider logic (lines 71-78 in Python)
		_, _, _, err := driver.ExecuteQuery(ctx, `
			MATCH (n)-[e:MENTIONS|RELATES_TO|HAS_MEMBER {uuid: $uuid}]->(m)
			DELETE e
		`, map[string]interface{}{
			"uuid": e.Uuid,
		})

		// TODO: Add AOSS client support if needed
		// if driver.GetAossClient() != nil { ... }

		return err
	}
}

// DeleteByUUIDs replicates the Python Edge.delete_by_uuids() class method
func DeleteEdgesByUUIDs(ctx context.Context, driver EdgeOperations, uuids []string) error {
	if len(uuids) == 0 {
		return nil
	}

	if driver.Provider() == GraphProviderLadybug {
		// ladybug provider logic (lines 91-107 in Python)
		_, _, _, err := driver.ExecuteQuery(ctx, `
			MATCH (n)-[e:MENTIONS|HAS_MEMBER]->(m)
			WHERE e.uuid IN $uuids
			DELETE e
		`, map[string]interface{}{
			"uuids": uuids,
		})
		if err != nil {
			return err
		}

		_, _, _, err = driver.ExecuteQuery(ctx, `
			MATCH (e:RelatesToNode_)
			WHERE e.uuid IN $uuids
			DETACH DELETE e
		`, map[string]interface{}{
			"uuids": uuids,
		})
		return err
	} else {
		// Non-ladybug provider logic (lines 108-116 in Python)
		_, _, _, err := driver.ExecuteQuery(ctx, `
			MATCH (n)-[e:MENTIONS|RELATES_TO|HAS_MEMBER]->(m)
			WHERE e.uuid IN $uuids
			DELETE e
		`, map[string]interface{}{
			"uuids": uuids,
		})

		// TODO: Add AOSS client support if needed
		// if driver.GetAossClient() != nil { ... }

		return err
	}
}

// EpisodicEdge represents edges between episodes and entities (equivalent to Python EpisodicEdge)
type EpisodicEdge struct {
	BaseEdge
}

// Save implements the Python EpisodicEdge.save() method
func (e *EpisodicEdge) Save(ctx context.Context, driver EdgeOperations) error {
	_, _, _, err := driver.ExecuteQuery(ctx, "EPISODIC_EDGE_SAVE_QUERY", map[string]interface{}{
		"episode_uuid": e.SourceNodeID,
		"entity_uuid":  e.TargetNodeID,
		"uuid":         e.Uuid,
		"group_id":     e.GroupID,
		"created_at":   e.CreatedAt,
	})
	return err
}

// GetByUUID implements the Python EpisodicEdge.get_by_uuid() class method
func GetEpisodicEdgeByUUID(ctx context.Context, driver EdgeOperations, uuid string) (*EpisodicEdge, error) {
	records, _, _, err := driver.ExecuteQuery(ctx, `
		MATCH (n:Episodic)-[e:MENTIONS {uuid: $uuid}]->(m:Entity)
		RETURN e.uuid AS uuid, e.group_id AS group_id, 
		       n.uuid AS source_node_uuid, m.uuid AS target_node_uuid,
		       e.created_at AS created_at
	`, map[string]interface{}{
		"uuid": uuid,
	})
	if err != nil {
		return nil, err
	}

	recordList, ok := records.([]map[string]interface{})
	if !ok || len(recordList) == 0 {
		return nil, fmt.Errorf("episodic edge with UUID %s not found", uuid)
	}

	record := recordList[0]
	return &EpisodicEdge{
		BaseEdge: BaseEdge{
			Uuid:         record["uuid"].(string),
			GroupID:      record["group_id"].(string),
			SourceNodeID: record["source_node_uuid"].(string),
			TargetNodeID: record["target_node_uuid"].(string),
			CreatedAt:    record["created_at"].(time.Time),
		},
	}, nil
}

// GetByUUIDs implements the Python EpisodicEdge.get_by_uuids() class method
func GetEpisodicEdgesByUUIDs(ctx context.Context, driver EdgeOperations, uuids []string) ([]*EpisodicEdge, error) {
	if len(uuids) == 0 {
		return []*EpisodicEdge{}, nil
	}

	records, _, _, err := driver.ExecuteQuery(ctx, `
		MATCH (n:Episodic)-[e:MENTIONS]->(m:Entity)
		WHERE e.uuid IN $uuids
		RETURN e.uuid AS uuid, e.group_id AS group_id,
		       n.uuid AS source_node_uuid, m.uuid AS target_node_uuid,
		       e.created_at AS created_at
	`, map[string]interface{}{
		"uuids": uuids,
	})
	if err != nil {
		return nil, err
	}

	var edges []*EpisodicEdge
	if recordList, ok := records.([]map[string]interface{}); ok {
		for _, record := range recordList {
			edge := &EpisodicEdge{
				BaseEdge: BaseEdge{
					Uuid:         record["uuid"].(string),
					GroupID:      record["group_id"].(string),
					SourceNodeID: record["source_node_uuid"].(string),
					TargetNodeID: record["target_node_uuid"].(string),
					CreatedAt:    record["created_at"].(time.Time),
				},
			}
			edges = append(edges, edge)
		}
	}

	return edges, nil
}

// EntityEdge represents relationships between entities (equivalent to Python EntityEdge)
type EntityEdge struct {
	BaseEdge

	// EntityEdge-specific fields (from Python EntityEdge class)
	Name          string                 `json:"name"`                 // matches Python name
	Fact          string                 `json:"fact"`                 // matches Python fact
	FactEmbedding []float32              `json:"fact_embedding"`       // matches Python fact_embedding
	Episodes      []string               `json:"episodes"`             // matches Python episodes
	ExpiredAt     *time.Time             `json:"expired_at,omitempty"` // matches Python expired_at
	ValidAt       *time.Time             `json:"valid_at,omitempty"`   // matches Python valid_at
	InvalidAt     *time.Time             `json:"invalid_at,omitempty"` // matches Python invalid_at
	Attributes    map[string]interface{} `json:"attributes"`           // matches Python attributes

	// Backward compatibility fields (from old Go Edge type)
	Type      EdgeType   `json:"type"`
	SourceID  string     `json:"source_id"` // alias for SourceNodeID uuid
	TargetID  string     `json:"target_id"` // alias for TargetNodeID uuid
	UpdatedAt time.Time  `json:"updated_at"`
	Summary   string     `json:"summary,omitempty"` // alias for Fact
	Strength  float64    `json:"strength,omitempty"`
	Embedding []float32  `json:"embedding,omitempty"` // general embedding field
	ValidFrom time.Time  `json:"valid_from"`
	ValidTo   *time.Time `json:"valid_to,omitempty"`
	SourceIDs []string   `json:"source_ids,omitempty"`
}

// EdgeType represents the type of an edge for backward compatibility
type EdgeType string

const (
	EntityEdgeType    EdgeType = "entity"
	EpisodicEdgeType  EdgeType = "episodic"
	CommunityEdgeType EdgeType = "community"
	SourceEdgeType    EdgeType = "source"
)

// Sync fields to maintain backward compatibility
func (e *EntityEdge) syncFields() {
	e.SourceID = e.SourceNodeID
	e.TargetID = e.TargetNodeID
	e.Summary = e.Fact
	if e.ValidAt != nil {
		e.ValidFrom = *e.ValidAt
	}
	if e.InvalidAt != nil {
		e.ValidTo = e.InvalidAt
	}
	e.Type = EntityEdgeType
}

// UpdateFromCompat updates the canonical fields from backward compatibility fields
func (e *EntityEdge) updateFromCompat() {
	if e.SourceID != "" {
		e.SourceNodeID = e.SourceID
	}
	if e.TargetID != "" {
		e.TargetNodeID = e.TargetID
	}
	if e.Summary != "" {
		e.Fact = e.Summary
	}
	if !e.ValidFrom.IsZero() {
		e.ValidAt = &e.ValidFrom
	}
	if e.ValidTo != nil {
		e.InvalidAt = e.ValidTo
	}
}

// GenerateEmbedding implements the Python EntityEdge.generate_embedding() method
func (e *EntityEdge) GenerateEmbedding(ctx context.Context, embedder interface{}) error {
	// TODO: Implement embedder interface and logic
	// text := strings.ReplaceAll(e.Fact, "\n", " ")
	// e.FactEmbedding = await embedder.Create([]string{text})
	return nil
}

// NewEntityEdge creates a new EntityEdge with backward compatibility
func NewEntityEdge(id, sourceID, targetID, groupID, name string, edgeType EdgeType) *EntityEdge {
	now := time.Now()
	edge := &EntityEdge{
		BaseEdge: BaseEdge{
			Uuid:         id,
			GroupID:      groupID,
			SourceNodeID: sourceID,
			TargetNodeID: targetID,
			CreatedAt:    now,
		},
		Type:     edgeType,
		SourceID: sourceID,
		TargetID: targetID,
		Name:     name,
		Summary:  name,
		Fact:     name,
	}
	return edge
}

// Save implements the Python EntityEdge.save() method
func (e *EntityEdge) Save(ctx context.Context, driver EdgeOperations) error {
	edgeData := map[string]interface{}{
		"source_uuid":    e.SourceNodeID,
		"target_uuid":    e.TargetNodeID,
		"uuid":           e.Uuid,
		"name":           e.Name,
		"group_id":       e.GroupID,
		"fact":           e.Fact,
		"fact_embedding": e.FactEmbedding,
		"episodes":       e.Episodes,
		"created_at":     e.CreatedAt,
		"expired_at":     e.ExpiredAt,
		"valid_at":       e.ValidAt,
		"invalid_at":     e.InvalidAt,
	}

	if driver.Provider() == GraphProviderLadybug {
		// ladybug-specific logic (lines 320-325 in Python)
		attributesJSON, _ := json.Marshal(e.Attributes)
		edgeData["attributes"] = string(attributesJSON)

		_, _, _, err := driver.ExecuteQuery(ctx, "ENTITY_EDGE_SAVE_QUERY_ladybug", edgeData)
		return err
	} else {
		// Non-ladybug logic (lines 326-335 in Python)
		for k, v := range e.Attributes {
			edgeData[k] = v
		}

		// TODO: Add AOSS client support if needed
		// if driver.GetAossClient() != nil { ... }

		_, _, _, err := driver.ExecuteQuery(ctx, "ENTITY_EDGE_SAVE_QUERY", map[string]interface{}{
			"edge_data": edgeData,
		})
		return err
	}
}

// GetByUUID implements the Python EntityEdge.get_by_uuid() class method
func GetEntityEdgeByUUID(ctx context.Context, driver EdgeOperations, uuid string) (*EntityEdge, error) {
	var query string
	if driver.Provider() == GraphProviderLadybug {
		query = `
			MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: $uuid})-[:RELATES_TO]->(m:Entity)
			RETURN e.uuid AS uuid, e.name AS name, e.fact AS fact, e.group_id AS group_id,
			       e.episodes AS episodes, e.created_at AS created_at, e.expired_at AS expired_at,
			       e.valid_at AS valid_at, e.invalid_at AS invalid_at, e.attributes AS attributes,
			       n.uuid AS source_node_uuid, m.uuid AS target_node_uuid
		`
	} else {
		query = `
			MATCH (n:Entity)-[e:RELATES_TO {uuid: $uuid}]->(m:Entity)
			RETURN e.uuid AS uuid, e.name AS name, e.fact AS fact, e.group_id AS group_id,
			       e.episodes AS episodes, e.created_at AS created_at, e.expired_at AS expired_at,
			       e.valid_at AS valid_at, e.invalid_at AS invalid_at, e AS attributes,
			       n.uuid AS source_node_uuid, m.uuid AS target_node_uuid
		`
	}

	records, _, _, err := driver.ExecuteQuery(ctx, query, map[string]interface{}{
		"uuid": uuid,
	})
	if err != nil {
		return nil, err
	}

	recordList, ok := records.([]map[string]interface{})
	if !ok || len(recordList) == 0 {
		return nil, fmt.Errorf("entity edge with UUID %s not found", uuid)
	}

	return buildEntityEdgeFromRecord(recordList[0], driver.Provider()), nil
}

// GetEntityEdgesByUUIDs implements the Python EntityEdge.get_by_uuids() class method
func GetEntityEdgesByUUIDs(ctx context.Context, driver EdgeOperations, uuids []string) ([]*EntityEdge, error) {
	if len(uuids) == 0 {
		return []*EntityEdge{}, nil
	}

	var query string
	if driver.Provider() == GraphProviderLadybug {
		query = `
			MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)
			WHERE e.uuid IN $uuids
			RETURN e.uuid AS uuid, e.name AS name, e.fact AS fact, e.group_id AS group_id,
			       e.episodes AS episodes, e.created_at AS created_at, e.expired_at AS expired_at,
			       e.valid_at AS valid_at, e.invalid_at AS invalid_at, e.attributes AS attributes,
			       n.uuid AS source_node_uuid, m.uuid AS target_node_uuid
		`
	} else {
		query = `
			MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
			WHERE e.uuid IN $uuids
			RETURN e.uuid AS uuid, e.name AS name, e.fact AS fact, e.group_id AS group_id,
			       e.episodes AS episodes, e.created_at AS created_at, e.expired_at AS expired_at,
			       e.valid_at AS valid_at, e.invalid_at AS invalid_at, e AS attributes,
			       n.uuid AS source_node_uuid, m.uuid AS target_node_uuid
		`
	}

	records, _, _, err := driver.ExecuteQuery(ctx, query, map[string]interface{}{
		"uuids": uuids,
	})
	if err != nil {
		return nil, err
	}

	var edges []*EntityEdge
	if recordList, ok := records.([]map[string]interface{}); ok {
		for _, record := range recordList {
			edge := buildEntityEdgeFromRecord(record, driver.Provider())
			edges = append(edges, edge)
		}
	}

	return edges, nil
}

// GetBetweenNodes implements the Python EntityEdge.get_between_nodes() class method
func GetEntityEdgesBetweenNodes(ctx context.Context, driver EdgeOperations, sourceNodeUUID, targetNodeUUID string) ([]*EntityEdge, error) {
	var query string
	if driver.Provider() == GraphProviderLadybug {
		query = `
			MATCH (n:Entity {uuid: $source_node_uuid})
			      -[:RELATES_TO]->(e:RelatesToNode_)
			      -[:RELATES_TO]->(m:Entity {uuid: $target_node_uuid})
			RETURN e.uuid AS uuid, e.name AS name, e.fact AS fact, e.group_id AS group_id,
			       e.episodes AS episodes, e.created_at AS created_at, e.expired_at AS expired_at,
			       e.valid_at AS valid_at, e.invalid_at AS invalid_at, e.attributes AS attributes,
			       n.uuid AS source_node_uuid, m.uuid AS target_node_uuid
		`
	} else {
		query = `
			MATCH (n:Entity {uuid: $source_node_uuid})-[e:RELATES_TO]->(m:Entity {uuid: $target_node_uuid})
			RETURN e.uuid AS uuid, e.name AS name, e.fact AS fact, e.group_id AS group_id,
			       e.episodes AS episodes, e.created_at AS created_at, e.expired_at AS expired_at,
			       e.valid_at AS valid_at, e.invalid_at AS invalid_at, e AS attributes,
			       n.uuid AS source_node_uuid, m.uuid AS target_node_uuid
		`
	}

	records, _, _, err := driver.ExecuteQuery(ctx, query, map[string]interface{}{
		"source_node_uuid": sourceNodeUUID,
		"target_node_uuid": targetNodeUUID,
	})
	if err != nil {
		return nil, err
	}

	var edges []*EntityEdge
	if recordList, ok := records.([]map[string]interface{}); ok {
		for _, record := range recordList {
			edge := buildEntityEdgeFromRecord(record, driver.Provider())
			edges = append(edges, edge)
		}
	}

	return edges, nil
}

// CommunityEdge represents edges between communities and their members (equivalent to Python CommunityEdge)
type CommunityEdge struct {
	BaseEdge
}

// Save implements the Python CommunityEdge.save() method
func (e *CommunityEdge) Save(ctx context.Context, driver EdgeOperations) error {
	_, _, _, err := driver.ExecuteQuery(ctx, "COMMUNITY_EDGE_SAVE_QUERY", map[string]interface{}{
		"community_uuid": e.SourceNodeID,
		"entity_uuid":    e.TargetNodeID,
		"uuid":           e.Uuid,
		"group_id":       e.GroupID,
		"created_at":     e.CreatedAt,
	})
	return err
}

// GetByUUID implements the Python CommunityEdge.get_by_uuid() class method
func GetCommunityEdgeByUUID(ctx context.Context, driver EdgeOperations, uuid string) (*CommunityEdge, error) {
	records, _, _, err := driver.ExecuteQuery(ctx, `
		MATCH (n:Community)-[e:HAS_MEMBER {uuid: $uuid}]->(m)
		RETURN e.uuid AS uuid, e.group_id AS group_id,
		       n.uuid AS source_node_uuid, m.uuid AS target_node_uuid,
		       e.created_at AS created_at
	`, map[string]interface{}{
		"uuid": uuid,
	})
	if err != nil {
		return nil, err
	}

	recordList, ok := records.([]map[string]interface{})
	if !ok || len(recordList) == 0 {
		return nil, fmt.Errorf("community edge with UUID %s not found", uuid)
	}

	record := recordList[0]
	return &CommunityEdge{
		BaseEdge: BaseEdge{
			Uuid:         record["uuid"].(string),
			GroupID:      record["group_id"].(string),
			SourceNodeID: record["source_node_uuid"].(string),
			TargetNodeID: record["target_node_uuid"].(string),
			CreatedAt:    record["created_at"].(time.Time),
		},
	}, nil
}

// GetByUUIDs implements the Python CommunityEdge.get_by_uuids() class method
func GetCommunityEdgesByUUIDs(ctx context.Context, driver EdgeOperations, uuids []string) ([]*CommunityEdge, error) {
	if len(uuids) == 0 {
		return []*CommunityEdge{}, nil
	}

	records, _, _, err := driver.ExecuteQuery(ctx, `
		MATCH (n:Community)-[e:HAS_MEMBER]->(m)
		WHERE e.uuid IN $uuids
		RETURN e.uuid AS uuid, e.group_id AS group_id,
		       n.uuid AS source_node_uuid, m.uuid AS target_node_uuid,
		       e.created_at AS created_at
	`, map[string]interface{}{
		"uuids": uuids,
	})
	if err != nil {
		return nil, err
	}

	var edges []*CommunityEdge
	if recordList, ok := records.([]map[string]interface{}); ok {
		for _, record := range recordList {
			edge := &CommunityEdge{
				BaseEdge: BaseEdge{
					Uuid:         record["uuid"].(string),
					GroupID:      record["group_id"].(string),
					SourceNodeID: record["source_node_uuid"].(string),
					TargetNodeID: record["target_node_uuid"].(string),
					CreatedAt:    record["created_at"].(time.Time),
				},
			}
			edges = append(edges, edge)
		}
	}

	return edges, nil
}

// Helper function to build EntityEdge from database record (equivalent to Python get_entity_edge_from_record)
func buildEntityEdgeFromRecord(record map[string]interface{}, provider GraphProvider) *EntityEdge {
	var attributes map[string]interface{}

	if provider == GraphProviderLadybug {
		// ladybug stores attributes as JSON string
		if attrStr, ok := record["attributes"].(string); ok && attrStr != "" {
			_ = json.Unmarshal([]byte(attrStr), &attributes)
		}
	} else {
		// Other providers store attributes as map, need to filter out standard fields
		if attrMap, ok := record["attributes"].(map[string]interface{}); ok {
			attributes = make(map[string]interface{})
			for k, v := range attrMap {
				// Filter out standard edge fields (matching Python logic lines 603-615)
				switch k {
				case "uuid", "source_node_uuid", "target_node_uuid", "fact", "fact_embedding",
					"name", "group_id", "episodes", "created_at", "expired_at", "valid_at", "invalid_at":
					// Skip standard fields
				default:
					attributes[k] = v
				}
			}
		}
	}

	episodes := []string{}
	if episodeList, ok := record["episodes"].([]interface{}); ok {
		for _, ep := range episodeList {
			if epStr, ok := ep.(string); ok {
				episodes = append(episodes, epStr)
			}
		}
	}

	// Handle optional time fields
	var expiredAt, validAt, invalidAt *time.Time
	if t, ok := record["expired_at"].(time.Time); ok {
		expiredAt = &t
	}
	if t, ok := record["valid_at"].(time.Time); ok {
		validAt = &t
	}
	if t, ok := record["invalid_at"].(time.Time); ok {
		invalidAt = &t
	}

	return &EntityEdge{
		BaseEdge: BaseEdge{
			Uuid:         record["uuid"].(string),
			GroupID:      record["group_id"].(string),
			SourceNodeID: record["source_node_uuid"].(string),
			TargetNodeID: record["target_node_uuid"].(string),
			CreatedAt:    record["created_at"].(time.Time),
		},
		Name:          record["name"].(string),
		Fact:          record["fact"].(string),
		FactEmbedding: convertToFloat32Array(record["fact_embedding"]),
		Episodes:      episodes,
		ExpiredAt:     expiredAt,
		ValidAt:       validAt,
		InvalidAt:     invalidAt,
		Attributes:    attributes,
	}
}

// Helper function to convert embedding to []float32
func convertToFloat32Array(embedding interface{}) []float32 {
	if embedding == nil {
		return nil
	}

	if floatList, ok := embedding.([]interface{}); ok {
		result := make([]float32, len(floatList))
		for i, f := range floatList {
			if fVal, ok := f.(float64); ok {
				result[i] = float32(fVal)
			}
		}
		return result
	}

	return nil
}
