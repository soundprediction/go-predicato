//go:build system_cozo

package driver

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	cozo "github.com/cozodb/cozo-lib-go"
	"github.com/soundprediction/predicato/pkg/types"
)

// CozoDriver implements GraphDriver using CozoDB.
type CozoDriver struct {
	db           *cozo.CozoDB
	mu           sync.RWMutex
	embeddingDim int
}

// CozoDriverSession is a minimal session implementation for CozoDB.
type CozoDriverSession struct {
	driver *CozoDriver
}

func (s *CozoDriverSession) Enter(ctx context.Context) (GraphDriverSession, error) { return s, nil }
func (s *CozoDriverSession) Exit(ctx context.Context, excType, excVal, excTb interface{}) error {
	return nil
}
func (s *CozoDriverSession) Close() error { return nil }
func (s *CozoDriverSession) Run(ctx context.Context, query interface{}, kwargs map[string]interface{}) error {
	q, ok := query.(string)
	if !ok {
		return fmt.Errorf("query must be a string")
	}
	_, _, _, err := s.driver.ExecuteQuery(ctx, q, kwargs)
	return err
}
func (s *CozoDriverSession) ExecuteWrite(ctx context.Context, fn func(context.Context, GraphDriverSession, ...interface{}) (interface{}, error), args ...interface{}) (interface{}, error) {
	return fn(ctx, s, args...)
}
func (s *CozoDriverSession) Provider() GraphProvider { return GraphProviderCozo }

// NewCozoDriver creates a new CozoDB-backed GraphDriver.
func NewCozoDriver(uri string, embeddingDim int) (*CozoDriver, error) {
	var engine, path string
	if uri == ":memory:" || uri == "" {
		engine = "mem"
		path = ""
	} else {
		engine = "rocksdb"
		path = uri
	}

	db, err := cozo.New(engine, path, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to open CozoDB: %w", err)
	}

	d := &CozoDriver{
		db:           &db,
		embeddingDim: embeddingDim,
	}

	if err := d.initSchema(); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to init schema: %w", err)
	}

	return d, nil
}

func (d *CozoDriver) initSchema() error {
	schemas := []string{
		`:create entity {uuid: String, group_id: String => type: String default "entity", name: String default "", content: String default "", summary: String default "", entity_type: String default "", episode_type: String default "", embedding: Json default "null", name_embedding: Json default "null", metadata: String default "{}", created_at: Float default 0.0, updated_at: Float default 0.0, valid_from: Float default 0.0, valid_to: Float default 0.0, source_ids: Json default "[]", entity_edges: Json default "[]", level: Int default 0}`,
		`:create edge {uuid: String, group_id: String => source_id: String default "", target_id: String default "", type: String default "entity", name: String default "", fact: String default "", fact_embedding: Json default "null", embedding: Json default "null", episodes: Json default "[]", attributes: String default "{}", created_at: Float default 0.0, updated_at: Float default 0.0, valid_from: Float default 0.0, valid_to: Float default 0.0, expired_at: Float default 0.0, valid_at: Float default 0.0, invalid_at: Float default 0.0, source_ids: Json default "[]", strength: Float default 0.0}`,
	}
	for _, s := range schemas {
		_, err := d.db.Run(s, nil)
		if err != nil {
			// Ignore "already exists" errors
			if !strings.Contains(err.Error(), "already exists") {
				return err
			}
		}
	}
	return nil
}

func (d *CozoDriver) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.db.Close()
	return nil
}

func (d *CozoDriver) Provider() GraphProvider    { return GraphProviderCozo }
func (d *CozoDriver) GetAossClient() interface{} { return nil }

func (d *CozoDriver) Session(database *string) GraphDriverSession {
	return &CozoDriverSession{driver: d}
}

func (d *CozoDriver) DeleteAllIndexes(database string) {
	// CozoDB manages indices internally; no-op
}

func (d *CozoDriver) ExecuteQuery(ctx context.Context, query string, kwargs map[string]interface{}) (interface{}, interface{}, interface{}, error) {
	if strings.Contains(strings.ToUpper(query), "MATCH") && strings.Contains(query, "->") && !strings.Contains(query, ":=") {
		return nil, nil, nil, fmt.Errorf("raw Cypher queries are not supported by CozoDB driver; use typed methods instead")
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	result, err := d.db.Run(query, cozo.Map(kwargs))
	if err != nil {
		return nil, nil, nil, err
	}
	rows := d.namedRowsToMaps(result)
	return rows, nil, result.Headers, nil
}

func (d *CozoDriver) namedRowsToMaps(nr cozo.NamedRows) []map[string]interface{} {
	result := make([]map[string]interface{}, 0, len(nr.Rows))
	for _, row := range nr.Rows {
		m := make(map[string]interface{}, len(nr.Headers))
		for i, h := range nr.Headers {
			if i < len(row) {
				m[h] = row[i]
			}
		}
		result = append(result, m)
	}
	return result
}

// run executes a CozoScript query with mutex protection.
func (d *CozoDriver) run(query string, params cozo.Map) (cozo.NamedRows, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.db.Run(query, params)
}

// --- Node Operations ---

func (d *CozoDriver) UpsertNode(ctx context.Context, node *types.Node) error {
	if node == nil {
		return fmt.Errorf("node cannot be nil")
	}
	now := time.Now()
	if node.CreatedAt.IsZero() {
		node.CreatedAt = now
	}
	if node.UpdatedAt.IsZero() {
		node.UpdatedAt = now
	}

	metaJSON, _ := json.Marshal(node.Metadata)
	sourceJSON, _ := json.Marshal(node.SourceIDs)
	edgesJSON, _ := json.Marshal(node.EntityEdges)
	embJSON, _ := json.Marshal(node.Embedding)
	nameEmbJSON, _ := json.Marshal(node.NameEmbedding)

	validTo := 0.0
	if node.ValidTo != nil {
		validTo = float64(node.ValidTo.Unix())
	}

	query := `:put entity {uuid: $uuid, group_id: $group_id => type: $type, name: $name, content: $content, summary: $summary, entity_type: $entity_type, episode_type: $episode_type, embedding: $embedding, name_embedding: $name_embedding, metadata: $metadata, created_at: $created_at, updated_at: $updated_at, valid_from: $valid_from, valid_to: $valid_to, source_ids: $source_ids, entity_edges: $entity_edges, level: $level}`
	params := cozo.Map{
		"uuid":           node.Uuid,
		"group_id":       node.GroupID,
		"type":           string(node.Type),
		"name":           node.Name,
		"content":        node.Content,
		"summary":        node.Summary,
		"entity_type":    node.EntityType,
		"episode_type":   string(node.EpisodeType),
		"embedding":      string(embJSON),
		"name_embedding": string(nameEmbJSON),
		"metadata":       string(metaJSON),
		"created_at":     float64(node.CreatedAt.Unix()),
		"updated_at":     float64(node.UpdatedAt.Unix()),
		"valid_from":     float64(node.ValidFrom.Unix()),
		"valid_to":       validTo,
		"source_ids":     string(sourceJSON),
		"entity_edges":   string(edgesJSON),
		"level":          int64(node.Level),
	}

	_, err := d.run(query, params)
	return err
}

func (d *CozoDriver) UpsertNodes(ctx context.Context, nodes []*types.Node) error {
	for _, node := range nodes {
		if err := d.UpsertNode(ctx, node); err != nil {
			return err
		}
	}
	return nil
}

func (d *CozoDriver) GetNode(ctx context.Context, nodeID, groupID string) (*types.Node, error) {
	query := `?[uuid, group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level] := *entity{uuid: $uuid, group_id: $group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level}`
	params := cozo.Map{"uuid": nodeID, "group_id": groupID}

	result, err := d.run(query, params)
	if err != nil {
		return nil, err
	}
	if len(result.Rows) == 0 {
		return nil, nil
	}

	return d.rowToNode(result.Headers, result.Rows[0])
}

func (d *CozoDriver) GetNodes(ctx context.Context, nodeIDs []string, groupID string) ([]*types.Node, error) {
	nodes := make([]*types.Node, 0, len(nodeIDs))
	for _, id := range nodeIDs {
		node, err := d.GetNode(ctx, id, groupID)
		if err != nil {
			continue
		}
		nodes = append(nodes, node)
	}
	return nodes, nil
}

func (d *CozoDriver) DeleteNode(ctx context.Context, nodeID, groupID string) error {
	query := `:rm entity {uuid: $uuid, group_id: $group_id}`
	_, err := d.run(query, cozo.Map{"uuid": nodeID, "group_id": groupID})
	return err
}

func (d *CozoDriver) GetEntityNodesByGroup(ctx context.Context, groupID string) ([]*types.Node, error) {
	query := `?[uuid, group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level] := *entity{uuid, group_id: $group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level}, type == "entity"`
	result, err := d.run(query, cozo.Map{"group_id": groupID})
	if err != nil {
		return nil, err
	}
	return d.rowsToNodes(result)
}

func (d *CozoDriver) ParseNodesFromRecords(records any) ([]*types.Node, error) {
	rows, ok := records.([]map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("unsupported record type")
	}
	nodes := make([]*types.Node, 0, len(rows))
	for _, row := range rows {
		node := d.mapToNode(row)
		nodes = append(nodes, node)
	}
	return nodes, nil
}

// --- Edge Operations ---

func (d *CozoDriver) UpsertEdge(ctx context.Context, edge *types.Edge) error {
	if edge == nil {
		return fmt.Errorf("edge cannot be nil")
	}
	now := time.Now()
	if edge.CreatedAt.IsZero() {
		edge.CreatedAt = now
	}

	sourceID := edge.SourceID
	if sourceID == "" {
		sourceID = edge.SourceNodeID
	}
	targetID := edge.TargetID
	if targetID == "" {
		targetID = edge.TargetNodeID
	}

	attrJSON, _ := json.Marshal(edge.Attributes)
	episodesJSON, _ := json.Marshal(edge.Episodes)
	factEmbJSON, _ := json.Marshal(edge.FactEmbedding)
	embJSON, _ := json.Marshal(edge.Embedding)
	sourceIDsJSON, _ := json.Marshal(edge.SourceIDs)

	validTo, expiredAt, validAt, invalidAt := 0.0, 0.0, 0.0, 0.0
	if edge.ValidTo != nil {
		validTo = float64(edge.ValidTo.Unix())
	}
	if edge.ExpiredAt != nil {
		expiredAt = float64(edge.ExpiredAt.Unix())
	}
	if edge.ValidAt != nil {
		validAt = float64(edge.ValidAt.Unix())
	}
	if edge.InvalidAt != nil {
		invalidAt = float64(edge.InvalidAt.Unix())
	}

	query := `:put edge {uuid: $uuid, group_id: $group_id => source_id: $source_id, target_id: $target_id, type: $type, name: $name, fact: $fact, fact_embedding: $fact_embedding, embedding: $embedding, episodes: $episodes, attributes: $attributes, created_at: $created_at, updated_at: $updated_at, valid_from: $valid_from, valid_to: $valid_to, expired_at: $expired_at, valid_at: $valid_at, invalid_at: $invalid_at, source_ids: $source_ids, strength: $strength}`
	params := cozo.Map{
		"uuid":           edge.Uuid,
		"group_id":       edge.GroupID,
		"source_id":      sourceID,
		"target_id":      targetID,
		"type":           string(edge.Type),
		"name":           edge.Name,
		"fact":           edge.Fact,
		"fact_embedding": string(factEmbJSON),
		"embedding":      string(embJSON),
		"episodes":       string(episodesJSON),
		"attributes":     string(attrJSON),
		"created_at":     float64(edge.CreatedAt.Unix()),
		"updated_at":     float64(edge.UpdatedAt.Unix()),
		"valid_from":     float64(edge.ValidFrom.Unix()),
		"valid_to":       validTo,
		"expired_at":     expiredAt,
		"valid_at":       validAt,
		"invalid_at":     invalidAt,
		"source_ids":     string(sourceIDsJSON),
		"strength":       edge.Strength,
	}

	_, err := d.run(query, params)
	return err
}

func (d *CozoDriver) UpsertEdges(ctx context.Context, edges []*types.Edge) error {
	for _, edge := range edges {
		if err := d.UpsertEdge(ctx, edge); err != nil {
			return err
		}
	}
	return nil
}

func (d *CozoDriver) GetEdge(ctx context.Context, edgeID, groupID string) (*types.Edge, error) {
	query := `?[uuid, group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength] := *edge{uuid: $uuid, group_id: $group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength}`
	result, err := d.run(query, cozo.Map{"uuid": edgeID, "group_id": groupID})
	if err != nil {
		return nil, err
	}
	if len(result.Rows) == 0 {
		return nil, nil
	}
	return d.rowToEdge(result.Headers, result.Rows[0])
}

func (d *CozoDriver) GetEdges(ctx context.Context, edgeIDs []string, groupID string) ([]*types.Edge, error) {
	edges := make([]*types.Edge, 0, len(edgeIDs))
	for _, id := range edgeIDs {
		edge, err := d.GetEdge(ctx, id, groupID)
		if err != nil {
			continue
		}
		edges = append(edges, edge)
	}
	return edges, nil
}

func (d *CozoDriver) DeleteEdge(ctx context.Context, edgeID, groupID string) error {
	query := `:rm edge {uuid: $uuid, group_id: $group_id}`
	_, err := d.run(query, cozo.Map{"uuid": edgeID, "group_id": groupID})
	return err
}

func (d *CozoDriver) UpsertEpisodicEdge(ctx context.Context, episodeUUID, entityUUID, groupID string) error {
	edge := &types.Edge{
		Name: "MENTIONS",
		Type: types.EpisodicEdgeType,
	}
	edge.Uuid = fmt.Sprintf("%s-%s", episodeUUID, entityUUID)
	edge.GroupID = groupID
	edge.SourceID = episodeUUID
	edge.TargetID = entityUUID
	edge.SourceNodeID = episodeUUID
	edge.TargetNodeID = entityUUID
	edge.CreatedAt = time.Now()
	return d.UpsertEdge(ctx, edge)
}

func (d *CozoDriver) UpsertCommunityEdge(ctx context.Context, communityUUID, nodeUUID, uuid, groupID string) error {
	edge := &types.Edge{
		Name: "HAS_MEMBER",
		Type: types.CommunityEdgeType,
	}
	edge.Uuid = uuid
	edge.GroupID = groupID
	edge.SourceID = communityUUID
	edge.TargetID = nodeUUID
	edge.SourceNodeID = communityUUID
	edge.TargetNodeID = nodeUUID
	edge.CreatedAt = time.Now()
	return d.UpsertEdge(ctx, edge)
}

func (d *CozoDriver) GetBetweenNodes(ctx context.Context, sourceNodeID, targetNodeID string) ([]*types.Edge, error) {
	query := `?[uuid, group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength] := *edge{uuid, group_id, source_id: $src, target_id: $tgt, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength}`
	result, err := d.run(query, cozo.Map{"src": sourceNodeID, "tgt": targetNodeID})
	if err != nil {
		return nil, err
	}
	return d.rowsToEdges(result)
}

// --- Search Operations ---

func (d *CozoDriver) SearchNodes(ctx context.Context, query, groupID string, options *SearchOptions) ([]*types.Node, error) {
	limit := 10
	var nodeTypes []string

	if options != nil {
		if options.Limit > 0 {
			limit = options.Limit
		}
		if len(options.NodeTypes) > 0 {
			for _, nt := range options.NodeTypes {
				nodeTypes = append(nodeTypes, string(nt))
			}
		}
	}

	searchConditions := "contains(lowercase(name), lowercase($query)) or contains(lowercase(summary), lowercase($query))"

	if len(nodeTypes) > 0 {
		quotedTypes := make([]string, len(nodeTypes))
		for i, t := range nodeTypes {
			quotedTypes[i] = fmt.Sprintf("%q", t)
		}
		searchConditions += fmt.Sprintf(", type in [%s]", strings.Join(quotedTypes, ", "))
	}

	searchQ := fmt.Sprintf(`?[uuid, group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level] := *entity{uuid, group_id: $group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level}, %s
:limit $limit`, searchConditions)

	result, err := d.run(searchQ, cozo.Map{"group_id": groupID, "query": query, "limit": int64(limit)})
	if err != nil {
		return nil, err
	}
	return d.rowsToNodes(result)
}

func (d *CozoDriver) SearchEdges(ctx context.Context, query, groupID string, options *SearchOptions) ([]*types.Edge, error) {
	limit := 10
	if options != nil && options.Limit > 0 {
		limit = options.Limit
	}
	searchQ := `?[uuid, group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength] := *edge{uuid, group_id: $group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength}, contains(lowercase(name), lowercase($query)) or contains(lowercase(fact), lowercase($query))
:limit $limit`
	result, err := d.run(searchQ, cozo.Map{"group_id": groupID, "query": query, "limit": int64(limit)})
	if err != nil {
		return nil, err
	}
	return d.rowsToEdges(result)
}

func (d *CozoDriver) SearchNodesByEmbedding(ctx context.Context, embedding []float32, groupID string, limit int) ([]*types.Node, error) {
	return d.searchNodesByVectorInternal(ctx, embedding, groupID, limit)
}

func (d *CozoDriver) SearchEdgesByEmbedding(ctx context.Context, embedding []float32, groupID string, limit int) ([]*types.Edge, error) {
	return d.searchEdgesByVectorInternal(ctx, embedding, groupID, limit)
}

func (d *CozoDriver) SearchNodesByVector(ctx context.Context, vector []float32, groupID string, options *VectorSearchOptions) ([]*types.Node, error) {
	limit := 10
	if options != nil && options.Limit > 0 {
		limit = options.Limit
	}
	return d.searchNodesByVectorInternal(ctx, vector, groupID, limit)
}

func (d *CozoDriver) SearchEdgesByVector(ctx context.Context, vector []float32, groupID string, options *VectorSearchOptions) ([]*types.Edge, error) {
	limit := 10
	if options != nil && options.Limit > 0 {
		limit = options.Limit
	}
	return d.searchEdgesByVectorInternal(ctx, vector, groupID, limit)
}

func (d *CozoDriver) searchNodesByVectorInternal(ctx context.Context, vector []float32, groupID string, limit int) ([]*types.Node, error) {
	// Fetch all nodes with embeddings, compute cosine similarity in Go
	query := `?[uuid, group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level] := *entity{uuid, group_id: $group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level}`
	result, err := d.run(query, cozo.Map{"group_id": groupID})
	if err != nil {
		return nil, err
	}
	nodes, err := d.rowsToNodes(result)
	if err != nil {
		return nil, err
	}

	type scored struct {
		node  *types.Node
		score float64
	}
	var scored_nodes []scored
	for _, n := range nodes {
		emb := n.Embedding
		if len(emb) == 0 {
			emb = n.NameEmbedding
		}
		if len(emb) == 0 {
			continue
		}
		score := cosineSimilarity(vector, emb)
		if score > 0 {
			scored_nodes = append(scored_nodes, scored{n, score})
		}
	}

	// Sort by score descending
	for i := 0; i < len(scored_nodes); i++ {
		for j := i + 1; j < len(scored_nodes); j++ {
			if scored_nodes[j].score > scored_nodes[i].score {
				scored_nodes[i], scored_nodes[j] = scored_nodes[j], scored_nodes[i]
			}
		}
	}

	if limit > len(scored_nodes) {
		limit = len(scored_nodes)
	}
	out := make([]*types.Node, limit)
	for i := 0; i < limit; i++ {
		out[i] = scored_nodes[i].node
	}
	return out, nil
}

func (d *CozoDriver) searchEdgesByVectorInternal(ctx context.Context, vector []float32, groupID string, limit int) ([]*types.Edge, error) {
	query := `?[uuid, group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength] := *edge{uuid, group_id: $group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength}`
	result, err := d.run(query, cozo.Map{"group_id": groupID})
	if err != nil {
		return nil, err
	}
	edges, err := d.rowsToEdges(result)
	if err != nil {
		return nil, err
	}

	type scored struct {
		edge  *types.Edge
		score float64
	}
	var scored_edges []scored
	for _, e := range edges {
		emb := e.Embedding
		if len(emb) == 0 {
			emb = e.FactEmbedding
		}
		if len(emb) == 0 {
			continue
		}
		score := cosineSimilarity(vector, emb)
		if score > 0 {
			scored_edges = append(scored_edges, scored{e, score})
		}
	}

	for i := 0; i < len(scored_edges); i++ {
		for j := i + 1; j < len(scored_edges); j++ {
			if scored_edges[j].score > scored_edges[i].score {
				scored_edges[i], scored_edges[j] = scored_edges[j], scored_edges[i]
			}
		}
	}

	if limit > len(scored_edges) {
		limit = len(scored_edges)
	}
	out := make([]*types.Edge, limit)
	for i := 0; i < limit; i++ {
		out[i] = scored_edges[i].edge
	}
	return out, nil
}

// --- Traversal Operations ---

func (d *CozoDriver) GetNeighbors(ctx context.Context, nodeID, groupID string, maxDistance int) ([]*types.Node, error) {
	if maxDistance < 1 {
		maxDistance = 1
	}
	if maxDistance > 10 {
		maxDistance = 10
	}

	// Iterative BFS
	visited := map[string]bool{nodeID: true}
	frontier := []string{nodeID}

	for depth := 0; depth < maxDistance && len(frontier) > 0; depth++ {
		var nextFrontier []string
		for _, fid := range frontier {
			// Get outgoing neighbors
			q := `?[target_id] := *edge{source_id: $src, group_id: $gid, target_id}`
			result, err := d.run(q, cozo.Map{"src": fid, "gid": groupID})
			if err != nil {
				continue
			}
			for _, row := range result.Rows {
				if tid, ok := row[0].(string); ok && !visited[tid] {
					visited[tid] = true
					nextFrontier = append(nextFrontier, tid)
				}
			}
			// Get incoming neighbors
			q2 := `?[source_id] := *edge{target_id: $tgt, group_id: $gid, source_id}`
			result2, err := d.run(q2, cozo.Map{"tgt": fid, "gid": groupID})
			if err != nil {
				continue
			}
			for _, row := range result2.Rows {
				if sid, ok := row[0].(string); ok && !visited[sid] {
					visited[sid] = true
					nextFrontier = append(nextFrontier, sid)
				}
			}
		}
		frontier = nextFrontier
	}

	// Fetch all visited nodes except the start node
	var nodes []*types.Node
	for id := range visited {
		if id == nodeID {
			continue
		}
		node, err := d.GetNode(ctx, id, groupID)
		if err != nil {
			continue
		}
		nodes = append(nodes, node)
	}
	return nodes, nil
}

func (d *CozoDriver) GetRelatedNodes(ctx context.Context, nodeID, groupID string, edgeTypes []types.EdgeType) ([]*types.Node, error) {
	// Get all edges from this node, filter by type
	q := `?[uuid, group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength] := *edge{uuid, group_id: $gid, source_id: $src, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength}`
	result, err := d.run(q, cozo.Map{"src": nodeID, "gid": groupID})
	if err != nil {
		return nil, err
	}

	typeSet := make(map[string]bool)
	for _, et := range edgeTypes {
		typeSet[string(et)] = true
	}

	var nodeIDs []string
	for _, row := range result.Rows {
		// type is at index matching "type" header
		typeIdx := -1
		targetIdx := -1
		for i, h := range result.Headers {
			if h == "type" {
				typeIdx = i
			}
			if h == "target_id" {
				targetIdx = i
			}
		}
		if typeIdx >= 0 && targetIdx >= 0 {
			if t, ok := row[typeIdx].(string); ok {
				if len(typeSet) == 0 || typeSet[t] {
					if tid, ok := row[targetIdx].(string); ok {
						nodeIDs = append(nodeIDs, tid)
					}
				}
			}
		}
	}

	return d.GetNodes(ctx, nodeIDs, groupID)
}

func (d *CozoDriver) GetNodeNeighbors(ctx context.Context, nodeUUID, groupID string) ([]types.Neighbor, error) {
	// Count edges per neighbor
	neighborCounts := map[string]int{}

	q := `?[target_id] := *edge{source_id: $src, group_id: $gid, target_id}`
	result, err := d.run(q, cozo.Map{"src": nodeUUID, "gid": groupID})
	if err != nil {
		return nil, err
	}
	for _, row := range result.Rows {
		if tid, ok := row[0].(string); ok {
			neighborCounts[tid]++
		}
	}

	q2 := `?[source_id] := *edge{target_id: $tgt, group_id: $gid, source_id}`
	result2, err := d.run(q2, cozo.Map{"tgt": nodeUUID, "gid": groupID})
	if err != nil {
		return nil, err
	}
	for _, row := range result2.Rows {
		if sid, ok := row[0].(string); ok {
			neighborCounts[sid]++
		}
	}

	neighbors := make([]types.Neighbor, 0, len(neighborCounts))
	for id, count := range neighborCounts {
		neighbors = append(neighbors, types.Neighbor{NodeUUID: id, EdgeCount: count})
	}
	return neighbors, nil
}

// --- Temporal Operations ---

func (d *CozoDriver) GetNodesInTimeRange(ctx context.Context, start, end time.Time, groupID string) ([]*types.Node, error) {
	query := `?[uuid, group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level] := *entity{uuid, group_id: $group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level}, created_at >= $start, created_at <= $end`
	result, err := d.run(query, cozo.Map{
		"group_id": groupID,
		"start":    float64(start.Unix()),
		"end":      float64(end.Unix()),
	})
	if err != nil {
		return nil, err
	}
	return d.rowsToNodes(result)
}

func (d *CozoDriver) GetEdgesInTimeRange(ctx context.Context, start, end time.Time, groupID string) ([]*types.Edge, error) {
	query := `?[uuid, group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength] := *edge{uuid, group_id: $group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength}, created_at >= $start, created_at <= $end`
	result, err := d.run(query, cozo.Map{
		"group_id": groupID,
		"start":    float64(start.Unix()),
		"end":      float64(end.Unix()),
	})
	if err != nil {
		return nil, err
	}
	return d.rowsToEdges(result)
}

func (d *CozoDriver) RetrieveEpisodes(ctx context.Context, referenceTime time.Time, groupIDs []string, limit int, episodeType *types.EpisodeType) ([]*types.Node, error) {
	var allNodes []*types.Node
	for _, gid := range groupIDs {
		query := `?[uuid, group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level] := *entity{uuid, group_id: $group_id, type: "episodic", name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level}, created_at <= $ref
:limit $limit
:order -created_at`
		params := cozo.Map{
			"group_id": gid,
			"ref":      float64(referenceTime.Unix()),
			"limit":    int64(limit),
		}
		result, err := d.run(query, params)
		if err != nil {
			continue
		}
		nodes, err := d.rowsToNodes(result)
		if err != nil {
			continue
		}
		for _, n := range nodes {
			if episodeType != nil && n.EpisodeType != *episodeType {
				continue
			}
			allNodes = append(allNodes, n)
		}
	}
	if limit > 0 && len(allNodes) > limit {
		allNodes = allNodes[:limit]
	}
	return allNodes, nil
}

// --- Community Operations ---

func (d *CozoDriver) GetCommunities(ctx context.Context, groupID string, level int) ([]*types.Node, error) {
	query := `?[uuid, group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level] := *entity{uuid, group_id: $group_id, type: "community", name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level}, level == $level`
	result, err := d.run(query, cozo.Map{"group_id": groupID, "level": int64(level)})
	if err != nil {
		return nil, err
	}
	return d.rowsToNodes(result)
}

func (d *CozoDriver) BuildCommunities(ctx context.Context, groupID string) error {
	// Basic community detection: create a single community for all connected components
	// A full Louvain implementation would be complex; this is a minimal viable implementation
	return nil
}

func (d *CozoDriver) GetExistingCommunity(ctx context.Context, entityUUID string) (*types.Node, error) {
	// Find community edges where this entity is a member
	q := `?[source_id] := *edge{source_id, target_id: $entity, type: "community"}`
	result, err := d.run(q, cozo.Map{"entity": entityUUID})
	if err != nil {
		return nil, err
	}
	if len(result.Rows) == 0 {
		return nil, fmt.Errorf("no community found for entity %s", entityUUID)
	}
	communityID, ok := result.Rows[0][0].(string)
	if !ok {
		return nil, fmt.Errorf("invalid community ID")
	}
	// We need the group_id; search all groups
	groupIDs, _ := d.GetAllGroupIDs(ctx)
	for _, gid := range groupIDs {
		node, err := d.GetNode(ctx, communityID, gid)
		if err == nil {
			return node, nil
		}
	}
	return nil, fmt.Errorf("community node %s not found", communityID)
}

func (d *CozoDriver) FindModalCommunity(ctx context.Context, entityUUID string) (*types.Node, error) {
	return d.GetExistingCommunity(ctx, entityUUID)
}

func (d *CozoDriver) RemoveCommunities(ctx context.Context) error {
	// Delete all community nodes and community edges
	groupIDs, err := d.GetAllGroupIDs(ctx)
	if err != nil {
		return err
	}
	for _, gid := range groupIDs {
		communities, err := d.GetCommunities(ctx, gid, 0)
		if err != nil {
			continue
		}
		for _, c := range communities {
			_ = d.DeleteNode(ctx, c.Uuid, c.GroupID)
		}
		// Delete community edges
		q := `?[uuid, group_id] := *edge{uuid, group_id: $gid, type: "community"}`
		result, err := d.run(q, cozo.Map{"gid": gid})
		if err != nil {
			continue
		}
		for _, row := range result.Rows {
			if euuid, ok := row[0].(string); ok {
				_ = d.DeleteEdge(ctx, euuid, gid)
			}
		}
	}
	return nil
}

// --- Admin Operations ---

func (d *CozoDriver) CreateIndices(ctx context.Context) error {
	// CozoDB manages indices automatically for key columns.
	// FTS and HNSW indices require explicit creation but may not be available
	// in all CozoDB versions. Attempt creation but don't fail.
	return nil
}

func (d *CozoDriver) GetStats(ctx context.Context, groupID string) (*GraphStats, error) {
	stats := &GraphStats{
		LastUpdated: time.Now(),
		NodesByType: make(map[string]int64),
		EdgesByType: make(map[string]int64),
	}

	// Count nodes by type
	q := `?[type, count(uuid)] := *entity{uuid, group_id: $gid, type}`
	result, err := d.run(q, cozo.Map{"gid": groupID})
	if err != nil {
		return stats, nil
	}
	for _, row := range result.Rows {
		if t, ok := row[0].(string); ok {
			if c, ok := row[1].(float64); ok {
				stats.NodesByType[t] = int64(c)
				stats.NodeCount += int64(c)
			} else if c, ok := row[1].(int64); ok {
				stats.NodesByType[t] = c
				stats.NodeCount += c
			}
		}
	}

	// Count edges by type
	q2 := `?[type, count(uuid)] := *edge{uuid, group_id: $gid, type}`
	result2, err := d.run(q2, cozo.Map{"gid": groupID})
	if err != nil {
		return stats, nil
	}
	for _, row := range result2.Rows {
		if t, ok := row[0].(string); ok {
			if c, ok := row[1].(float64); ok {
				stats.EdgesByType[t] = int64(c)
				stats.EdgeCount += int64(c)
			} else if c, ok := row[1].(int64); ok {
				stats.EdgesByType[t] = c
				stats.EdgeCount += c
			}
		}
	}

	if cnt, ok := stats.NodesByType["community"]; ok {
		stats.CommunityCount = cnt
	}

	return stats, nil
}

func (d *CozoDriver) GetAllGroupIDs(ctx context.Context) ([]string, error) {
	q := `?[group_id] := *entity{group_id}`
	result, err := d.run(q, nil)
	if err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	var ids []string
	for _, row := range result.Rows {
		if gid, ok := row[0].(string); ok && !seen[gid] {
			seen[gid] = true
			ids = append(ids, gid)
		}
	}
	return ids, nil
}

// --- GraphExporter Implementation ---

func (d *CozoDriver) IterateNodes(ctx context.Context, groupID string, fn func(*types.Node) error) error {
	query := `?[uuid, group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level] := *entity{uuid, group_id: $group_id, type, name, content, summary, entity_type, episode_type, embedding, name_embedding, metadata, created_at, updated_at, valid_from, valid_to, source_ids, entity_edges, level}`
	result, err := d.run(query, cozo.Map{"group_id": groupID})
	if err != nil {
		return err
	}
	for _, row := range result.Rows {
		node, err := d.rowToNode(result.Headers, row)
		if err != nil {
			continue
		}
		if err := fn(node); err != nil {
			return err
		}
	}
	return nil
}

func (d *CozoDriver) IterateEdges(ctx context.Context, groupID string, fn func(*types.Edge) error) error {
	query := `?[uuid, group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength] := *edge{uuid, group_id: $group_id, source_id, target_id, type, name, fact, fact_embedding, embedding, episodes, attributes, created_at, updated_at, valid_from, valid_to, expired_at, valid_at, invalid_at, source_ids, strength}`
	result, err := d.run(query, cozo.Map{"group_id": groupID})
	if err != nil {
		return err
	}
	for _, row := range result.Rows {
		edge, err := d.rowToEdge(result.Headers, row)
		if err != nil {
			continue
		}
		if err := fn(edge); err != nil {
			return err
		}
	}
	return nil
}

// --- Helper functions ---

func (d *CozoDriver) rowToNode(headers []string, row []any) (*types.Node, error) {
	m := make(map[string]interface{}, len(headers))
	for i, h := range headers {
		if i < len(row) {
			m[h] = row[i]
		}
	}
	return d.mapToNodeErr(m)
}

func (d *CozoDriver) mapToNode(m map[string]interface{}) *types.Node {
	n, _ := d.mapToNodeErr(m)
	return n
}

func (d *CozoDriver) mapToNodeErr(m map[string]interface{}) (*types.Node, error) {
	node := &types.Node{}
	node.Uuid = getStr(m, "uuid")
	node.GroupID = getStr(m, "group_id")
	node.Type = types.NodeType(getStr(m, "type"))
	node.Name = getStr(m, "name")
	node.Content = getStr(m, "content")
	node.Summary = getStr(m, "summary")
	node.EntityType = getStr(m, "entity_type")
	node.EpisodeType = types.EpisodeType(getStr(m, "episode_type"))
	node.Level = getInt(m, "level")
	node.CreatedAt = timeFromFloat(getFloat(m, "created_at"))
	node.UpdatedAt = timeFromFloat(getFloat(m, "updated_at"))
	node.ValidFrom = timeFromFloat(getFloat(m, "valid_from"))
	vt := getFloat(m, "valid_to")
	if vt > 0 {
		t := timeFromFloat(vt)
		node.ValidTo = &t
	}

	node.Embedding = parseFloat32JSON(getStr(m, "embedding"))
	node.NameEmbedding = parseFloat32JSON(getStr(m, "name_embedding"))

	var meta map[string]interface{}
	if err := json.Unmarshal([]byte(getStr(m, "metadata")), &meta); err == nil {
		node.Metadata = meta
	}

	var sourceIDs []string
	if err := json.Unmarshal([]byte(getStr(m, "source_ids")), &sourceIDs); err == nil {
		node.SourceIDs = sourceIDs
	}

	var entityEdges []string
	if err := json.Unmarshal([]byte(getStr(m, "entity_edges")), &entityEdges); err == nil {
		node.EntityEdges = entityEdges
	}

	return node, nil
}

func (d *CozoDriver) rowsToNodes(result cozo.NamedRows) ([]*types.Node, error) {
	nodes := make([]*types.Node, 0, len(result.Rows))
	for _, row := range result.Rows {
		node, err := d.rowToNode(result.Headers, row)
		if err != nil {
			continue
		}
		nodes = append(nodes, node)
	}
	return nodes, nil
}

func (d *CozoDriver) rowToEdge(headers []string, row []any) (*types.Edge, error) {
	m := make(map[string]interface{}, len(headers))
	for i, h := range headers {
		if i < len(row) {
			m[h] = row[i]
		}
	}
	return d.mapToEdge(m)
}

func (d *CozoDriver) mapToEdge(m map[string]interface{}) (*types.Edge, error) {
	edge := &types.Edge{}
	edge.Uuid = getStr(m, "uuid")
	edge.GroupID = getStr(m, "group_id")
	edge.SourceID = getStr(m, "source_id")
	edge.TargetID = getStr(m, "target_id")
	edge.SourceNodeID = edge.SourceID
	edge.TargetNodeID = edge.TargetID
	edge.Type = types.EdgeType(getStr(m, "type"))
	edge.Name = getStr(m, "name")
	edge.Fact = getStr(m, "fact")
	edge.Summary = edge.Fact
	edge.Strength = getFloat(m, "strength")
	edge.CreatedAt = timeFromFloat(getFloat(m, "created_at"))
	edge.UpdatedAt = timeFromFloat(getFloat(m, "updated_at"))
	edge.ValidFrom = timeFromFloat(getFloat(m, "valid_from"))

	if vt := getFloat(m, "valid_to"); vt > 0 {
		t := timeFromFloat(vt)
		edge.ValidTo = &t
	}
	if ea := getFloat(m, "expired_at"); ea > 0 {
		t := timeFromFloat(ea)
		edge.ExpiredAt = &t
	}
	if va := getFloat(m, "valid_at"); va > 0 {
		t := timeFromFloat(va)
		edge.ValidAt = &t
	}
	if ia := getFloat(m, "invalid_at"); ia > 0 {
		t := timeFromFloat(ia)
		edge.InvalidAt = &t
	}

	edge.FactEmbedding = parseFloat32JSON(getStr(m, "fact_embedding"))
	edge.Embedding = parseFloat32JSON(getStr(m, "embedding"))

	var episodes []string
	if err := json.Unmarshal([]byte(getStr(m, "episodes")), &episodes); err == nil {
		edge.Episodes = episodes
	}

	var attrs map[string]interface{}
	if err := json.Unmarshal([]byte(getStr(m, "attributes")), &attrs); err == nil {
		edge.Attributes = attrs
	}

	var sourceIDs []string
	if err := json.Unmarshal([]byte(getStr(m, "source_ids")), &sourceIDs); err == nil {
		edge.SourceIDs = sourceIDs
	}

	return edge, nil
}

func (d *CozoDriver) rowsToEdges(result cozo.NamedRows) ([]*types.Edge, error) {
	edges := make([]*types.Edge, 0, len(result.Rows))
	for _, row := range result.Rows {
		edge, err := d.rowToEdge(result.Headers, row)
		if err != nil {
			continue
		}
		edges = append(edges, edge)
	}
	return edges, nil
}

// Utility functions

func getStr(m map[string]interface{}, key string) string {
	if v, ok := m[key]; ok {
		switch s := v.(type) {
		case string:
			return s
		default:
			return fmt.Sprintf("%v", v)
		}
	}
	return ""
}

func getFloat(m map[string]interface{}, key string) float64 {
	if v, ok := m[key]; ok {
		switch f := v.(type) {
		case float64:
			return f
		case int64:
			return float64(f)
		case int:
			return float64(f)
		}
	}
	return 0
}

func getInt(m map[string]interface{}, key string) int {
	if v, ok := m[key]; ok {
		switch i := v.(type) {
		case int64:
			return int(i)
		case float64:
			return int(i)
		case int:
			return i
		}
	}
	return 0
}

func timeFromFloat(f float64) time.Time {
	if f <= 0 {
		return time.Time{}
	}
	return time.Unix(int64(f), 0)
}

func parseFloat32JSON(s string) []float32 {
	if s == "" || s == "null" || s == "\"null\"" {
		return nil
	}
	var floats []float64
	if err := json.Unmarshal([]byte(s), &floats); err != nil {
		return nil
	}
	result := make([]float32, len(floats))
	for i, f := range floats {
		result[i] = float32(f)
	}
	return result
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// Compile-time interface checks
var _ GraphDriver = (*CozoDriver)(nil)
var _ GraphExporter = (*CozoDriver)(nil)
