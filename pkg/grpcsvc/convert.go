package grpcsvc

import (
	"time"

	"github.com/soundprediction/predicato/pkg/grpcsvc/pb"
	"github.com/soundprediction/predicato/pkg/types"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// --- timestamp helpers ---

func tsToProto(t time.Time) *timestamppb.Timestamp {
	if t.IsZero() {
		return nil
	}
	return timestamppb.New(t)
}

func tsPtrToProto(t *time.Time) *timestamppb.Timestamp {
	if t == nil || t.IsZero() {
		return nil
	}
	return timestamppb.New(*t)
}

func tsFromProto(ts *timestamppb.Timestamp) time.Time {
	if ts == nil {
		return time.Time{}
	}
	return ts.AsTime()
}

func tsPtrFromProto(ts *timestamppb.Timestamp) *time.Time {
	if ts == nil {
		return nil
	}
	t := ts.AsTime()
	return &t
}

// --- metadata (map[string]interface{}) <-> structpb.Struct ---

func structToProto(m map[string]interface{}) *structpb.Struct {
	if len(m) == 0 {
		return nil
	}
	s, err := structpb.NewStruct(m)
	if err != nil {
		return nil // non-serializable metadata is dropped rather than failing the RPC
	}
	return s
}

func structFromProto(s *structpb.Struct) map[string]interface{} {
	if s == nil {
		return nil
	}
	return s.AsMap()
}

// --- Node ---

func nodeToProto(n *types.Node) *pb.Node {
	if n == nil {
		return nil
	}
	return &pb.Node{
		Uuid:        n.Uuid,
		Name:        n.Name,
		Type:        string(n.Type),
		GroupId:     n.GroupID,
		EntityType:  n.EntityType,
		Summary:     n.Summary,
		EpisodeType: string(n.EpisodeType),
		Content:     n.Content,
		EntityEdges: n.EntityEdges,
		SourceIds:   n.SourceIDs,
		Level:       int32(n.Level),
		CreatedAt:   tsToProto(n.CreatedAt),
		UpdatedAt:   tsToProto(n.UpdatedAt),
		ValidFrom:   tsToProto(n.ValidFrom),
		ValidTo:     tsPtrToProto(n.ValidTo),
		Reference:   tsToProto(n.Reference),
		Metadata:    structToProto(n.Metadata),
	}
}

func nodeFromProto(p *pb.Node) *types.Node {
	if p == nil {
		return nil
	}
	return &types.Node{
		Uuid:        p.Uuid,
		Name:        p.Name,
		Type:        types.NodeType(p.Type),
		GroupID:     p.GroupId,
		EntityType:  p.EntityType,
		Summary:     p.Summary,
		EpisodeType: types.EpisodeType(p.EpisodeType),
		Content:     p.Content,
		EntityEdges: p.EntityEdges,
		SourceIDs:   p.SourceIds,
		Level:       int(p.Level),
		CreatedAt:   tsFromProto(p.CreatedAt),
		UpdatedAt:   tsFromProto(p.UpdatedAt),
		ValidFrom:   tsFromProto(p.ValidFrom),
		ValidTo:     tsPtrFromProto(p.ValidTo),
		Reference:   tsFromProto(p.Reference),
		Metadata:    structFromProto(p.Metadata),
	}
}

func nodesToProto(ns []*types.Node) []*pb.Node {
	if len(ns) == 0 {
		return nil
	}
	out := make([]*pb.Node, 0, len(ns))
	for _, n := range ns {
		out = append(out, nodeToProto(n))
	}
	return out
}

func nodesFromProto(ps []*pb.Node) []*types.Node {
	if len(ps) == 0 {
		return nil
	}
	out := make([]*types.Node, 0, len(ps))
	for _, p := range ps {
		out = append(out, nodeFromProto(p))
	}
	return out
}

// --- Edge (types.EntityEdge) ---

func edgeToProto(e *types.Edge) *pb.Edge {
	if e == nil {
		return nil
	}
	return &pb.Edge{
		Uuid:         e.Uuid,
		GroupId:      e.GroupID,
		SourceNodeId: e.SourceNodeID,
		TargetNodeId: e.TargetNodeID,
		Name:         e.Name,
		Fact:         e.Fact,
		Type:         string(e.Type),
		SourceId:     e.SourceID,
		TargetId:     e.TargetID,
		Summary:      e.Summary,
		Episodes:     e.Episodes,
		SourceIds:    e.SourceIDs,
		Strength:     e.Strength,
		CreatedAt:    tsToProto(e.CreatedAt),
		UpdatedAt:    tsToProto(e.UpdatedAt),
		ValidFrom:    tsToProto(e.ValidFrom),
		ValidTo:      tsPtrToProto(e.ValidTo),
		ExpiredAt:    tsPtrToProto(e.ExpiredAt),
		ValidAt:      tsPtrToProto(e.ValidAt),
		InvalidAt:    tsPtrToProto(e.InvalidAt),
		Metadata:     structToProto(e.Metadata),
		Attributes:   structToProto(e.Attributes),
	}
}

func edgeFromProto(p *pb.Edge) *types.Edge {
	if p == nil {
		return nil
	}
	e := &types.Edge{
		UpdatedAt:  tsFromProto(p.UpdatedAt),
		ValidFrom:  tsFromProto(p.ValidFrom),
		ExpiredAt:  tsPtrFromProto(p.ExpiredAt),
		ValidAt:    tsPtrFromProto(p.ValidAt),
		InvalidAt:  tsPtrFromProto(p.InvalidAt),
		ValidTo:    tsPtrFromProto(p.ValidTo),
		Attributes: structFromProto(p.Attributes),
		Name:       p.Name,
		Fact:       p.Fact,
		Type:       types.EdgeType(p.Type),
		SourceID:   p.SourceId,
		TargetID:   p.TargetId,
		Summary:    p.Summary,
		Episodes:   p.Episodes,
		SourceIDs:  p.SourceIds,
		Strength:   p.Strength,
	}
	e.Uuid = p.Uuid
	e.GroupID = p.GroupId
	e.SourceNodeID = p.SourceNodeId
	e.TargetNodeID = p.TargetNodeId
	e.CreatedAt = tsFromProto(p.CreatedAt)
	e.Metadata = structFromProto(p.Metadata)
	return e
}

func edgesToProto(es []*types.Edge) []*pb.Edge {
	if len(es) == 0 {
		return nil
	}
	out := make([]*pb.Edge, 0, len(es))
	for _, e := range es {
		out = append(out, edgeToProto(e))
	}
	return out
}

func edgesFromProto(ps []*pb.Edge) []*types.Edge {
	if len(ps) == 0 {
		return nil
	}
	out := make([]*types.Edge, 0, len(ps))
	for _, p := range ps {
		out = append(out, edgeFromProto(p))
	}
	return out
}

// --- SearchConfig ---

func searchConfigToProto(c *types.SearchConfig) *pb.SearchConfig {
	if c == nil {
		return nil
	}
	out := &pb.SearchConfig{
		ExcludeEntityTypes:  c.ExcludeEntityTypes,
		PreferredPredicates: c.PreferredPredicates,
		Limit:               int32(c.Limit),
		CenterNodeDistance:  int32(c.CenterNodeDistance),
		MinScore:            c.MinScore,
		IncludeEdges:        c.IncludeEdges,
	}
	if c.NodeConfig != nil {
		out.NodeConfig = &pb.NodeSearchConfig{Reranker: c.NodeConfig.Reranker, SearchMethods: c.NodeConfig.SearchMethods, MinScore: c.NodeConfig.MinScore}
	}
	if c.EdgeConfig != nil {
		out.EdgeConfig = &pb.EdgeSearchConfig{Reranker: c.EdgeConfig.Reranker, SearchMethods: c.EdgeConfig.SearchMethods, MinScore: c.EdgeConfig.MinScore}
	}
	if c.Filters != nil {
		f := &pb.SearchFilters{
			GroupIds:    c.Filters.GroupIDs,
			EntityTypes: c.Filters.EntityTypes,
		}
		for _, nt := range c.Filters.NodeTypes {
			f.NodeTypes = append(f.NodeTypes, string(nt))
		}
		for _, et := range c.Filters.EdgeTypes {
			f.EdgeTypes = append(f.EdgeTypes, string(et))
		}
		if c.Filters.TimeRange != nil {
			f.TimeRange = &pb.TimeRange{Start: tsToProto(c.Filters.TimeRange.Start), End: tsToProto(c.Filters.TimeRange.End)}
		}
		out.Filters = f
	}
	return out
}

func searchConfigFromProto(p *pb.SearchConfig) *types.SearchConfig {
	if p == nil {
		return nil
	}
	out := &types.SearchConfig{
		ExcludeEntityTypes:  p.ExcludeEntityTypes,
		PreferredPredicates: p.PreferredPredicates,
		Limit:               int(p.Limit),
		CenterNodeDistance:  int(p.CenterNodeDistance),
		MinScore:            p.MinScore,
		IncludeEdges:        p.IncludeEdges,
	}
	if p.NodeConfig != nil {
		out.NodeConfig = &types.NodeSearchConfig{Reranker: p.NodeConfig.Reranker, SearchMethods: p.NodeConfig.SearchMethods, MinScore: p.NodeConfig.MinScore}
	}
	if p.EdgeConfig != nil {
		out.EdgeConfig = &types.EdgeSearchConfig{Reranker: p.EdgeConfig.Reranker, SearchMethods: p.EdgeConfig.SearchMethods, MinScore: p.EdgeConfig.MinScore}
	}
	if p.Filters != nil {
		f := &types.SearchFilters{GroupIDs: p.Filters.GroupIds, EntityTypes: p.Filters.EntityTypes}
		for _, nt := range p.Filters.NodeTypes {
			f.NodeTypes = append(f.NodeTypes, types.NodeType(nt))
		}
		for _, et := range p.Filters.EdgeTypes {
			f.EdgeTypes = append(f.EdgeTypes, types.EdgeType(et))
		}
		if p.Filters.TimeRange != nil {
			f.TimeRange = &types.TimeRange{Start: tsFromProto(p.Filters.TimeRange.Start), End: tsFromProto(p.Filters.TimeRange.End)}
		}
		out.Filters = f
	}
	return out
}

// --- SearchResults ---

func searchResultsToProto(r *types.SearchResults) *pb.SearchResults {
	if r == nil {
		return &pb.SearchResults{}
	}
	return &pb.SearchResults{Query: r.Query, Nodes: nodesToProto(r.Nodes), Edges: edgesToProto(r.Edges), Total: int32(r.Total)}
}

func searchResultsFromProto(p *pb.SearchResults) *types.SearchResults {
	if p == nil {
		return nil
	}
	return &types.SearchResults{Query: p.Query, Nodes: nodesFromProto(p.Nodes), Edges: edgesFromProto(p.Edges), Total: int(p.Total)}
}

// --- Episode + AddEpisodeResults ---

func episodeFromProto(p *pb.Episode) types.Episode {
	if p == nil {
		return types.Episode{}
	}
	return types.Episode{
		ID:        p.Id,
		Name:      p.Name,
		Content:   p.Content,
		Source:    p.Source,
		GroupID:   p.GroupId,
		Reference: tsFromProto(p.Reference),
		CreatedAt: tsFromProto(p.CreatedAt),
		Metadata:  structFromProto(p.Metadata),
	}
}

func episodeToProto(e types.Episode) *pb.Episode {
	return &pb.Episode{
		Id:        e.ID,
		Name:      e.Name,
		Content:   e.Content,
		Source:    e.Source,
		GroupId:   e.GroupID,
		Reference: tsToProto(e.Reference),
		CreatedAt: tsToProto(e.CreatedAt),
		Metadata:  structToProto(e.Metadata),
	}
}

func addResultsToProto(r *types.AddEpisodeResults) *pb.AddEpisodeResults {
	if r == nil {
		return &pb.AddEpisodeResults{}
	}
	return &pb.AddEpisodeResults{
		Episode:        nodeToProto(r.Episode),
		EpisodicEdges:  edgesToProto(r.EpisodicEdges),
		Nodes:          nodesToProto(r.Nodes),
		Edges:          edgesToProto(r.Edges),
		Communities:    nodesToProto(r.Communities),
		CommunityEdges: edgesToProto(r.CommunityEdges),
		Rules:          nodesToProto(r.Rules),
		RuleEdges:      edgesToProto(r.RuleEdges),
	}
}

func addResultsFromProto(p *pb.AddEpisodeResults) *types.AddEpisodeResults {
	if p == nil {
		return nil
	}
	return &types.AddEpisodeResults{
		Episode:        nodeFromProto(p.Episode),
		EpisodicEdges:  edgesFromProto(p.EpisodicEdges),
		Nodes:          nodesFromProto(p.Nodes),
		Edges:          edgesFromProto(p.Edges),
		Communities:    nodesFromProto(p.Communities),
		CommunityEdges: edgesFromProto(p.CommunityEdges),
		Rules:          nodesFromProto(p.Rules),
		RuleEdges:      edgesFromProto(p.RuleEdges),
	}
}
