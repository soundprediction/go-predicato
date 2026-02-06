package utils

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/parquet-go/parquet-go"
	"github.com/soundprediction/predicato/pkg/types"
)

// ParquetGraphWriter handles writing nodes and edges to Parquet files
type ParquetGraphWriter struct {
	baseDir string
}

// NewParquetGraphWriter creates a new Parquet writer
// baseDir should be the directory where parquet files will be stored
func NewParquetGraphWriter(baseDir string) (*ParquetGraphWriter, error) {
	// Ensure directories exist
	dirs := []string{"episodes", "entity_nodes", "entity_edges", "episodic_edges"}
	for _, d := range dirs {
		if err := os.MkdirAll(filepath.Join(baseDir, d), 0755); err != nil {
			return nil, fmt.Errorf("failed to create directory %s: %w", d, err)
		}
	}

	return &ParquetGraphWriter{baseDir: baseDir}, nil
}

// ParquetEpisode represents the schema for an episode in Parquet
type ParquetEpisode struct {
	Reference *time.Time `parquet:"reference"`
	CreatedAt *time.Time `parquet:"created_at"`
	UpdatedAt *time.Time `parquet:"updated_at"`
	ValidFrom *time.Time `parquet:"valid_from"`
	ID        string     `parquet:"id"`
	Name      string     `parquet:"name"`
	Content   string     `parquet:"content"`
	GroupID   string     `parquet:"group_id"`
	Metadata  string     `parquet:"metadata"` // JSON string
	Embedding []float32  `parquet:"embedding"`
}

// ParquetEntityNode represents the schema for an entity node in Parquet
type ParquetEntityNode struct {
	CreatedAt     *time.Time `parquet:"created_at"`
	UpdatedAt     *time.Time `parquet:"updated_at"`
	ValidFrom     *time.Time `parquet:"valid_from"`
	ValidTo       *time.Time `parquet:"valid_to"`
	ID            string     `parquet:"id"`
	Name          string     `parquet:"name"`
	EntityType    string     `parquet:"entity_type"`
	GroupID       string     `parquet:"group_id"`
	Summary       string     `parquet:"summary"`
	Metadata      string     `parquet:"metadata"` // JSON string
	EpisodeID     string     `parquet:"episode_id"`
	Embedding     []float32  `parquet:"embedding"`
	NameEmbedding []float32  `parquet:"name_embedding"`
}

// ParquetEntityEdge represents the schema for an entity edge in Parquet
type ParquetEntityEdge struct {
	CreatedAt     *time.Time `parquet:"created_at"`
	ValidFrom     *time.Time `parquet:"valid_from"`
	InvalidAt     *time.Time `parquet:"invalid_at"`
	ExpiredAt     *time.Time `parquet:"expired_at"`
	ID            string     `parquet:"id"`
	SourceID      string     `parquet:"source_id"`
	TargetID      string     `parquet:"target_id"`
	Name          string     `parquet:"name"`
	Fact          string     `parquet:"fact"`
	Summary       string     `parquet:"summary"`
	EdgeType      string     `parquet:"edge_type"`
	GroupID       string     `parquet:"group_id"`
	Episodes      string     `parquet:"episodes"` // JSON string
	Metadata      string     `parquet:"metadata"` // JSON string
	EpisodeID     string     `parquet:"episode_id"`
	Embedding     []float32  `parquet:"embedding"`
	FactEmbedding []float32  `parquet:"fact_embedding"`
}

// ParquetEpisodicEdge represents the schema for an episodic edge in Parquet
type ParquetEpisodicEdge struct {
	ID        string     `parquet:"id"`
	SourceID  string     `parquet:"source_id"`
	TargetID  string     `parquet:"target_id"`
	Name      string     `parquet:"name"`
	EdgeType  string     `parquet:"edge_type"`
	GroupID   string     `parquet:"group_id"`
	CreatedAt *time.Time `parquet:"created_at"`
	ValidFrom *time.Time `parquet:"valid_from"`
	EpisodeID string     `parquet:"episode_id"`
}

// WriteEpisode writes an episode node to Parquet
func (w *ParquetGraphWriter) WriteEpisode(ctx context.Context, episode *types.Node) error {
	metadataJSON, err := json.Marshal(episode.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	groupID := episode.GroupID
	if groupID == "" {
		groupID = "default"
	}

	pe := ParquetEpisode{
		ID:        episode.Uuid,
		Name:      episode.Name,
		Content:   episode.Content,
		GroupID:   groupID,
		Embedding: episode.Embedding,
		Metadata:  string(metadataJSON),
	}

	if !episode.Reference.IsZero() {
		pe.Reference = &episode.Reference
	}
	if !episode.CreatedAt.IsZero() {
		pe.CreatedAt = &episode.CreatedAt
	}
	if !episode.UpdatedAt.IsZero() {
		pe.UpdatedAt = &episode.UpdatedAt
	}
	if !episode.ValidFrom.IsZero() {
		pe.ValidFrom = &episode.ValidFrom
	}

	filename := fmt.Sprintf("episode_%s.parquet", episode.Uuid)
	path := filepath.Join(w.baseDir, "episodes", filename)

	return parquet.WriteFile(path, []ParquetEpisode{pe})
}

// WriteEntityNodes writes entity nodes to Parquet
func (w *ParquetGraphWriter) WriteEntityNodes(ctx context.Context, nodes []*types.Node, episodeID string) error {
	if len(nodes) == 0 {
		return nil
	}

	parquetNodes := make([]ParquetEntityNode, 0, len(nodes))
	for _, node := range nodes {
		metadataJSON, err := json.Marshal(node.Metadata)
		if err != nil {
			return fmt.Errorf("failed to marshal metadata: %w", err)
		}

		pn := ParquetEntityNode{
			ID:            node.Uuid,
			Name:          node.Name,
			EntityType:    node.EntityType,
			GroupID:       node.GroupID,
			Summary:       node.Summary,
			Embedding:     node.Embedding,
			NameEmbedding: node.NameEmbedding,
			Metadata:      string(metadataJSON),
			EpisodeID:     episodeID,
		}

		if !node.CreatedAt.IsZero() {
			pn.CreatedAt = &node.CreatedAt
		}
		if !node.UpdatedAt.IsZero() {
			pn.UpdatedAt = &node.UpdatedAt
		}
		if !node.ValidFrom.IsZero() {
			pn.ValidFrom = &node.ValidFrom
		}
		if node.ValidTo != nil && !node.ValidTo.IsZero() {
			pn.ValidTo = node.ValidTo
		}

		parquetNodes = append(parquetNodes, pn)
	}

	// Write batch to a single file
	filename := fmt.Sprintf("entity_nodes_%s_%d.parquet", episodeID, time.Now().UnixNano())
	path := filepath.Join(w.baseDir, "entity_nodes", filename)

	return parquet.WriteFile(path, parquetNodes)
}

// WriteEntityEdges writes entity edges to Parquet
func (w *ParquetGraphWriter) WriteEntityEdges(ctx context.Context, edges []*types.Edge, episodeID string) error {
	if len(edges) == 0 {
		return nil
	}

	parquetEdges := make([]ParquetEntityEdge, 0, len(edges))
	for _, edge := range edges {
		metadataJSON, err := json.Marshal(edge.BaseEdge.Metadata)
		if err != nil {
			return fmt.Errorf("failed to marshal metadata: %w", err)
		}

		episodesJSON, err := json.Marshal(edge.Episodes)
		if err != nil {
			return fmt.Errorf("failed to marshal episodes: %w", err)
		}

		pe := ParquetEntityEdge{
			ID:            edge.Uuid,
			SourceID:      edge.SourceID,
			TargetID:      edge.TargetID,
			Name:          edge.Name,
			Fact:          edge.Fact,
			Summary:       edge.Summary,
			EdgeType:      string(edge.Type),
			GroupID:       edge.GroupID,
			Embedding:     edge.Embedding,
			FactEmbedding: edge.FactEmbedding,
			Episodes:      string(episodesJSON),
			Metadata:      string(metadataJSON),
			EpisodeID:     episodeID,
		}

		if !edge.CreatedAt.IsZero() {
			pe.CreatedAt = &edge.CreatedAt
		}
		if !edge.ValidFrom.IsZero() {
			pe.ValidFrom = &edge.ValidFrom
		}
		if edge.InvalidAt != nil && !edge.InvalidAt.IsZero() {
			pe.InvalidAt = edge.InvalidAt
		}
		if edge.ExpiredAt != nil && !edge.ExpiredAt.IsZero() {
			pe.ExpiredAt = edge.ExpiredAt
		}

		parquetEdges = append(parquetEdges, pe)
	}

	filename := fmt.Sprintf("entity_edges_%s_%d.parquet", episodeID, time.Now().UnixNano())
	path := filepath.Join(w.baseDir, "entity_edges", filename)

	return parquet.WriteFile(path, parquetEdges)
}

// WriteEpisodicEdges writes episodic edges to Parquet
func (w *ParquetGraphWriter) WriteEpisodicEdges(ctx context.Context, edges []*types.Edge, episodeID string) error {
	if len(edges) == 0 {
		return nil
	}

	parquetEdges := make([]ParquetEpisodicEdge, 0, len(edges))
	for _, edge := range edges {
		pe := ParquetEpisodicEdge{
			ID:        edge.Uuid,
			SourceID:  edge.SourceID,
			TargetID:  edge.TargetID,
			Name:      edge.Name,
			EdgeType:  string(edge.Type),
			GroupID:   edge.GroupID,
			EpisodeID: episodeID,
		}

		if !edge.CreatedAt.IsZero() {
			pe.CreatedAt = &edge.CreatedAt
		}
		if !edge.ValidFrom.IsZero() {
			pe.ValidFrom = &edge.ValidFrom
		}

		parquetEdges = append(parquetEdges, pe)
	}

	filename := fmt.Sprintf("episodic_edges_%s_%d.parquet", episodeID, time.Now().UnixNano())
	path := filepath.Join(w.baseDir, "episodic_edges", filename)

	return parquet.WriteFile(path, parquetEdges)
}

// Close implements a closer interface, currently no-op as we write file-per-call
func (w *ParquetGraphWriter) Close() error {
	return nil
}
