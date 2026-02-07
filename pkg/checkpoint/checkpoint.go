package checkpoint

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/soundprediction/predicato/pkg/types"
	"github.com/soundprediction/predicato/pkg/utils"
)

// ErrInvalidEpisodeID is returned when an episode ID contains invalid characters
var ErrInvalidEpisodeID = errors.New("invalid episode ID: contains path traversal or invalid characters")

// ProcessingStep represents a step in the addEpisodeChunked pipeline
type ProcessingStep string

const (
	StepInitial              ProcessingStep = "initial"
	StepPrepared             ProcessingStep = "prepared"
	StepGotPreviousEpisodes  ProcessingStep = "got_previous_episodes"
	StepCreatedChunks        ProcessingStep = "created_chunks"
	StepExtractedEntities    ProcessingStep = "extracted_entities"
	StepDeduplicatedEntities ProcessingStep = "deduplicated_entities"
	StepExtractedEdges       ProcessingStep = "extracted_edges"
	StepResolvedEdges        ProcessingStep = "resolved_edges"
	StepExtractedAttributes  ProcessingStep = "extracted_attributes"
	StepBuiltEpisodicEdges   ProcessingStep = "built_episodic_edges"
	StepPerformedGraphUpdate ProcessingStep = "performed_graph_update"
	StepUpdatedCommunities   ProcessingStep = "updated_communities"
	StepCompleted            ProcessingStep = "completed"
)

// EpisodeCheckpoint represents the state of a partially processed episode
type EpisodeCheckpoint struct {

	// Timestamp tracking
	CreatedAt       time.Time          `json:"created_at"`
	LastUpdatedAt   time.Time          `json:"last_updated_at"`
	Options         *AddEpisodeOptions `json:"options,omitempty"`
	MainEpisodeNode *types.Node        `json:"main_episode_node,omitempty"`
	// Episode identification
	EpisodeID string         `json:"episode_id"`
	GroupID   string         `json:"group_id"`
	Step      ProcessingStep `json:"step"`

	LastError      string `json:"last_error,omitempty"`
	LastErrorStack string `json:"last_error_stack,omitempty"`

	// Original episode data
	Episode types.Episode `json:"episode"`

	// STEP 1-2: Preparation data
	Chunks           []string      `json:"chunks,omitempty"`
	PreviousEpisodes []*types.Node `json:"previous_episodes,omitempty"`

	// STEP 3: Chunk structures
	ChunkEpisodeNodes []*types.Node        `json:"chunk_episode_nodes,omitempty"`
	EpisodeTuples     []utils.EpisodeTuple `json:"episode_tuples,omitempty"`

	// STEP 5: Extracted entities
	ExtractedNodesByChunk [][]*types.Node `json:"extracted_nodes_by_chunk,omitempty"`

	// STEP 6: Deduplicated entities
	DedupeChunkIndices []int         `json:"dedupe_chunk_indices,omitempty"`
	AllResolvedNodes   []*types.Node `json:"all_resolved_nodes,omitempty"`

	// STEP 7: Extracted edges
	AllExtractedEdges []*types.Edge `json:"all_extracted_edges,omitempty"`

	// STEP 8: Resolved edges
	ResolvedEdges    []*types.Edge `json:"resolved_edges,omitempty"`
	InvalidatedEdges []*types.Edge `json:"invalidated_edges,omitempty"`

	// STEP 9: Hydrated nodes with attributes
	HydratedNodes []*types.Node `json:"hydrated_nodes,omitempty"`

	// STEP 10: Episodic edges
	EpisodicEdges []*types.Edge `json:"episodic_edges,omitempty"`

	// STEP 12: Communities
	Communities    []*types.Node `json:"communities,omitempty"`
	CommunityEdges []*types.Edge `json:"community_edges,omitempty"`
	AttemptCount   int           `json:"attempt_count"`
	MaxCharacters  int           `json:"max_characters"`
}

// AddEpisodeOptions is a copy of the predicato.AddEpisodeOptions for checkpoint serialization
type AddEpisodeOptions struct {
	EntityTypes          map[string]interface{}              `json:"entity_types,omitempty"`
	EdgeTypes            map[string]interface{}              `json:"edge_types,omitempty"`
	EdgeTypeMap          map[string]map[string][]interface{} `json:"edge_type_map,omitempty"`
	ExcludedEntityTypes  []string                            `json:"excluded_entity_types,omitempty"`
	PreviousEpisodeUUIDs []string                            `json:"previous_episode_uuids,omitempty"`
	MaxCharacters        int                                 `json:"max_characters"`
	OverwriteExisting    bool                                `json:"overwrite_existing"`
	DeferGraphIngestion  bool                                `json:"defer_graph_ingestion"`
}

// CheckpointManager manages episode checkpoints
type CheckpointManager struct {
	checkpointDir string
}

// NewCheckpointManager creates a new checkpoint manager
// If checkpointDir is empty, uses os.TempDir()/predicato-checkpoints
func NewCheckpointManager(checkpointDir string) (*CheckpointManager, error) {
	if checkpointDir == "" {
		checkpointDir = filepath.Join(os.TempDir(), "predicato-checkpoints")
	}

	// Create checkpoint directory if it doesn't exist
	if err := os.MkdirAll(checkpointDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create checkpoint directory: %w", err)
	}

	return &CheckpointManager{
		checkpointDir: checkpointDir,
	}, nil
}

// validateEpisodeID checks that the episode ID is safe for use in file paths.
// It rejects IDs containing path separators, path traversal sequences, or null bytes.
func validateEpisodeID(episodeID string) error {
	if episodeID == "" {
		return ErrInvalidEpisodeID
	}

	// Check for path traversal sequences
	if strings.Contains(episodeID, "..") {
		return ErrInvalidEpisodeID
	}

	// Check for path separators
	if strings.ContainsAny(episodeID, `/\`) {
		return ErrInvalidEpisodeID
	}

	// Check for null bytes (can truncate paths in some systems)
	if strings.ContainsRune(episodeID, '\x00') {
		return ErrInvalidEpisodeID
	}

	return nil
}

// isPathWithinDirectory checks that the resolved path is within the expected directory.
// This provides defense-in-depth against path traversal attacks.
func isPathWithinDirectory(path, directory string) bool {
	// Clean both paths to resolve any . or .. components
	cleanPath := filepath.Clean(path)
	cleanDir := filepath.Clean(directory)

	// Ensure the directory path ends with separator for proper prefix matching
	if !strings.HasSuffix(cleanDir, string(filepath.Separator)) {
		cleanDir += string(filepath.Separator)
	}

	// Check if the path starts with the directory
	return strings.HasPrefix(cleanPath, cleanDir) || cleanPath == filepath.Clean(directory)
}

// GetCheckpointPath returns the file path for an episode's checkpoint.
// Returns an error if the episode ID contains invalid characters or path traversal sequences.
func (m *CheckpointManager) GetCheckpointPath(episodeID string) (string, error) {
	if err := validateEpisodeID(episodeID); err != nil {
		return "", err
	}

	filename := fmt.Sprintf("checkpoint_%s.json", episodeID)
	fullPath := filepath.Join(m.checkpointDir, filename)

	// Defense-in-depth: verify the resolved path is within the checkpoint directory
	if !isPathWithinDirectory(fullPath, m.checkpointDir) {
		return "", ErrInvalidEpisodeID
	}

	return fullPath, nil
}

// Save persists the checkpoint to disk
func (m *CheckpointManager) Save(ctx context.Context, checkpoint *EpisodeCheckpoint) error {
	checkpoint.LastUpdatedAt = time.Now()

	data, err := json.MarshalIndent(checkpoint, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal checkpoint: %w", err)
	}

	checkpointPath, err := m.GetCheckpointPath(checkpoint.EpisodeID)
	if err != nil {
		return fmt.Errorf("invalid episode ID: %w", err)
	}

	// Write to a temporary file first, then rename for atomic write
	tmpPath := checkpointPath + ".tmp"
	if err := os.WriteFile(tmpPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write checkpoint file: %w", err)
	}

	if err := os.Rename(tmpPath, checkpointPath); err != nil {
		return fmt.Errorf("failed to rename checkpoint file: %w", err)
	}

	return nil
}

// Load retrieves a checkpoint from disk
func (m *CheckpointManager) Load(ctx context.Context, episodeID string) (*EpisodeCheckpoint, error) {
	checkpointPath, err := m.GetCheckpointPath(episodeID)
	if err != nil {
		return nil, fmt.Errorf("invalid episode ID: %w", err)
	}

	data, err := os.ReadFile(checkpointPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil // No checkpoint exists
		}
		return nil, fmt.Errorf("failed to read checkpoint file: %w", err)
	}

	var checkpoint EpisodeCheckpoint
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return nil, fmt.Errorf("failed to unmarshal checkpoint: %w", err)
	}

	return &checkpoint, nil
}

// Delete removes a checkpoint from disk
func (m *CheckpointManager) Delete(ctx context.Context, episodeID string) error {
	checkpointPath, err := m.GetCheckpointPath(episodeID)
	if err != nil {
		return fmt.Errorf("invalid episode ID: %w", err)
	}

	if err := os.Remove(checkpointPath); err != nil {
		if os.IsNotExist(err) {
			return nil // Already deleted
		}
		return fmt.Errorf("failed to delete checkpoint file: %w", err)
	}

	return nil
}

// Exists checks if a checkpoint exists for an episode
func (m *CheckpointManager) Exists(ctx context.Context, episodeID string) (bool, error) {
	checkpointPath, err := m.GetCheckpointPath(episodeID)
	if err != nil {
		return false, fmt.Errorf("invalid episode ID: %w", err)
	}

	_, err = os.Stat(checkpointPath)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil
		}
		return false, fmt.Errorf("failed to check checkpoint existence: %w", err)
	}

	return true, nil
}

// List returns all checkpoint files in the checkpoint directory
func (m *CheckpointManager) List(ctx context.Context) ([]*EpisodeCheckpoint, error) {
	entries, err := os.ReadDir(m.checkpointDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read checkpoint directory: %w", err)
	}

	var checkpoints []*EpisodeCheckpoint
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		// Only process .json files, skip .tmp files
		if filepath.Ext(entry.Name()) != ".json" {
			continue
		}

		data, err := os.ReadFile(filepath.Join(m.checkpointDir, entry.Name()))
		if err != nil {
			continue // Skip files we can't read
		}

		var checkpoint EpisodeCheckpoint
		if err := json.Unmarshal(data, &checkpoint); err != nil {
			continue // Skip files we can't unmarshal
		}

		checkpoints = append(checkpoints, &checkpoint)
	}

	return checkpoints, nil
}

// UpdateStep updates the checkpoint's current step
func (m *CheckpointManager) UpdateStep(ctx context.Context, episodeID string, step ProcessingStep) error {
	checkpoint, err := m.Load(ctx, episodeID)
	if err != nil {
		return err
	}
	if checkpoint == nil {
		return fmt.Errorf("checkpoint not found for episode %s", episodeID)
	}

	checkpoint.Step = step
	return m.Save(ctx, checkpoint)
}

// RecordError records an error in the checkpoint
func (m *CheckpointManager) RecordError(ctx context.Context, episodeID string, err error, stackTrace string) error {
	checkpoint, loadErr := m.Load(ctx, episodeID)
	if loadErr != nil {
		return loadErr
	}
	if checkpoint == nil {
		return fmt.Errorf("checkpoint not found for episode %s", episodeID)
	}

	checkpoint.AttemptCount++
	checkpoint.LastError = err.Error()
	checkpoint.LastErrorStack = stackTrace

	return m.Save(ctx, checkpoint)
}

// GetCheckpointDir returns the checkpoint directory path
func (m *CheckpointManager) GetCheckpointDir() string {
	return m.checkpointDir
}

// CleanOld removes checkpoints older than the specified duration
func (m *CheckpointManager) CleanOld(ctx context.Context, maxAge time.Duration) (int, error) {
	checkpoints, err := m.List(ctx)
	if err != nil {
		return 0, err
	}

	cutoff := time.Now().Add(-maxAge)
	removed := 0

	for _, checkpoint := range checkpoints {
		if checkpoint.LastUpdatedAt.Before(cutoff) {
			if err := m.Delete(ctx, checkpoint.EpisodeID); err != nil {
				// Log but don't fail the entire cleanup
				continue
			}
			removed++
		}
	}

	return removed, nil
}
