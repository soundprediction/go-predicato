package checkpoint

import (
	"context"
	"fmt"
	"runtime/debug"
	"time"

	"github.com/soundprediction/predicato/pkg/types"
)

// NewCheckpoint creates a new checkpoint for an episode at the initial step
func NewCheckpoint(episode types.Episode, options *AddEpisodeOptions, maxCharacters int) *EpisodeCheckpoint {
	now := time.Now()
	return &EpisodeCheckpoint{
		EpisodeID:     episode.ID,
		GroupID:       episode.GroupID,
		Step:          StepInitial,
		CreatedAt:     now,
		LastUpdatedAt: now,
		AttemptCount:  0,
		Episode:       episode,
		Options:       options,
		MaxCharacters: maxCharacters,
	}
}

// CanRetry determines if a checkpoint should be retried based on attempt count and age
func (c *EpisodeCheckpoint) CanRetry(maxAttempts int, maxAge time.Duration) bool {
	if c.AttemptCount >= maxAttempts {
		return false
	}

	age := time.Since(c.CreatedAt)
	return age <= maxAge
}

// GetProgress returns a human-readable progress description
func (c *EpisodeCheckpoint) GetProgress() string {
	steps := []ProcessingStep{
		StepInitial,
		StepPrepared,
		StepGotPreviousEpisodes,
		StepCreatedChunks,
		StepExtractedEntities,
		StepDeduplicatedEntities,
		StepExtractedEdges,
		StepResolvedEdges,
		StepExtractedAttributes,
		StepBuiltEpisodicEdges,
		StepPerformedGraphUpdate,
		StepUpdatedCommunities,
		StepCompleted,
	}

	currentIdx := -1
	for i, step := range steps {
		if step == c.Step {
			currentIdx = i
			break
		}
	}

	if currentIdx == -1 {
		return "Unknown step"
	}

	percentage := (float64(currentIdx) / float64(len(steps)-1)) * 100
	return fmt.Sprintf("%.0f%% (%s)", percentage, c.Step)
}

// IsRecoverable determines if an error at the current step is likely recoverable
func (c *EpisodeCheckpoint) IsRecoverable() bool {
	// Steps that involve LLM calls are generally recoverable (transient failures)
	recoverableSteps := map[ProcessingStep]bool{
		StepExtractedEntities:    true,
		StepDeduplicatedEntities: true,
		StepExtractedEdges:       true,
		StepResolvedEdges:        true,
		StepExtractedAttributes:  true,
		StepUpdatedCommunities:   true,
	}

	return recoverableSteps[c.Step]
}

// SaveWithStep is a helper that updates the step and saves in one operation
func (m *CheckpointManager) SaveWithStep(ctx context.Context, checkpoint *EpisodeCheckpoint, step ProcessingStep) error {
	checkpoint.Step = step
	return m.Save(ctx, checkpoint)
}

// SaveWithError is a helper that records an error and saves in one operation
func (m *CheckpointManager) SaveWithError(ctx context.Context, checkpoint *EpisodeCheckpoint, err error) error {
	checkpoint.AttemptCount++
	checkpoint.LastError = err.Error()
	checkpoint.LastErrorStack = string(debug.Stack())
	return m.Save(ctx, checkpoint)
}

// LoadOrCreate loads an existing checkpoint or creates a new one
func (m *CheckpointManager) LoadOrCreate(ctx context.Context, episode types.Episode, options *AddEpisodeOptions, maxCharacters int) (*EpisodeCheckpoint, bool, error) {
	existing, err := m.Load(ctx, episode.ID)
	if err != nil {
		return nil, false, err
	}

	if existing != nil {
		return existing, true, nil
	}

	// Create new checkpoint
	checkpoint := NewCheckpoint(episode, options, maxCharacters)
	if err := m.Save(ctx, checkpoint); err != nil {
		return nil, false, err
	}

	return checkpoint, false, nil
}

// GetNextStep returns the next step in the pipeline after the current step
func GetNextStep(current ProcessingStep) (ProcessingStep, error) {
	steps := map[ProcessingStep]ProcessingStep{
		StepInitial:              StepPrepared,
		StepPrepared:             StepGotPreviousEpisodes,
		StepGotPreviousEpisodes:  StepCreatedChunks,
		StepCreatedChunks:        StepExtractedEntities,
		StepExtractedEntities:    StepDeduplicatedEntities,
		StepDeduplicatedEntities: StepExtractedEdges,
		StepExtractedEdges:       StepResolvedEdges,
		StepResolvedEdges:        StepExtractedAttributes,
		StepExtractedAttributes:  StepBuiltEpisodicEdges,
		StepBuiltEpisodicEdges:   StepPerformedGraphUpdate,
		StepPerformedGraphUpdate: StepUpdatedCommunities,
		StepUpdatedCommunities:   StepCompleted,
	}

	next, ok := steps[current]
	if !ok {
		return "", fmt.Errorf("unknown current step: %s", current)
	}

	return next, nil
}

// Summary provides a human-readable summary of the checkpoint
func (c *EpisodeCheckpoint) Summary() string {
	summary := fmt.Sprintf("Episode: %s\n", c.EpisodeID)
	summary += fmt.Sprintf("Group: %s\n", c.GroupID)
	summary += fmt.Sprintf("Progress: %s\n", c.GetProgress())
	summary += fmt.Sprintf("Created: %s\n", c.CreatedAt.Format(time.RFC3339))
	summary += fmt.Sprintf("Last Updated: %s\n", c.LastUpdatedAt.Format(time.RFC3339))
	summary += fmt.Sprintf("Attempts: %d\n", c.AttemptCount)

	if c.LastError != "" {
		summary += fmt.Sprintf("Last Error: %s\n", c.LastError)
	}

	if c.Chunks != nil {
		summary += fmt.Sprintf("Chunks: %d\n", len(c.Chunks))
	}

	if c.ExtractedNodesByChunk != nil {
		totalNodes := 0
		for _, nodes := range c.ExtractedNodesByChunk {
			totalNodes += len(nodes)
		}
		summary += fmt.Sprintf("Extracted Nodes: %d\n", totalNodes)
	}

	if c.AllResolvedNodes != nil {
		summary += fmt.Sprintf("Resolved Nodes: %d\n", len(c.AllResolvedNodes))
	}

	if c.ResolvedEdges != nil {
		summary += fmt.Sprintf("Resolved Edges: %d\n", len(c.ResolvedEdges))
	}

	return summary
}

// FindStalled returns checkpoints that haven't been updated recently
func (m *CheckpointManager) FindStalled(ctx context.Context, stalledDuration time.Duration) ([]*EpisodeCheckpoint, error) {
	checkpoints, err := m.List(ctx)
	if err != nil {
		return nil, err
	}

	cutoff := time.Now().Add(-stalledDuration)
	var stalled []*EpisodeCheckpoint

	for _, checkpoint := range checkpoints {
		if checkpoint.Step != StepCompleted && checkpoint.LastUpdatedAt.Before(cutoff) {
			stalled = append(stalled, checkpoint)
		}
	}

	return stalled, nil
}

// FindFailed returns checkpoints that have exceeded max attempts
func (m *CheckpointManager) FindFailed(ctx context.Context, maxAttempts int) ([]*EpisodeCheckpoint, error) {
	checkpoints, err := m.List(ctx)
	if err != nil {
		return nil, err
	}

	var failed []*EpisodeCheckpoint
	for _, checkpoint := range checkpoints {
		if checkpoint.Step != StepCompleted && checkpoint.AttemptCount >= maxAttempts {
			failed = append(failed, checkpoint)
		}
	}

	return failed, nil
}

// GetStatistics returns statistics about checkpoints
type CheckpointStatistics struct {
	Total      int
	Completed  int
	InProgress int
	Failed     int
	Stalled    int
	ByStep     map[ProcessingStep]int
}

func (m *CheckpointManager) GetStatistics(ctx context.Context, maxAttempts int, stalledDuration time.Duration) (*CheckpointStatistics, error) {
	checkpoints, err := m.List(ctx)
	if err != nil {
		return nil, err
	}

	stats := &CheckpointStatistics{
		Total:  len(checkpoints),
		ByStep: make(map[ProcessingStep]int),
	}

	cutoff := time.Now().Add(-stalledDuration)

	for _, checkpoint := range checkpoints {
		stats.ByStep[checkpoint.Step]++

		if checkpoint.Step == StepCompleted {
			stats.Completed++
		} else if checkpoint.AttemptCount >= maxAttempts {
			stats.Failed++
		} else if checkpoint.LastUpdatedAt.Before(cutoff) {
			stats.Stalled++
		} else {
			stats.InProgress++
		}
	}

	return stats, nil
}
