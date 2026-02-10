package nlp_test

import (
	"testing"

	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/stretchr/testify/assert"
)

func TestGetProvider(t *testing.T) {
	tests := []struct {
		id      nlp.ProviderID
		want    nlp.Provider
		wantErr bool
	}{
		{
			id: nlp.ProviderCandle,
			want: nlp.Provider{
				ID:          nlp.ProviderCandle,
				Name:        "Candle",
				Description: "Local ML models via HuggingFace candle, GLiNER, and GLiNER2 (pure Rust, auto-downloads from HF Hub)",
				IsLocal:     true,
			},
			wantErr: false,
		},
		{
			id:      "nonexistent",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(string(tt.id), func(t *testing.T) {
			got, found := nlp.GetProvider(tt.id)
			if tt.wantErr {
				assert.False(t, found)
			} else {
				assert.True(t, found)
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestGetModel(t *testing.T) {
	t.Run("Candle Embedding Model", func(t *testing.T) {
		id := "sentence-transformers/all-MiniLM-L6-v2"
		got, found := nlp.GetModel(id)
		assert.True(t, found)
		assert.Equal(t, id, got.ID)
		assert.Equal(t, nlp.ProviderCandle, got.ProviderID)
		assert.Contains(t, got.Capabilities, nlp.TaskEmbedding)
	})

	t.Run("Candle Summarization Model", func(t *testing.T) {
		id := "google/flan-t5-base"
		got, found := nlp.GetModel(id)
		assert.True(t, found)
		assert.Equal(t, id, got.ID)
		assert.Equal(t, nlp.ProviderCandle, got.ProviderID)
		assert.Contains(t, got.Capabilities, nlp.TaskSummarization)
	})

	t.Run("GLiNER Model", func(t *testing.T) {
		id := "urchade/gliner_multi-v2.1"
		got, found := nlp.GetModel(id)
		assert.True(t, found)
		assert.Equal(t, id, got.ID)
		assert.Contains(t, got.Capabilities, nlp.TaskNamedEntityRecognition)
		assert.Contains(t, got.Capabilities, nlp.TaskRelationExtraction)
	})

	t.Run("Nonexistent Model", func(t *testing.T) {
		_, found := nlp.GetModel("fake-model")
		assert.False(t, found)
	})
}

func TestGetModelsByProvider(t *testing.T) {
	models := nlp.GetModelsByProvider(nlp.ProviderCandle)
	assert.NotEmpty(t, models)
	for _, m := range models {
		assert.Equal(t, nlp.ProviderCandle, m.ProviderID)
	}
}

func TestGetModelsByCapability(t *testing.T) {
	t.Run("Embedding", func(t *testing.T) {
		models := nlp.GetModelsByCapability(nlp.TaskEmbedding)
		assert.NotEmpty(t, models)
		for _, m := range models {
			assert.Contains(t, m.Capabilities, nlp.TaskEmbedding)
		}
	})

	t.Run("NER", func(t *testing.T) {
		models := nlp.GetModelsByCapability(nlp.TaskNamedEntityRecognition)
		assert.NotEmpty(t, models)
		hasGliner := false
		for _, m := range models {
			if m.ProviderID == nlp.ProviderGLiNER {
				hasGliner = true
			}
		}
		assert.True(t, hasGliner, "Should have GLiNER NER models")
	})
}
