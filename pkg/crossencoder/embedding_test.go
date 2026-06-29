package crossencoder

import (
	"context"
	"testing"
)

type closeTrackingEmbedder struct {
	closed int
}

func (e *closeTrackingEmbedder) Embed(context.Context, []string) ([][]float32, error) {
	return [][]float32{{1}, {1}}, nil
}

func (e *closeTrackingEmbedder) EmbedSingle(context.Context, string) ([]float32, error) {
	return []float32{1}, nil
}

func (e *closeTrackingEmbedder) Dimensions() int { return 1 }

func (e *closeTrackingEmbedder) Close() error {
	e.closed++
	return nil
}

func TestEmbeddingRerankerCloseSkipsBorrowedEmbedder(t *testing.T) {
	emb := &closeTrackingEmbedder{}
	client := NewEmbeddingRerankerClient(emb, EmbeddingConfig{BorrowEmbedder: true})

	if err := client.Close(); err != nil {
		t.Fatalf("Close returned error: %v", err)
	}
	if emb.closed != 0 {
		t.Fatalf("borrowed embedder was closed %d times", emb.closed)
	}
}

func TestEmbeddingRerankerCloseOwnsEmbedderByDefault(t *testing.T) {
	emb := &closeTrackingEmbedder{}
	client := NewEmbeddingRerankerClient(emb, EmbeddingConfig{})

	if err := client.Close(); err != nil {
		t.Fatalf("Close returned error: %v", err)
	}
	if emb.closed != 1 {
		t.Fatalf("owned embedder was closed %d times", emb.closed)
	}
}
