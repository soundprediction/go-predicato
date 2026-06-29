package embedder

import (
	"context"
	"errors"
	"testing"
)

type stubEmbedder struct {
	embed       [][]float32
	embedSingle []float32
	err         error
	dims        int
	called      int
}

func (s *stubEmbedder) Embed(context.Context, []string) ([][]float32, error) {
	s.called++
	if s.err != nil {
		return nil, s.err
	}
	return s.embed, nil
}

func (s *stubEmbedder) EmbedSingle(context.Context, string) ([]float32, error) {
	s.called++
	if s.err != nil {
		return nil, s.err
	}
	return s.embedSingle, nil
}

func (s *stubEmbedder) Dimensions() int { return s.dims }
func (s *stubEmbedder) Close() error    { return nil }

func TestFallbackClientUsesPrimaryWhenHealthy(t *testing.T) {
	primary := &stubEmbedder{embed: [][]float32{{1, 2}}, dims: 2}
	fallback := &stubEmbedder{embed: [][]float32{{3, 4}}, dims: 2}
	client := NewFallbackClient(primary, fallback)

	got, err := client.Embed(context.Background(), []string{"a"})
	if err != nil {
		t.Fatalf("Embed returned error: %v", err)
	}
	if got[0][0] != 1 || fallback.called != 0 {
		t.Fatalf("got %v, fallback calls %d", got, fallback.called)
	}
}

func TestFallbackClientUsesFallbackOnPrimaryError(t *testing.T) {
	primary := &stubEmbedder{err: errors.New("remote down"), dims: 2}
	fallback := &stubEmbedder{embed: [][]float32{{3, 4}}, dims: 2}
	client := NewFallbackClient(primary, fallback)

	got, err := client.Embed(context.Background(), []string{"a"})
	if err != nil {
		t.Fatalf("Embed returned error: %v", err)
	}
	if got[0][0] != 3 || primary.called != 1 || fallback.called != 1 {
		t.Fatalf("got %v, primary calls %d, fallback calls %d", got, primary.called, fallback.called)
	}
}
