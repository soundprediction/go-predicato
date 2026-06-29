package crossencoder

import (
	"context"
	"errors"
	"testing"
)

type stubReranker struct {
	ranked []RankedPassage
	err    error
	called int
}

func (s *stubReranker) Rank(context.Context, string, []string) ([]RankedPassage, error) {
	s.called++
	if s.err != nil {
		return nil, s.err
	}
	return s.ranked, nil
}

func (s *stubReranker) Close() error { return nil }

func TestFallbackRerankerUsesPrimaryWhenHealthy(t *testing.T) {
	primary := &stubReranker{ranked: []RankedPassage{{Passage: "primary", Score: 1}}}
	fallback := &stubReranker{ranked: []RankedPassage{{Passage: "fallback", Score: 1}}}
	client := NewFallbackClient(primary, fallback)

	got, err := client.Rank(context.Background(), "q", []string{"p"})
	if err != nil {
		t.Fatalf("Rank returned error: %v", err)
	}
	if got[0].Passage != "primary" || fallback.called != 0 {
		t.Fatalf("got %v, fallback calls %d", got, fallback.called)
	}
}

func TestFallbackRerankerUsesFallbackOnPrimaryError(t *testing.T) {
	primary := &stubReranker{err: errors.New("remote down")}
	fallback := &stubReranker{ranked: []RankedPassage{{Passage: "fallback", Score: 1}}}
	client := NewFallbackClient(primary, fallback)

	got, err := client.Rank(context.Background(), "q", []string{"p"})
	if err != nil {
		t.Fatalf("Rank returned error: %v", err)
	}
	if got[0].Passage != "fallback" || primary.called != 1 || fallback.called != 1 {
		t.Fatalf("got %v, primary calls %d, fallback calls %d", got, primary.called, fallback.called)
	}
}
