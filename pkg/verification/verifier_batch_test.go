package verification

import (
	"context"
	"testing"
)

// hashEmbedder is a deterministic, dependency-free embedder: identical text always
// maps to the same vector (so a fact equal to a claim scores cosine ~1.0), and the
// mapping is stable across calls, which is all the equivalence test needs.
type hashEmbedder struct{}

func (hashEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, 8)
		for j := 0; j < len(t); j++ {
			v[j%8] += float32(t[j])
		}
		out[i] = v
	}
	return out, nil
}
func (hashEmbedder) EmbedSingle(ctx context.Context, text string) ([]float32, error) {
	v, err := hashEmbedder{}.Embed(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return v[0], nil
}
func (hashEmbedder) Dimensions() int { return 8 }
func (hashEmbedder) Close() error    { return nil }

// countingNLI returns a deterministic verdict (entailment when premise==hypothesis,
// else neutral) and counts how many ClassifyBatch calls it receives, so the test
// can assert VerifyBatch makes exactly one regardless of unit count.
type countingNLI struct{ calls int }

func (c *countingNLI) Classify(_ context.Context, premise, hypothesis string) (*NLIScores, error) {
	c.calls++
	return scoreFor(premise, hypothesis), nil
}
func (c *countingNLI) ClassifyBatch(_ context.Context, premises, hypotheses []string) ([]NLIScores, error) {
	c.calls++
	out := make([]NLIScores, len(premises))
	for i := range premises {
		out[i] = *scoreFor(premises[i], hypotheses[i])
	}
	return out, nil
}
func (c *countingNLI) Close() error { return nil }
func scoreFor(premise, hypothesis string) *NLIScores {
	if premise == hypothesis {
		return &NLIScores{Entailment: 0.95, Neutral: 0.04, Contradiction: 0.01}
	}
	return &NLIScores{Entailment: 0.10, Neutral: 0.80, Contradiction: 0.10}
}

// TestVerifyBatchMatchesPerUnitVerify is the contract: VerifyBatch over N units must
// produce, unit-for-unit, exactly what Verify produces on each unit alone — while
// invoking the NLI model once instead of N times.
func TestVerifyBatchMatchesPerUnitVerify(t *testing.T) {
	responses := []string{
		"Aspirin reduces fever in adults.",
		"The patient has no documented allergies.",
		"Metformin is first-line therapy for type 2 diabetes.",
	}
	factsPer := [][]string{
		{"Aspirin reduces fever in adults.", "Aspirin is an analgesic."},
		{"Penicillin allergy was recorded last year."},
		{"Metformin is first-line therapy for type 2 diabetes."},
	}

	// Per-unit baseline (one ClassifyBatch call each).
	perUnit := &countingNLI{}
	vSingle := NewVerifier(hashEmbedder{}, perUnit, nil)
	want := make([]*VerificationResult, len(responses))
	for i := range responses {
		r, err := vSingle.Verify(context.Background(), responses[i], factsPer[i], nil)
		if err != nil {
			t.Fatalf("Verify[%d]: %v", i, err)
		}
		want[i] = r
	}

	// Batched (one ClassifyBatch call total).
	batched := &countingNLI{}
	vBatch := NewVerifier(hashEmbedder{}, batched, nil)
	got, err := vBatch.VerifyBatch(context.Background(), responses, factsPer, nil)
	if err != nil {
		t.Fatalf("VerifyBatch: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d results, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i].TrustScore != want[i].TrustScore {
			t.Errorf("unit %d trust score: got %v want %v", i, got[i].TrustScore, want[i].TrustScore)
		}
		if got[i].Stats != want[i].Stats {
			t.Errorf("unit %d stats: got %+v want %+v", i, got[i].Stats, want[i].Stats)
		}
		if len(got[i].Claims) != len(want[i].Claims) {
			t.Fatalf("unit %d claim count: got %d want %d", i, len(got[i].Claims), len(want[i].Claims))
		}
		for j := range want[i].Claims {
			if got[i].Claims[j].Verdict != want[i].Claims[j].Verdict {
				t.Errorf("unit %d claim %d verdict: got %v want %v", i, j,
					got[i].Claims[j].Verdict, want[i].Claims[j].Verdict)
			}
		}
	}

	// The whole point: one model invocation for the batch, vs one per unit.
	if batched.calls != 1 {
		t.Errorf("VerifyBatch made %d NLI calls, want 1", batched.calls)
	}
	if perUnit.calls != len(responses) {
		t.Errorf("per-unit baseline made %d NLI calls, want %d", perUnit.calls, len(responses))
	}
}

// TestVerifyBatchNilNLI confirms the embedding-only fallback (used on CPU-only
// deployments where the NLI model is disabled) still matches per-unit Verify and
// never touches a model.
func TestVerifyBatchNilNLI(t *testing.T) {
	responses := []string{"Aspirin reduces fever.", "Unrelated filler sentence here."}
	factsPer := [][]string{
		{"Aspirin reduces fever."},
		{"A completely different topic about ships."},
	}
	v := NewVerifier(hashEmbedder{}, nil, nil)

	want := make([]*VerificationResult, len(responses))
	for i := range responses {
		r, err := v.Verify(context.Background(), responses[i], factsPer[i], nil)
		if err != nil {
			t.Fatalf("Verify[%d]: %v", i, err)
		}
		want[i] = r
	}
	got, err := v.VerifyBatch(context.Background(), responses, factsPer, nil)
	if err != nil {
		t.Fatalf("VerifyBatch: %v", err)
	}
	for i := range want {
		if got[i].TrustScore != want[i].TrustScore || got[i].Stats != want[i].Stats {
			t.Errorf("unit %d: got trust=%v stats=%+v want trust=%v stats=%+v",
				i, got[i].TrustScore, got[i].Stats, want[i].TrustScore, want[i].Stats)
		}
	}
}

// TestVerifyBatchLengthMismatch guards the input contract.
func TestVerifyBatchLengthMismatch(t *testing.T) {
	v := NewVerifier(hashEmbedder{}, nil, nil)
	if _, err := v.VerifyBatch(context.Background(), []string{"a", "b"}, [][]string{{"x"}}, nil); err == nil {
		t.Fatal("expected error on responses/factsPer length mismatch, got nil")
	}
}
