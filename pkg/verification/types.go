// Package verification provides post-hoc hallucination detection for RAG responses.
//
// It verifies each claim in an LLM-generated response against source facts
// from a knowledge graph, producing per-claim verdicts and an overall trust score.
// The pipeline runs entirely on local models (embeddings + NLI cross-encoder)
// with no LLM calls required.
package verification

import "context"

// Verdict represents the entailment classification for a claim.
type Verdict string

const (
	VerdictSupported    Verdict = "supported"
	VerdictContradicted Verdict = "contradicted"
	VerdictUnverifiable Verdict = "unverifiable"
	VerdictSkipped      Verdict = "skipped" // meta-statement or too short
)

// NLIScores holds the raw probability distribution from the NLI model.
type NLIScores struct {
	Entailment    float32
	Neutral       float32
	Contradiction float32
}

// ClaimVerdict is the verification result for a single claim.
type ClaimVerdict struct {
	// Text is the original claim sentence.
	Text string

	// Verdict is the entailment classification.
	Verdict Verdict

	// Confidence is the probability of the assigned verdict (0.0-1.0).
	Confidence float32

	// NLI holds the raw NLI scores when available.
	NLI *NLIScores

	// Evidence is the best-matching source text for this claim.
	Evidence string

	// EvidenceScore is the semantic similarity between claim and evidence (0.0-1.0).
	EvidenceScore float64

	// IsMetaStatement indicates the claim is about the system's own limitations
	// (e.g., "I don't have information about...") and should not be flagged.
	IsMetaStatement bool
}

// VerificationResult is the aggregate result of verifying all claims in a response.
type VerificationResult struct {
	// TrustScore is the overall trust score for the response (0.0-1.0).
	// Computed as the fraction of verifiable claims that are supported.
	TrustScore float64

	// Pass indicates whether the response passes the trust threshold.
	Pass bool

	// Claims contains per-claim verification results.
	Claims []ClaimVerdict

	// Stats contains summary statistics.
	Stats VerificationStats
}

// VerificationStats provides summary counts for a verification run.
type VerificationStats struct {
	TotalClaims    int
	Supported      int
	Contradicted   int
	Unverifiable   int
	Skipped        int
	MetaStatements int
}

// VerifyOptions configures the verification pipeline.
type VerifyOptions struct {
	// GateThreshold is the minimum evidence similarity score required to run NLI.
	// Claims below this threshold are marked unverifiable without NLI.
	// Default: 0.25
	GateThreshold float64

	// SupportThreshold is the minimum evidence score for a claim to be
	// potentially supported (before NLI confirmation).
	// Default: 0.40
	SupportThreshold float64

	// ContradictionThreshold is the NLI contradiction probability above which
	// a claim is flagged as contradicted.
	// Default: 0.50
	ContradictionThreshold float32

	// EntailmentThreshold is the NLI entailment probability above which
	// a claim can override a low evidence score and be marked supported.
	// Default: 0.50
	EntailmentThreshold float32

	// TrustPassScore is the minimum TrustScore for the response to pass.
	// Default: 0.70
	TrustPassScore float64
}

// DefaultVerifyOptions returns options with sensible defaults.
func DefaultVerifyOptions() *VerifyOptions {
	return &VerifyOptions{
		GateThreshold:          0.25,
		SupportThreshold:       0.40,
		ContradictionThreshold: 0.50,
		EntailmentThreshold:    0.50,
		TrustPassScore:         0.70,
	}
}

func (o *VerifyOptions) withDefaults() *VerifyOptions {
	if o == nil {
		return DefaultVerifyOptions()
	}
	d := DefaultVerifyOptions()
	if o.GateThreshold == 0 {
		o.GateThreshold = d.GateThreshold
	}
	if o.SupportThreshold == 0 {
		o.SupportThreshold = d.SupportThreshold
	}
	if o.ContradictionThreshold == 0 {
		o.ContradictionThreshold = d.ContradictionThreshold
	}
	if o.EntailmentThreshold == 0 {
		o.EntailmentThreshold = d.EntailmentThreshold
	}
	if o.TrustPassScore == 0 {
		o.TrustPassScore = d.TrustPassScore
	}
	return o
}

// NLIClient defines the interface for Natural Language Inference models.
// Implementations classify the relationship between a premise (evidence)
// and a hypothesis (claim) as entailment, neutral, or contradiction.
type NLIClient interface {
	// Classify returns NLI scores for a single premise-hypothesis pair.
	Classify(ctx context.Context, premise, hypothesis string) (*NLIScores, error)

	// ClassifyBatch returns NLI scores for multiple premise-hypothesis pairs.
	ClassifyBatch(ctx context.Context, premises, hypotheses []string) ([]NLIScores, error)

	// Close cleans up resources.
	Close() error
}
