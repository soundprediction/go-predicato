package verification

import (
	"context"
	"fmt"
	"log/slog"

	"github.com/soundprediction/predicato/pkg/embedder"
)

// Verifier orchestrates the claim verification pipeline:
// 1. Split response into claims
// 2. Match each claim against source facts via embeddings
// 3. Run NLI entailment check on matched pairs
// 4. Aggregate into overall trust score
type Verifier struct {
	embedder embedder.Client
	nli      NLIClient
	logger   *slog.Logger
}

// NewVerifier creates a new verification pipeline.
func NewVerifier(emb embedder.Client, nli NLIClient, logger *slog.Logger) *Verifier {
	if logger == nil {
		logger = slog.Default()
	}
	return &Verifier{
		embedder: emb,
		nli:      nli,
		logger:   logger,
	}
}

// Verify checks each claim in response against the provided source facts.
// facts should be the text of the RAG context that was provided to the LLM.
func (v *Verifier) Verify(ctx context.Context, response string, facts []string, opts *VerifyOptions) (*VerificationResult, error) {
	opts = opts.withDefaults()

	// 1. Split into claims
	claims := SplitClaims(response)
	if len(claims) == 0 {
		return &VerificationResult{
			TrustScore: 1.0,
			Pass:       true,
		}, nil
	}

	// 2. Match evidence
	matches, err := MatchEvidence(ctx, v.embedder, claims, facts)
	if err != nil {
		return nil, err
	}

	// 3. Run NLI on each claim and determine verdict
	verdicts := make([]ClaimVerdict, len(matches))
	var premiseBatch, hypothesisBatch []string
	var nliIndices []int

	for i, m := range matches {
		verdicts[i] = ClaimVerdict{
			Text:            m.Claim.Text,
			Evidence:        m.BestEvidence,
			EvidenceScore:   m.BestScore,
			IsMetaStatement: m.Claim.IsMetaStatement,
		}

		// Skip meta-statements
		if m.Claim.IsMetaStatement {
			verdicts[i].Verdict = VerdictSkipped
			verdicts[i].Confidence = 1.0
			continue
		}

		// Gate: skip NLI if evidence is too dissimilar
		if m.BestScore < opts.GateThreshold {
			verdicts[i].Verdict = VerdictUnverifiable
			verdicts[i].Confidence = float32(1.0 - m.BestScore)
			continue
		}

		// Queue for NLI batch
		premiseBatch = append(premiseBatch, m.BestEvidence)
		hypothesisBatch = append(hypothesisBatch, m.Claim.Text)
		nliIndices = append(nliIndices, i)
	}

	// Run NLI batch
	if len(premiseBatch) > 0 && v.nli != nil {
		nliResults, err := v.nli.ClassifyBatch(ctx, premiseBatch, hypothesisBatch)
		if err != nil {
			v.logger.Warn("NLI batch classification failed, marking claims unverifiable", "error", err)
			for _, idx := range nliIndices {
				verdicts[idx].Verdict = VerdictUnverifiable
			}
		} else {
			for j, idx := range nliIndices {
				scores := nliResults[j]
				verdicts[idx].NLI = &scores
				verdicts[idx].Verdict, verdicts[idx].Confidence = classifyVerdict(
					matches[idx], scores, opts,
				)
			}
		}
	} else if v.nli == nil {
		// No NLI client — fall back to evidence-score-only classification
		for _, idx := range nliIndices {
			if matches[idx].BestScore >= opts.SupportThreshold {
				verdicts[idx].Verdict = VerdictSupported
				verdicts[idx].Confidence = float32(matches[idx].BestScore)
			} else {
				verdicts[idx].Verdict = VerdictUnverifiable
				verdicts[idx].Confidence = float32(1.0 - matches[idx].BestScore)
			}
		}
	}

	// 4. Aggregate
	result := aggregate(verdicts, opts)
	return result, nil
}

// VerifyBatch verifies multiple independent (response, facts) units in a single
// pass, running ONE NLI ClassifyBatch across every gated claim-evidence pair from
// all units — instead of one ClassifyBatch per unit. It returns one result per
// unit, in input order; each result is identical to what Verify(responses[u],
// factsPer[u]) would return, but the (GPU-bound) NLI model is invoked once for the
// whole batch rather than len(responses) times. This is the amortization the
// per-claim recheck loop needs: N model-asserted claims cost one forward pass, not
// N. With a nil NLI client it falls back to embedding-similarity verdicts per unit,
// exactly like Verify.
func (v *Verifier) VerifyBatch(ctx context.Context, responses []string, factsPer [][]string, opts *VerifyOptions) ([]*VerificationResult, error) {
	if len(responses) != len(factsPer) {
		return nil, fmt.Errorf("verify batch: responses and factsPer length mismatch (%d != %d)", len(responses), len(factsPer))
	}
	opts = opts.withDefaults()

	results := make([]*VerificationResult, len(responses))
	verdictsPer := make([][]ClaimVerdict, len(responses))
	matchesPer := make([][]EvidenceMatch, len(responses))

	// pending maps a position in the global NLI batch back to its (unit, verdict).
	type pending struct{ unit, vi int }
	var premiseBatch, hypothesisBatch []string
	var pend []pending

	for u, response := range responses {
		claims := SplitClaims(response)
		if len(claims) == 0 {
			results[u] = &VerificationResult{TrustScore: 1.0, Pass: true}
			continue
		}
		matches, err := MatchEvidence(ctx, v.embedder, claims, factsPer[u])
		if err != nil {
			return nil, err
		}
		matchesPer[u] = matches

		verdicts := make([]ClaimVerdict, len(matches))
		for i, m := range matches {
			verdicts[i] = ClaimVerdict{
				Text:            m.Claim.Text,
				Evidence:        m.BestEvidence,
				EvidenceScore:   m.BestScore,
				IsMetaStatement: m.Claim.IsMetaStatement,
			}
			switch {
			case m.Claim.IsMetaStatement:
				verdicts[i].Verdict = VerdictSkipped
				verdicts[i].Confidence = 1.0
			case m.BestScore < opts.GateThreshold:
				verdicts[i].Verdict = VerdictUnverifiable
				verdicts[i].Confidence = float32(1.0 - m.BestScore)
			case v.nli == nil:
				// Embedding-similarity fallback (identical to Verify's nil-NLI path).
				if m.BestScore >= opts.SupportThreshold {
					verdicts[i].Verdict = VerdictSupported
					verdicts[i].Confidence = float32(m.BestScore)
				} else {
					verdicts[i].Verdict = VerdictUnverifiable
					verdicts[i].Confidence = float32(1.0 - m.BestScore)
				}
			default:
				premiseBatch = append(premiseBatch, m.BestEvidence)
				hypothesisBatch = append(hypothesisBatch, m.Claim.Text)
				pend = append(pend, pending{unit: u, vi: i})
			}
		}
		verdictsPer[u] = verdicts
	}

	// Single NLI pass over every gated pair from all units.
	if len(premiseBatch) > 0 && v.nli != nil {
		nliResults, err := v.nli.ClassifyBatch(ctx, premiseBatch, hypothesisBatch)
		if err != nil {
			v.logger.Warn("NLI batch classification failed, marking claims unverifiable", "error", err)
			for _, p := range pend {
				verdictsPer[p.unit][p.vi].Verdict = VerdictUnverifiable
			}
		} else {
			for k, p := range pend {
				scores := nliResults[k]
				verdictsPer[p.unit][p.vi].NLI = &scores
				verdictsPer[p.unit][p.vi].Verdict, verdictsPer[p.unit][p.vi].Confidence =
					classifyVerdict(matchesPer[p.unit][p.vi], scores, opts)
			}
		}
	}

	for u := range responses {
		if results[u] != nil {
			continue // empty-claims short-circuit set above
		}
		results[u] = aggregate(verdictsPer[u], opts)
	}
	return results, nil
}

// classifyVerdict determines the verdict for a claim based on NLI scores and evidence match.
func classifyVerdict(match EvidenceMatch, nli NLIScores, opts *VerifyOptions) (Verdict, float32) {
	// Contradiction takes priority
	if nli.Contradiction > opts.ContradictionThreshold {
		return VerdictContradicted, nli.Contradiction
	}

	// Supported: good evidence score OR strong entailment, AND no contradiction
	if (match.BestScore >= opts.SupportThreshold || nli.Entailment > opts.EntailmentThreshold) &&
		nli.Contradiction <= opts.ContradictionThreshold {
		return VerdictSupported, nli.Entailment
	}

	// Hallucination pattern + low evidence + low entailment = contradicted
	if match.Claim.HasHallucinationPattern &&
		match.BestScore < 0.20 &&
		nli.Entailment < 0.3 {
		return VerdictContradicted, 1.0 - nli.Entailment
	}

	return VerdictUnverifiable, nli.Neutral
}

// aggregate computes the overall trust score and stats.
func aggregate(verdicts []ClaimVerdict, opts *VerifyOptions) *VerificationResult {
	var stats VerificationStats
	stats.TotalClaims = len(verdicts)

	for _, v := range verdicts {
		switch v.Verdict {
		case VerdictSupported:
			stats.Supported++
		case VerdictContradicted:
			stats.Contradicted++
		case VerdictUnverifiable:
			stats.Unverifiable++
		case VerdictSkipped:
			stats.Skipped++
		}
		if v.IsMetaStatement {
			stats.MetaStatements++
		}
	}

	// Trust score = supported / (supported + contradicted + unverifiable)
	verifiable := stats.Supported + stats.Contradicted + stats.Unverifiable
	trustScore := 1.0
	if verifiable > 0 {
		trustScore = float64(stats.Supported) / float64(verifiable)
	}

	return &VerificationResult{
		TrustScore: trustScore,
		Pass:       trustScore >= opts.TrustPassScore,
		Claims:     verdicts,
		Stats:      stats,
	}
}
