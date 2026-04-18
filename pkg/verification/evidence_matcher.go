package verification

import (
	"context"
	"math"

	"github.com/soundprediction/predicato/pkg/embedder"
)

// EvidenceMatch represents a claim paired with its best-matching source evidence.
type EvidenceMatch struct {
	Claim         Claim
	BestEvidence  string
	BestScore     float64
	AvgScore      float64
}

// MatchEvidence finds the best-matching source fact for each claim using
// bi-encoder embeddings and cosine similarity.
func MatchEvidence(
	ctx context.Context,
	emb embedder.Client,
	claims []Claim,
	facts []string,
) ([]EvidenceMatch, error) {
	if len(claims) == 0 || len(facts) == 0 {
		matches := make([]EvidenceMatch, len(claims))
		for i, c := range claims {
			matches[i] = EvidenceMatch{Claim: c}
		}
		return matches, nil
	}

	// Batch-embed all texts together: claims first, then facts
	allTexts := make([]string, 0, len(claims)+len(facts))
	for _, c := range claims {
		allTexts = append(allTexts, c.Text)
	}
	allTexts = append(allTexts, facts...)

	allEmbeddings, err := emb.Embed(ctx, allTexts)
	if err != nil {
		return nil, err
	}

	claimEmbeddings := allEmbeddings[:len(claims)]
	factEmbeddings := allEmbeddings[len(claims):]

	matches := make([]EvidenceMatch, len(claims))
	for i, claimEmb := range claimEmbeddings {
		bestIdx := -1
		bestScore := -1.0
		var totalScore float64

		for j, factEmb := range factEmbeddings {
			score := cosine(claimEmb, factEmb)
			totalScore += score
			if score > bestScore {
				bestScore = score
				bestIdx = j
			}
		}

		matches[i] = EvidenceMatch{
			Claim:    claims[i],
			BestScore: bestScore,
			AvgScore:  totalScore / float64(len(facts)),
		}
		if bestIdx >= 0 {
			matches[i].BestEvidence = facts[bestIdx]
		}
	}

	return matches, nil
}

func cosine(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}
