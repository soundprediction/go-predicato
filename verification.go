package predicato

import (
	"context"

	"github.com/soundprediction/predicato/pkg/verification"
)

// SetVerifier sets the verification pipeline on the client.
// The verifier uses the client's embedder for evidence matching
// and the provided NLI client for entailment classification.
func (c *Client) SetVerifier(nli verification.NLIClient) {
	c.verifier = verification.NewVerifier(c.embedder, nli, c.logger)
}

// VerifyResponse checks each claim in an LLM response against source facts.
// facts should be the RAG context texts that were provided to the LLM.
// Returns per-claim verdicts and an overall trust score.
//
// If no verifier has been configured via SetVerifier, this returns nil
// (verification is optional and disabled by default).
func (c *Client) VerifyResponse(ctx context.Context, response string, facts []string, opts *verification.VerifyOptions) (*verification.VerificationResult, error) {
	if c.verifier == nil {
		return nil, nil
	}
	return c.verifier.Verify(ctx, response, facts, opts)
}

// GetVerifier returns the configured verifier, or nil if not set.
func (c *Client) GetVerifier() *verification.Verifier {
	return c.verifier
}
