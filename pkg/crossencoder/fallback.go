package crossencoder

import (
	"context"
	"errors"
)

// FallbackClient tries Primary first and uses Fallback when Primary returns an
// error. This keeps reranking available when a remote cross-encoder service is
// configured but temporarily unavailable.
type FallbackClient struct {
	Primary  Client
	Fallback Client
}

// NewFallbackClient creates a reranker with runtime failover.
func NewFallbackClient(primary, fallback Client) *FallbackClient {
	return &FallbackClient{Primary: primary, Fallback: fallback}
}

func (c *FallbackClient) Rank(ctx context.Context, query string, passages []string) ([]RankedPassage, error) {
	if c == nil {
		return nil, errors.New("fallback reranker is nil")
	}
	if c.Primary != nil {
		ranked, err := c.Primary.Rank(ctx, query, passages)
		if err == nil {
			return ranked, nil
		}
		if c.Fallback == nil {
			return nil, err
		}
	}
	if c.Fallback == nil {
		return nil, errors.New("no reranker client available")
	}
	return c.Fallback.Rank(ctx, query, passages)
}

func (c *FallbackClient) Close() error {
	if c == nil {
		return nil
	}
	var err error
	if c.Primary != nil {
		err = errors.Join(err, c.Primary.Close())
	}
	if c.Fallback != nil {
		err = errors.Join(err, c.Fallback.Close())
	}
	return err
}
