package embedder

import (
	"context"
	"errors"
)

// FallbackClient tries Primary first and uses Fallback when Primary returns an
// error. This is intended for deployments where a remote embedding service is
// preferred but local inference should keep retrieval available during outages.
type FallbackClient struct {
	Primary  Client
	Fallback Client
}

// NewFallbackClient creates an embedding client with runtime failover.
func NewFallbackClient(primary, fallback Client) *FallbackClient {
	return &FallbackClient{Primary: primary, Fallback: fallback}
}

func (c *FallbackClient) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if c == nil {
		return nil, errors.New("fallback embedder is nil")
	}
	if c.Primary != nil {
		embeddings, err := c.Primary.Embed(ctx, texts)
		if err == nil {
			return embeddings, nil
		}
		if c.Fallback == nil {
			return nil, err
		}
	}
	if c.Fallback == nil {
		return nil, errors.New("no embedding client available")
	}
	return c.Fallback.Embed(ctx, texts)
}

func (c *FallbackClient) EmbedSingle(ctx context.Context, text string) ([]float32, error) {
	if c == nil {
		return nil, errors.New("fallback embedder is nil")
	}
	if c.Primary != nil {
		embedding, err := c.Primary.EmbedSingle(ctx, text)
		if err == nil {
			return embedding, nil
		}
		if c.Fallback == nil {
			return nil, err
		}
	}
	if c.Fallback == nil {
		return nil, errors.New("no embedding client available")
	}
	return c.Fallback.EmbedSingle(ctx, text)
}

func (c *FallbackClient) Dimensions() int {
	if c == nil {
		return 0
	}
	if c.Primary != nil {
		if dims := c.Primary.Dimensions(); dims > 0 {
			return dims
		}
	}
	if c.Fallback != nil {
		return c.Fallback.Dimensions()
	}
	return 0
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
