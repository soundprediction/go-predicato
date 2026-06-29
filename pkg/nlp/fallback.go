package nlp

import (
	"context"
	"errors"
	"sync"

	"github.com/soundprediction/predicato/pkg/types"
)

// FallbackClient tries Primary first and uses Fallback when Primary returns an
// error. It is intended for remote-first extraction deployments where a local
// model service should keep ingestion available during remote outages.
type FallbackClient struct {
	Primary         Client
	Fallback        Client
	fallbackFactory func() (Client, error)
	fallbackMu      sync.Mutex
}

// NewFallbackClient creates an NLP client with runtime failover.
func NewFallbackClient(primary, fallback Client) *FallbackClient {
	return &FallbackClient{Primary: primary, Fallback: fallback}
}

// NewLazyFallbackClient creates an NLP client whose fallback is initialized only
// after the primary returns an error.
func NewLazyFallbackClient(primary Client, fallbackFactory func() (Client, error)) *FallbackClient {
	return &FallbackClient{Primary: primary, fallbackFactory: fallbackFactory}
}

func (c *FallbackClient) getFallback() (Client, error) {
	if c.Fallback != nil {
		return c.Fallback, nil
	}
	if c.fallbackFactory == nil {
		return nil, nil
	}
	c.fallbackMu.Lock()
	defer c.fallbackMu.Unlock()
	if c.Fallback != nil {
		return c.Fallback, nil
	}
	fallback, err := c.fallbackFactory()
	if err != nil {
		return nil, err
	}
	c.Fallback = fallback
	return c.Fallback, nil
}

func (c *FallbackClient) Chat(ctx context.Context, messages []types.Message) (*types.Response, error) {
	if c.Primary != nil {
		resp, err := c.Primary.Chat(ctx, messages)
		if err == nil {
			return resp, nil
		}
		fallback, fallbackErr := c.getFallback()
		if fallbackErr != nil {
			return nil, errors.Join(err, fallbackErr)
		}
		if fallback == nil {
			return nil, err
		}
		return fallback.Chat(ctx, messages)
	}
	fallback, err := c.getFallback()
	if err != nil {
		return nil, err
	}
	if fallback == nil {
		return nil, errors.New("no NLP client available")
	}
	return fallback.Chat(ctx, messages)
}

func (c *FallbackClient) ChatWithStructuredOutput(ctx context.Context, messages []types.Message, schema any) (*types.Response, error) {
	if c.Primary != nil {
		resp, err := c.Primary.ChatWithStructuredOutput(ctx, messages, schema)
		if err == nil {
			return resp, nil
		}
		fallback, fallbackErr := c.getFallback()
		if fallbackErr != nil {
			return nil, errors.Join(err, fallbackErr)
		}
		if fallback == nil {
			return nil, err
		}
		return fallback.ChatWithStructuredOutput(ctx, messages, schema)
	}
	fallback, err := c.getFallback()
	if err != nil {
		return nil, err
	}
	if fallback == nil {
		return nil, errors.New("no NLP client available")
	}
	return fallback.ChatWithStructuredOutput(ctx, messages, schema)
}

func (c *FallbackClient) ExtractEntities(ctx context.Context, text string, entityTypes []string) ([]ExtractedEntity, error) {
	if c.Primary != nil {
		entities, err := c.Primary.ExtractEntities(ctx, text, entityTypes)
		if err == nil {
			return entities, nil
		}
		fallback, fallbackErr := c.getFallback()
		if fallbackErr != nil {
			return nil, errors.Join(err, fallbackErr)
		}
		if fallback == nil {
			return nil, err
		}
		return fallback.ExtractEntities(ctx, text, entityTypes)
	}
	fallback, err := c.getFallback()
	if err != nil {
		return nil, err
	}
	if fallback == nil {
		return nil, errors.New("no NLP client available")
	}
	return fallback.ExtractEntities(ctx, text, entityTypes)
}

func (c *FallbackClient) ExtractRelations(ctx context.Context, text string, relationTypes []string) ([]ExtractedRelation, error) {
	if c.Primary != nil {
		relations, err := c.Primary.ExtractRelations(ctx, text, relationTypes)
		if err == nil {
			return relations, nil
		}
		fallback, fallbackErr := c.getFallback()
		if fallbackErr != nil {
			return nil, errors.Join(err, fallbackErr)
		}
		if fallback == nil {
			return nil, err
		}
		return fallback.ExtractRelations(ctx, text, relationTypes)
	}
	fallback, err := c.getFallback()
	if err != nil {
		return nil, err
	}
	if fallback == nil {
		return nil, errors.New("no NLP client available")
	}
	return fallback.ExtractRelations(ctx, text, relationTypes)
}

func (c *FallbackClient) ExtractExtended(ctx context.Context, text string, entityTypes, relationTypes []string) (*ExtendedExtractionResult, error) {
	if c.Primary != nil {
		result, err := c.Primary.ExtractExtended(ctx, text, entityTypes, relationTypes)
		if err == nil {
			return result, nil
		}
		fallback, fallbackErr := c.getFallback()
		if fallbackErr != nil {
			return nil, errors.Join(err, fallbackErr)
		}
		if fallback == nil {
			return nil, err
		}
		return fallback.ExtractExtended(ctx, text, entityTypes, relationTypes)
	}
	fallback, err := c.getFallback()
	if err != nil {
		return nil, err
	}
	if fallback == nil {
		return nil, errors.New("no NLP client available")
	}
	return fallback.ExtractExtended(ctx, text, entityTypes, relationTypes)
}

func (c *FallbackClient) Summarize(ctx context.Context, text string) (string, error) {
	if c.Primary != nil {
		summary, err := c.Primary.Summarize(ctx, text)
		if err == nil {
			return summary, nil
		}
		fallback, fallbackErr := c.getFallback()
		if fallbackErr != nil {
			return "", errors.Join(err, fallbackErr)
		}
		if fallback == nil {
			return "", err
		}
		return fallback.Summarize(ctx, text)
	}
	fallback, err := c.getFallback()
	if err != nil {
		return "", err
	}
	if fallback == nil {
		return "", errors.New("no NLP client available")
	}
	return fallback.Summarize(ctx, text)
}

func (c *FallbackClient) GenerateText(ctx context.Context, prompt string) (string, error) {
	if c.Primary != nil {
		text, err := c.Primary.GenerateText(ctx, prompt)
		if err == nil {
			return text, nil
		}
		fallback, fallbackErr := c.getFallback()
		if fallbackErr != nil {
			return "", errors.Join(err, fallbackErr)
		}
		if fallback == nil {
			return "", err
		}
		return fallback.GenerateText(ctx, prompt)
	}
	fallback, err := c.getFallback()
	if err != nil {
		return "", err
	}
	if fallback == nil {
		return "", errors.New("no NLP client available")
	}
	return fallback.GenerateText(ctx, prompt)
}

func (c *FallbackClient) GetModel() string {
	if c.Primary != nil {
		if model := c.Primary.GetModel(); model != "" {
			return model
		}
	}
	if c.Fallback != nil {
		return c.Fallback.GetModel()
	}
	return ""
}

func (c *FallbackClient) GetCapabilities() []TaskCapability {
	if c.Primary != nil {
		return c.Primary.GetCapabilities()
	}
	if c.Fallback != nil {
		return c.Fallback.GetCapabilities()
	}
	return nil
}

func (c *FallbackClient) Close() error {
	var err error
	if c.Primary != nil {
		err = errors.Join(err, c.Primary.Close())
	}
	if c.Fallback != nil {
		err = errors.Join(err, c.Fallback.Close())
	}
	return err
}
