package nlp

import (
	"context"
	"errors"
	"testing"

	"github.com/soundprediction/predicato/pkg/types"
)

type fallbackTestClient struct {
	entities []ExtractedEntity
	err      error
	called   int
	closed   int
}

func (c *fallbackTestClient) Chat(context.Context, []types.Message) (*types.Response, error) {
	return nil, errors.New("not implemented")
}

func (c *fallbackTestClient) ChatWithStructuredOutput(context.Context, []types.Message, any) (*types.Response, error) {
	return nil, errors.New("not implemented")
}

func (c *fallbackTestClient) ExtractEntities(context.Context, string, []string) ([]ExtractedEntity, error) {
	c.called++
	if c.err != nil {
		return nil, c.err
	}
	return c.entities, nil
}

func (c *fallbackTestClient) ExtractRelations(context.Context, string, []string) ([]ExtractedRelation, error) {
	return nil, errors.New("not implemented")
}

func (c *fallbackTestClient) ExtractExtended(context.Context, string, []string, []string) (*ExtendedExtractionResult, error) {
	return nil, errors.New("not implemented")
}

func (c *fallbackTestClient) Summarize(context.Context, string) (string, error) {
	return "", errors.New("not implemented")
}

func (c *fallbackTestClient) GenerateText(context.Context, string) (string, error) {
	return "", errors.New("not implemented")
}

func (c *fallbackTestClient) GetModel() string { return "test" }

func (c *fallbackTestClient) GetCapabilities() []TaskCapability { return nil }

func (c *fallbackTestClient) Close() error {
	c.closed++
	return nil
}

func TestFallbackClientUsesFallbackOnPrimaryError(t *testing.T) {
	primary := &fallbackTestClient{err: errors.New("remote down")}
	fallback := &fallbackTestClient{entities: []ExtractedEntity{{Text: "fallback"}}}
	client := NewFallbackClient(primary, fallback)

	got, err := client.ExtractEntities(context.Background(), "text", []string{"condition"})
	if err != nil {
		t.Fatalf("ExtractEntities returned error: %v", err)
	}
	if got[0].Text != "fallback" || primary.called != 1 || fallback.called != 1 {
		t.Fatalf("got %v, primary calls %d, fallback calls %d", got, primary.called, fallback.called)
	}
}

func TestFallbackClientCloseClosesBothClients(t *testing.T) {
	primary := &fallbackTestClient{}
	fallback := &fallbackTestClient{}
	client := NewFallbackClient(primary, fallback)

	if err := client.Close(); err != nil {
		t.Fatalf("Close returned error: %v", err)
	}
	if primary.closed != 1 || fallback.closed != 1 {
		t.Fatalf("primary closed %d times, fallback closed %d times", primary.closed, fallback.closed)
	}
}

func TestLazyFallbackClientInitializesFallbackOnlyAfterPrimaryError(t *testing.T) {
	primary := &fallbackTestClient{entities: []ExtractedEntity{{Text: "primary"}}}
	fallbackCalls := 0
	client := NewLazyFallbackClient(primary, func() (Client, error) {
		fallbackCalls++
		return &fallbackTestClient{entities: []ExtractedEntity{{Text: "fallback"}}}, nil
	})

	got, err := client.ExtractEntities(context.Background(), "text", []string{"condition"})
	if err != nil {
		t.Fatalf("ExtractEntities returned error: %v", err)
	}
	if got[0].Text != "primary" || fallbackCalls != 0 {
		t.Fatalf("got %v, fallback factory calls %d", got, fallbackCalls)
	}

	primary.err = errors.New("remote down")
	got, err = client.ExtractEntities(context.Background(), "text", []string{"condition"})
	if err != nil {
		t.Fatalf("ExtractEntities returned error after primary failure: %v", err)
	}
	if got[0].Text != "fallback" || fallbackCalls != 1 {
		t.Fatalf("got %v, fallback factory calls %d", got, fallbackCalls)
	}
}
