package nlp

import (
	"context"
	"fmt"
	"time"

	"github.com/sony/gobreaker"
	"github.com/soundprediction/predicato/pkg/alert"
	"github.com/soundprediction/predicato/pkg/config"
	"github.com/soundprediction/predicato/pkg/types"
)

// CircuitBreakerClient wraps a Client with circuit breaking logic
type CircuitBreakerClient struct {
	client  Client
	cb      *gobreaker.CircuitBreaker
	alerter alert.Alerter
	name    string
}

// NewCircuitBreakerClient creates a new circuit breaker client
// Note: If cfg.Enabled is false, the wrapper is still created but the circuit breaker
// behavior is determined by the gobreaker settings. Consider checking cfg.Enabled
// at the call site if you want to skip wrapping entirely.
func NewCircuitBreakerClient(client Client, cfg config.CircuitBreakerConfig, alerter alert.Alerter, name string) *CircuitBreakerClient {
	st := gobreaker.Settings{
		Name:        name,
		MaxRequests: cfg.MaxRequests,
		Interval:    time.Duration(cfg.Interval) * time.Second,
		Timeout:     time.Duration(cfg.Timeout) * time.Second,
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			failureRatio := float64(counts.TotalFailures) / float64(counts.Requests)
			return counts.Requests >= 3 && failureRatio >= cfg.ReadyToTripRatio
		},
		OnStateChange: func(name string, from gobreaker.State, to gobreaker.State) {
			if to == gobreaker.StateOpen {
				// Trip! Alert!
				msg := fmt.Sprintf("Circuit Breaker '%s' changed status from %s to %s. Too many failures detected.", name, from, to)
				if alerter != nil {
					_ = alerter.Alert(fmt.Sprintf("URGENT: Circuit Breaker Tripped - %s", name), msg)
				}
				fmt.Println(msg)
			}
		},
	}

	return &CircuitBreakerClient{
		client:  client,
		cb:      gobreaker.NewCircuitBreaker(st),
		alerter: alerter,
		name:    name,
	}
}

// Chat implements Client
func (c *CircuitBreakerClient) Chat(ctx context.Context, messages []types.Message) (*types.Response, error) {
	resp, err := c.cb.Execute(func() (interface{}, error) {
		return c.client.Chat(ctx, messages)
	})

	if err != nil {
		return nil, err
	}
	return resp.(*types.Response), nil
}

// ChatWithStructuredOutput implements Client
func (c *CircuitBreakerClient) ChatWithStructuredOutput(ctx context.Context, messages []types.Message, schema any) (*types.Response, error) {
	resp, err := c.cb.Execute(func() (interface{}, error) {
		return c.client.ChatWithStructuredOutput(ctx, messages, schema)
	})

	if err != nil {
		return nil, err
	}
	return resp.(*types.Response), nil
}

// GetModel returns the model identifier of the wrapped client.
func (c *CircuitBreakerClient) GetModel() string {
	return c.client.GetModel()
}

// Close implements Client
func (c *CircuitBreakerClient) Close() error {
	return c.client.Close()
}

// ExtractEntities implements Client
func (c *CircuitBreakerClient) ExtractEntities(ctx context.Context, text string, entityTypes []string) ([]ExtractedEntity, error) {
	resp, err := c.cb.Execute(func() (interface{}, error) {
		return c.client.ExtractEntities(ctx, text, entityTypes)
	})

	if err != nil {
		return nil, err
	}
	return resp.([]ExtractedEntity), nil
}

// ExtractRelations implements Client
func (c *CircuitBreakerClient) ExtractRelations(ctx context.Context, text string, relationTypes []string) ([]ExtractedRelation, error) {
	resp, err := c.cb.Execute(func() (interface{}, error) {
		return c.client.ExtractRelations(ctx, text, relationTypes)
	})

	if err != nil {
		return nil, err
	}
	return resp.([]ExtractedRelation), nil
}

// Summarize implements Client
func (c *CircuitBreakerClient) Summarize(ctx context.Context, text string) (string, error) {
	resp, err := c.cb.Execute(func() (interface{}, error) {
		return c.client.Summarize(ctx, text)
	})

	if err != nil {
		return "", err
	}
	return resp.(string), nil
}

// GenerateText implements Client
func (c *CircuitBreakerClient) GenerateText(ctx context.Context, prompt string) (string, error) {
	resp, err := c.cb.Execute(func() (interface{}, error) {
		return c.client.GenerateText(ctx, prompt)
	})

	if err != nil {
		return "", err
	}
	return resp.(string), nil
}

// ExtractExtended implements Client
func (c *CircuitBreakerClient) ExtractExtended(ctx context.Context, text string, entityTypes, relationTypes []string) (*ExtendedExtractionResult, error) {
	resp, err := c.cb.Execute(func() (interface{}, error) {
		return c.client.ExtractExtended(ctx, text, entityTypes, relationTypes)
	})

	if err != nil {
		return nil, err
	}
	return resp.(*ExtendedExtractionResult), nil
}

// GetCapabilities returns the list of capabilities supported by this client.
func (c *CircuitBreakerClient) GetCapabilities() []TaskCapability {
	return c.client.GetCapabilities()
}
