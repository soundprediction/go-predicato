package gliner

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/soundprediction/go-gline-rs/pkg/gline"
)

type Client struct {
	spanModel     *gline.Model
	relationModel *gline.RelationModel
	mu            sync.Mutex
}

func NewClient(modelID string) (*Client, error) {
	// Ensure init
	if err := gline.Init(); err != nil {
		return nil, fmt.Errorf("failed to init gline: %w", err)
	}

	// Use download + load or direct load.
	// Since we added NewSpanModelFromHF, let's use that if it's a HF ID.
	// But wait, the user might pass a local path too?
	// Let's support HF ID for now as primary.

	// Check if modelID looks like a path
	if _, err := os.Stat(modelID); err == nil {
		// It's a directory or file.
		// If it's a dir, assume it has model.onnx and tokenizer.json?
		// go-gline-rs NewSpanModel expects file paths.
		// Let's assume standard structure: modelID/model.onnx
		modelPath := filepath.Join(modelID, "model.onnx")
		tokPath := filepath.Join(modelID, "tokenizer.json")
		m, err := gline.NewSpanModel(modelPath, tokPath)
		if err != nil {
			return nil, err
		}
		return &Client{spanModel: m}, nil
	}

	// Assume HF ID
	m, err := gline.NewSpanModelFromHF(modelID)
	if err != nil {
		return nil, err
	}
	return &Client{spanModel: m}, nil
}

func (c *Client) LoadRelationModel(modelID string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if _, err := os.Stat(modelID); err == nil {
		modelPath := filepath.Join(modelID, "model.onnx")
		tokPath := filepath.Join(modelID, "tokenizer.json")
		m, err := gline.NewRelationModel(modelPath, tokPath)
		if err != nil {
			return err
		}
		c.relationModel = m
		return nil
	}

	m, err := gline.NewRelationModelFromHF(modelID)
	if err != nil {
		return err
	}
	c.relationModel = m
	return nil
}

func (c *Client) Close() {
	if c.spanModel != nil {
		c.spanModel.Close()
	}
	if c.relationModel != nil {
		c.relationModel.Close()
	}
}

type Entity struct {
	Text  string
	Label string
	Score float32
}

type Relation struct {
	Source string
	Target string
	Type   string
	Score  float32
}

func (c *Client) ExtractEntities(text string, labels []string) ([]Entity, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.spanModel == nil {
		return nil, fmt.Errorf("span model not loaded")
	}

	results, err := c.spanModel.Predict([]string{text}, labels)
	if err != nil {
		return nil, err
	}

	if len(results) == 0 {
		return []Entity{}, nil
	}

	var entities []Entity
	for _, e := range results[0] {
		entities = append(entities, Entity{
			Text:  e.Text,
			Label: e.Label,
			Score: e.Probability,
		})
	}
	return entities, nil
}

func (c *Client) ExtractRelations(text string, entityLabels []string, schema map[string][2][]string) ([]Relation, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.relationModel == nil {
		return nil, fmt.Errorf("relation model not loaded")
	}

	// Add schema
	for rel, types := range schema {
		if err := c.relationModel.AddRelationSchema(rel, types[0], types[1]); err != nil {
			return nil, fmt.Errorf("failed to add schema for %s: %w", rel, err)
		}
	}

	results, err := c.relationModel.Predict([]string{text}, entityLabels)
	if err != nil {
		return nil, err
	}

	if len(results) == 0 {
		return []Relation{}, nil
	}

	var relations []Relation
	for _, r := range results[0] {
		relations = append(relations, Relation{
			Source: r.Source,
			Target: r.Target,
			Type:   r.Relation,
			Score:  r.Probability,
		})
	}
	return relations, nil
}

// GetModel returns the model identifier.
func (c *Client) GetModel() string {
	return "gliner-native"
}
