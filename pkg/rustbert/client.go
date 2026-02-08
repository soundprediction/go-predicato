package rustbert

import (
	"context"
	"fmt"
	"sync"

	"github.com/soundprediction/go-rust-bert/pkg/rustbert"
	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/types"
)

// Client wraps go-rust-bert models for use in Predicato.
type Client struct {
	nerModel           *rustbert.NERModel
	summarizationModel *rustbert.SummarizationModel
	qaModel            *rustbert.QAModel
	textGenModel       *rustbert.TextGenerationModel
	config             Config
	mu                 sync.Mutex
}

// Config holds configuration for RustBert models
type Config struct {
	NERModelID           string
	SummarizationModelID string
}

// NewClient creates a new RustBert client.
func NewClient(cfg Config) *Client {
	return &Client{
		config: cfg,
	}
}

// LoadNERModel loads the NER model.
// If modelID is empty, it uses the default (BERT-based).
// If modelID is set, it downloads artifacts and loads from files.
func (c *Client) LoadNERModel() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.nerModel != nil {
		return nil
	}

	if c.config.NERModelID != "" {
		fmt.Printf("Loading custom NER model: %s\n", c.config.NERModelID)
		modelPath, configPath, vocabPath, mergesPath, err := rustbert.DownloadArtifacts(c.config.NERModelID, "")
		if err != nil {
			return fmt.Errorf("failed to download artifacts for %s: %w", c.config.NERModelID, err)
		}

		// Assuming BERT for now as default custom model type if not specified
		// TODO: Add ModelType to Config
		m, err := rustbert.NewNERModelFromFiles(modelPath, configPath, vocabPath, mergesPath, rustbert.ModelTypeBert)
		if err != nil {
			return fmt.Errorf("failed to create custom NER model: %w", err)
		}
		c.nerModel = m
		return nil
	}

	// Using default BERT NER model
	m, err := rustbert.NewNERModel()
	if err != nil {
		return fmt.Errorf("failed to create NER model: %w", err)
	}
	c.nerModel = m
	return nil
}

// LoadSummarizationModel loads the Summarization model (BART/DistilBART default).
func (c *Client) LoadSummarizationModel() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.summarizationModel != nil {
		return nil
	}

	m, err := rustbert.NewSummarizationModel()
	if err != nil {
		return fmt.Errorf("failed to create Summarization model: %w", err)
	}
	c.summarizationModel = m
	return nil
}

// Close closes all loaded models.
func (c *Client) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.nerModel != nil {
		c.nerModel.Close()
		c.nerModel = nil
	}
	if c.summarizationModel != nil {
		c.summarizationModel.Close()
		c.summarizationModel = nil
	}
}

// Entity represents an extracted entity
type Entity struct {
	Text  string
	Label string
	Score float64
}

// ExtractEntities extracts named entities from the text using the NER model.
func (c *Client) ExtractEntities(ctx context.Context, text string, entityTypes []string) ([]nlp.ExtractedEntity, error) {
	// Load on first use if not loaded
	if c.nerModel == nil {
		if err := c.LoadNERModel(); err != nil {
			return nil, err
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	results, err := c.nerModel.Predict(text)
	if err != nil {
		return nil, fmt.Errorf("NER prediction failed: %w", err)
	}

	var entities []nlp.ExtractedEntity
	for _, r := range results {
		// Filter by entityTypes if provided
		if len(entityTypes) > 0 {
			found := false
			for _, t := range entityTypes {
				if r.Label == t {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}

		entities = append(entities, nlp.ExtractedEntity{
			Text:       r.Word,
			Label:      r.Label,
			Confidence: r.Score,
		})
	}
	return entities, nil
}

// ExtractRelations is not supported by RustBert currently
func (c *Client) ExtractRelations(ctx context.Context, text string, relationTypes []string) ([]nlp.ExtractedRelation, error) {
	return nil, fmt.Errorf("RustBert does not support relation extraction")
}

// Summarize generates a summary of the text.
func (c *Client) Summarize(ctx context.Context, text string) (string, error) {
	if c.summarizationModel == nil {
		if err := c.LoadSummarizationModel(); err != nil {
			return "", err
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Summarize returns a list of summaries (usually just 1 for single input)
	results, err := c.summarizationModel.Summarize(text)
	if err != nil {
		return "", fmt.Errorf("summarization failed: %w", err)
	}

	if len(results) > 0 {
		return results[0], nil
	}
	return "", nil
}

// LoadTextGenerationModel loads the Text Generation model.
func (c *Client) LoadTextGenerationModel() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.textGenModel != nil {
		return nil
	}

	m, err := rustbert.NewTextGenerationModel()
	if err != nil {
		return fmt.Errorf("failed to create Text Generation model: %w", err)
	}
	c.textGenModel = m
	return nil
}

// GenerateText generates text from a prompt.
func (c *Client) GenerateText(ctx context.Context, prompt string) (string, error) {
	if c.textGenModel == nil {
		if err := c.LoadTextGenerationModel(); err != nil {
			return "", err
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	result, err := c.textGenModel.Generate(prompt, "")
	if err != nil {
		return "", fmt.Errorf("text generation failed: %w", err)
	}

	return result, nil
}

// Chat implements the Client interface (not supported by raw RustBert models usually)
func (c *Client) Chat(ctx context.Context, messages []types.Message) (*types.Response, error) {
	return nil, fmt.Errorf("RustBert does not support Chat interface")
}

// ChatWithStructuredOutput implements the Client interface
func (c *Client) ChatWithStructuredOutput(ctx context.Context, messages []types.Message, schema any) (*types.Response, error) {
	return nil, fmt.Errorf("RustBert does not support ChatWithStructuredOutput")
}

// GetModel returns the model identifier
func (c *Client) GetModel() string {
	if c.config.NERModelID != "" {
		return c.config.NERModelID
	}
	return "rustbert-default"
}

// GetCapabilities returns the list of capabilities supported by this client.
func (c *Client) GetCapabilities() []nlp.TaskCapability {
	return []nlp.TaskCapability{
		nlp.TaskNamedEntityRecognition,
		nlp.TaskSummarization,
		nlp.TaskTextGeneration,
	}
}

// ExtractExtended performs structured extraction (entities, relations, triples, rules) from the text.
func (c *Client) ExtractExtended(ctx context.Context, text string, entityTypes, relationTypes []string) (*nlp.ExtendedExtractionResult, error) {
	return nil, fmt.Errorf("RustBert does not support extended extraction")
}
