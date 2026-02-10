//go:build cgo

package candle

import (
	"context"
	"fmt"
	"sync"

	gocandle "github.com/soundprediction/go-candle/pkg/candle"
	"github.com/soundprediction/predicato/pkg/gliner"
	"github.com/soundprediction/predicato/pkg/gliner2"
	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/types"
)

// CandleNLPConfig configures the candle NLP client.
type CandleNLPConfig struct {
	// T5ModelID for summarization (e.g. "google/flan-t5-base").
	T5ModelID string `json:"t5_model_id,omitempty"`
	// TextGenModelID for text generation (e.g. "HuggingFaceTB/SmolLM2-360M-Instruct").
	TextGenModelID string `json:"text_gen_model_id,omitempty"`
	// TranslationPair for Marian MT (e.g. "en-fr").
	TranslationPair string `json:"translation_pair,omitempty"`
	// GLiNERModelID for NER (e.g. "onnx-community/gliner_small-v2.1").
	GLiNERModelID string `json:"gliner_model_id,omitempty"`
	// GLiNER2Config for extended extraction. Optional.
	GLiNER2Config *gliner2.Config `json:"gliner2_config,omitempty"`
	// CacheDir overrides the default HF cache directory.
	CacheDir string `json:"cache_dir,omitempty"`
}

// Client implements nlp.Client by composing go-candle pipelines with GLiNER and GLiNER2.
// All go-candle models are auto-downloaded from HuggingFace Hub on first use.
type Client struct {
	t5Pipeline      *gocandle.T5Pipeline
	textGenPipeline *gocandle.TextGenerationPipeline
	translPipeline  *gocandle.TranslationPipeline
	glinerClient    *gliner.Client
	gliner2Client   *gliner2.Client
	config          CandleNLPConfig
	mu              sync.Mutex
}

// NewClient creates a new candle NLP client. Pipelines are lazily initialized on first use.
func NewClient(cfg CandleNLPConfig) *Client {
	return &Client{config: cfg}
}

func (c *Client) loadT5() error {
	if c.t5Pipeline != nil {
		return nil
	}
	if c.config.T5ModelID == "" {
		return fmt.Errorf("T5 model ID not configured")
	}
	p, err := gocandle.NewT5Pipeline(gocandle.T5Config{
		ModelID:  c.config.T5ModelID,
		CacheDir: c.config.CacheDir,
	})
	if err != nil {
		return fmt.Errorf("failed to create T5 pipeline: %w", err)
	}
	c.t5Pipeline = p
	return nil
}

func (c *Client) loadTextGen() error {
	if c.textGenPipeline != nil {
		return nil
	}
	if c.config.TextGenModelID == "" {
		return fmt.Errorf("text generation model ID not configured")
	}
	p, err := gocandle.NewTextGenerationPipeline(gocandle.TextGenerationConfig{
		ModelID:  c.config.TextGenModelID,
		CacheDir: c.config.CacheDir,
	})
	if err != nil {
		return fmt.Errorf("failed to create text generation pipeline: %w", err)
	}
	c.textGenPipeline = p
	return nil
}

func (c *Client) loadTranslation() error {
	if c.translPipeline != nil {
		return nil
	}
	if c.config.TranslationPair == "" {
		return fmt.Errorf("translation language pair not configured")
	}
	p, err := gocandle.NewTranslationPipeline(gocandle.TranslationConfig{
		LanguagePair: c.config.TranslationPair,
		CacheDir:     c.config.CacheDir,
	})
	if err != nil {
		return fmt.Errorf("failed to create translation pipeline: %w", err)
	}
	c.translPipeline = p
	return nil
}

func (c *Client) loadGLiNER() error {
	if c.glinerClient != nil {
		return nil
	}
	if c.config.GLiNERModelID == "" {
		return fmt.Errorf("GLiNER model ID not configured")
	}
	client, err := gliner.NewClient(c.config.GLiNERModelID)
	if err != nil {
		return fmt.Errorf("failed to create GLiNER client: %w", err)
	}
	c.glinerClient = client
	return nil
}

func (c *Client) loadGLiNER2() error {
	if c.gliner2Client != nil {
		return nil
	}
	if c.config.GLiNER2Config == nil {
		return fmt.Errorf("GLiNER2 config not provided")
	}
	client, err := gliner2.NewClient(*c.config.GLiNER2Config)
	if err != nil {
		return fmt.Errorf("failed to create GLiNER2 client: %w", err)
	}
	c.gliner2Client = client
	return nil
}

// Chat is not natively supported; use LLMAdapter for Chat routing.
func (c *Client) Chat(_ context.Context, _ []types.Message) (*types.Response, error) {
	return nil, fmt.Errorf("candle does not support Chat interface; use LLMAdapter for task routing")
}

// ChatWithStructuredOutput is not supported.
func (c *Client) ChatWithStructuredOutput(_ context.Context, _ []types.Message, _ any) (*types.Response, error) {
	return nil, fmt.Errorf("candle does not support ChatWithStructuredOutput")
}

// ExtractEntities delegates to GLiNER v1 for NER.
func (c *Client) ExtractEntities(_ context.Context, text string, entityTypes []string) ([]nlp.ExtractedEntity, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.loadGLiNER(); err != nil {
		return nil, err
	}

	entities, err := c.glinerClient.ExtractEntities(text, entityTypes)
	if err != nil {
		return nil, fmt.Errorf("GLiNER entity extraction failed: %w", err)
	}

	var result []nlp.ExtractedEntity
	for _, e := range entities {
		result = append(result, nlp.ExtractedEntity{
			Text:       e.Text,
			Label:      e.Label,
			Confidence: float64(e.Score),
		})
	}
	return result, nil
}

// ExtractRelations delegates to GLiNER v1 for relation extraction.
func (c *Client) ExtractRelations(_ context.Context, text string, relationTypes []string) ([]nlp.ExtractedRelation, error) {
	// GLiNER v1 relation extraction requires a schema (map[string][2][]string),
	// which the nlp.Client interface doesn't provide directly.
	// Return an error consistent with the existing gliner adapter behavior.
	return nil, fmt.Errorf("ExtractRelations with simple signature not supported by candle client; use GLiNER adapter with schema or GLiNER2")
}

// ExtractExtended delegates to GLiNER2 for structured extraction.
func (c *Client) ExtractExtended(ctx context.Context, text string, entityTypes, relationTypes []string) (*nlp.ExtendedExtractionResult, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.loadGLiNER2(); err != nil {
		return nil, err
	}

	return c.gliner2Client.ExtractExtended(ctx, text, entityTypes, relationTypes)
}

// Summarize uses T5Pipeline with "summarize: " prefix.
func (c *Client) Summarize(_ context.Context, text string) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.loadT5(); err != nil {
		return "", err
	}

	return c.t5Pipeline.Generate("summarize: " + text)
}

// GenerateText uses TextGenerationPipeline.
func (c *Client) GenerateText(_ context.Context, prompt string) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.loadTextGen(); err != nil {
		return "", err
	}

	return c.textGenPipeline.Generate(prompt)
}

// GetModel returns the primary model identifier.
func (c *Client) GetModel() string {
	if c.config.GLiNERModelID != "" {
		return "candle(" + c.config.GLiNERModelID + ")"
	}
	if c.config.T5ModelID != "" {
		return "candle(" + c.config.T5ModelID + ")"
	}
	if c.config.TextGenModelID != "" {
		return "candle(" + c.config.TextGenModelID + ")"
	}
	return "candle"
}

// GetCapabilities returns the union of capabilities from all configured sub-providers.
func (c *Client) GetCapabilities() []nlp.TaskCapability {
	var caps []nlp.TaskCapability

	if c.config.GLiNERModelID != "" {
		caps = append(caps, nlp.TaskNamedEntityRecognition, nlp.TaskRelationExtraction)
	}
	if c.config.GLiNER2Config != nil {
		caps = append(caps, nlp.TaskExtendedExtraction)
	}
	if c.config.T5ModelID != "" {
		caps = append(caps, nlp.TaskSummarization)
	}
	if c.config.TextGenModelID != "" {
		caps = append(caps, nlp.TaskTextGeneration)
	}
	if c.config.TranslationPair != "" {
		caps = append(caps, nlp.TaskTranslation)
	}
	return caps
}

// Close cleans up all pipelines and sub-clients.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.t5Pipeline != nil {
		c.t5Pipeline.Close()
		c.t5Pipeline = nil
	}
	if c.textGenPipeline != nil {
		c.textGenPipeline.Close()
		c.textGenPipeline = nil
	}
	if c.translPipeline != nil {
		c.translPipeline.Close()
		c.translPipeline = nil
	}
	if c.glinerClient != nil {
		c.glinerClient.Close()
		c.glinerClient = nil
	}
	if c.gliner2Client != nil {
		c.gliner2Client.Close()
		c.gliner2Client = nil
	}
	return nil
}
