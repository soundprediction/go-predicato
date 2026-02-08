package nlp

import "slices"

// TaskCapability represents a specific NLP task that a model can perform.
type TaskCapability string

const (
	// TaskEmbedding represents text embedding generation.
	TaskEmbedding TaskCapability = "embedding"
	// TaskNamedEntityRecognition represents named entity recognition (NER).
	TaskNamedEntityRecognition TaskCapability = "ner"
	// TaskRelationExtraction represents relation extraction between entities.
	TaskRelationExtraction TaskCapability = "relation_extraction"
	// TaskSummarization represents text summarization.
	TaskSummarization TaskCapability = "summarization"
	// TaskQuestionAnswering represents extractive question answering.
	TaskQuestionAnswering TaskCapability = "question_answering"
	// TaskTextGeneration represents open-ended text generation (chat/completion).
	TaskTextGeneration TaskCapability = "text_generation"
	// TaskTranslation represents text translation.
	TaskTranslation TaskCapability = "translation"
	// TaskExtendedExtraction represents structured extraction of extended triples and rules.
	TaskExtendedExtraction TaskCapability = "extended_extraction"
)

// ProviderID represents a unique identifier for an AI provider.
type ProviderID string

const (
	// ProviderEmbedEverything is the ID for the EmbedEverything local provider.
	ProviderEmbedEverything ProviderID = "embedeverything"
	// ProviderGLiNER is the ID for the GLiNER local provider.
	ProviderGLiNER ProviderID = "gliner"
	// ProviderGLiNER2 is the ID for GLiNER2 provider.
	ProviderGLiNER2 ProviderID = "gliner2"
	// ProviderRustBert is the ID for RustBert local provider.
	ProviderRustBert ProviderID = "rustbert"
	// ProviderOpenAI is the ID for OpenAI.
	ProviderOpenAI ProviderID = "openai"
	// ProviderAnthropic is the ID for Anthropic.
	ProviderAnthropic ProviderID = "anthropic"
	// ProviderGoogle is the ID for Google (Gemini).
	ProviderGoogle ProviderID = "google"
	// ProviderAzure is the ID for Azure OpenAI.
	ProviderAzure ProviderID = "azure"
	// ProviderOpenAICompatible is the ID for generic OpenAI-compatible providers.
	ProviderOpenAICompatible ProviderID = "openai_compatible"
)

// Provider represents an AI model provider.
type Provider struct {
	ID          ProviderID
	Name        string
	Description string
	IsLocal     bool
}

// Model represents a specific AI model.
type Model struct {
	ID          string
	Name        string
	ProviderID  ProviderID
	Description string
	// Family is an optional grouping identifier (e.g., "gpt-4", "llama-3")
	Family       string
	Capabilities []TaskCapability
}

// BuiltInProviders contains the standard set of supported providers.
var BuiltInProviders = map[ProviderID]Provider{
	ProviderEmbedEverything: {
		ID:          ProviderEmbedEverything,
		Name:        "EmbedEverything",
		Description: "Local generic embedding models via Rust bindings",
		IsLocal:     true,
	},
	ProviderGLiNER: {
		ID:          ProviderGLiNER,
		Name:        "GLiNER",
		Description: "Local GLiNER models for zero-shot NER and Relation Extraction",
		IsLocal:     true,
	},
	ProviderGLiNER2: {
		ID:          ProviderGLiNER2,
		Name:        "GLiNER2",
		Description: "GLiNER2 models for entity extraction, fact extraction, text classification, and structured data extraction",
		IsLocal:     false, // Can be local API or remote Fastino
	},
	ProviderRustBert: {
		ID:          ProviderRustBert,
		Name:        "RustBert",
		Description: "Rust-based BERT models for various NLP tasks via bindings",
		IsLocal:     true,
	},
	ProviderOpenAI: {
		ID:          ProviderOpenAI,
		Name:        "OpenAI",
		Description: "Cloud-based advanced LLMs (GPT-4, etc.)",
		IsLocal:     false,
	},
	ProviderAnthropic: {
		ID:          ProviderAnthropic,
		Name:        "Anthropic",
		Description: "Cloud-based advanced LLMs (Claude, etc.)",
		IsLocal:     false,
	},
	ProviderGoogle: {
		ID:          ProviderGoogle,
		Name:        "Google",
		Description: "Cloud-based advanced LLMs (Gemini)",
		IsLocal:     false,
	},
	ProviderAzure: {
		ID:          ProviderAzure,
		Name:        "Azure OpenAI",
		Description: "Enterprise-grade OpenAI models hosting",
		IsLocal:     false,
	},
	ProviderOpenAICompatible: {
		ID:          ProviderOpenAICompatible,
		Name:        "OpenAI Compatible",
		Description: "Generic provider compatible with OpenAI API (e.g. vLLM, Ollama)",
		IsLocal:     false, // Can be local or remote, but treating as generic API
	},
}

// BuiltInModels contains a curated list of built-in models.
var BuiltInModels = []Model{
	// --- EmbedEverything ---
	{
		ID:           "sentence-transformers/all-MiniLM-L6-v2",
		Name:         "all-MiniLM-L6-v2",
		ProviderID:   ProviderEmbedEverything,
		Capabilities: []TaskCapability{TaskEmbedding},
		Description:  "Fast and effective general-purpose sentence embedding model",
	},
	{
		ID:           "sentence-transformers/all-mpnet-base-v2",
		Name:         "all-mpnet-base-v2",
		ProviderID:   ProviderEmbedEverything,
		Capabilities: []TaskCapability{TaskEmbedding},
		Description:  "Higher quality, slightly slower general-purpose sentence embedding model",
	},

	// --- GLiNER ---
	{
		ID:           "urchade/gliner_multi-v2.1",
		Name:         "GLiNER Multi v2.1",
		ProviderID:   ProviderGLiNER,
		Capabilities: []TaskCapability{TaskNamedEntityRecognition, TaskRelationExtraction},
		Description:  "Multilingual GLiNER model for zero-shot NER and Relation Extraction",
	},
	{
		ID:           "urchade/gliner_base",
		Name:         "GLiNER Base",
		ProviderID:   ProviderGLiNER,
		Capabilities: []TaskCapability{TaskNamedEntityRecognition},
		Description:  "Base English GLiNER model for zero-shot NER",
	},
	{
		ID:           "urchade/gliner_small-v2.1",
		Name:         "GLiNER Small v2.1",
		ProviderID:   ProviderGLiNER,
		Capabilities: []TaskCapability{TaskNamedEntityRecognition},
		Description:  "Lightweight multilingual GLiNER model",
	},

	// --- GLiNER2 ---
	{
		ID:           "fastino/gliner2-base-v1",
		Name:         "GLiNER2 Base v1",
		ProviderID:   ProviderGLiNER2,
		Capabilities: []TaskCapability{TaskNamedEntityRecognition, TaskRelationExtraction, TaskExtendedExtraction},
		Description:  "Fastino GLiNER2 base model for entity and relation extraction",
		Family:       "gliner2",
	},
	{
		ID:           "fastino/gliner2-large-v1",
		Name:         "GLiNER2 Large v1",
		ProviderID:   ProviderGLiNER2,
		Capabilities: []TaskCapability{TaskNamedEntityRecognition, TaskRelationExtraction, TaskExtendedExtraction},
		Description:  "Fastino GLiNER2 large model for enhanced entity and relation extraction",
		Family:       "gliner2",
	},

	// --- RustBert ---
	// Default models often used by rust-bert if not specified, or explicit ones.
	// Note: go-rust-bert wrappers might abstract exact IDs, but these are for reference in registry.
	{
		ID:           "bert-base-ner", // Conceptual ID for the default BERT NER
		Name:         "BERT NER",
		ProviderID:   ProviderRustBert,
		Capabilities: []TaskCapability{TaskNamedEntityRecognition},
		Description:  "Default BERT-based Named Entity Recognition",
	},
	{
		ID:           "distilbart-cnn-12-6", // Conceptual ID for default summarizer
		Name:         "DistilBART Summarization",
		ProviderID:   ProviderRustBert,
		Capabilities: []TaskCapability{TaskSummarization},
		Description:  "Default DistilBART model for summarization",
	},
	{
		ID:           "distilbert-base-cased-distilled-squad", // Conceptual ID for default QA
		Name:         "DistilBERT QA",
		ProviderID:   ProviderRustBert,
		Capabilities: []TaskCapability{TaskQuestionAnswering},
		Description:  "Default DistilBERT model for Question Answering",
	},
	{
		ID:           "gpt2",
		Name:         "GPT-2",
		ProviderID:   ProviderRustBert,
		Capabilities: []TaskCapability{TaskTextGeneration},
		Description:  "Default GPT-2 model for text generation",
	},
}

// HasCapability checks if a client supports a given capability.
func HasCapability(client Client, cap TaskCapability) bool {
	return slices.Contains(client.GetCapabilities(), cap)
}

// GetProvider returns the provider with the given ID.
func GetProvider(id ProviderID) (Provider, bool) {
	p, ok := BuiltInProviders[id]
	return p, ok
}

// GetModel returns the model with the given ID.
func GetModel(id string) (Model, bool) {
	for _, m := range BuiltInModels {
		if m.ID == id {
			return m, true
		}
	}
	return Model{}, false
}

// GetModelsByProvider returns all models for a specific provider.
func GetModelsByProvider(providerID ProviderID) []Model {
	var models []Model
	for _, m := range BuiltInModels {
		if m.ProviderID == providerID {
			models = append(models, m)
		}
	}
	return models
}

// GetModelsByCapability returns all models capable of a specific task.
func GetModelsByCapability(capability TaskCapability) []Model {
	var models []Model
	for _, m := range BuiltInModels {
		if slices.Contains(m.Capabilities, capability) {
			models = append(models, m)
		}
	}
	return models
}
