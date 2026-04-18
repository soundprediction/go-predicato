# Local Inference Plan: Eliminating LLM Dependencies

## Goal

Provide batteries-included local inference for every ML task in predicato. Users set a mode in config and everything works — models auto-download from HuggingFace Hub on first use. No API keys, no token fees, no external services.

## Current State: LLM Dependency Audit

### Tasks That Currently Require an LLM

| # | Task | Priority | Call Sites | What the LLM Does |
|---|------|----------|------------|-------------------|
| 1 | Entity Extraction | Critical | `node_operations.go:114`, `bulk_utils.go:342` | Extract named entities from text with types |
| 2 | Relationship Extraction | Critical | `edge_operations.go:152`, `bulk_utils.go:379` | Extract subject-predicate-object triples |
| 3 | Extended Extraction | Critical | `ingestion_factstore.go:303` | Extract triples with context (condition, temporal, certainty, scope) + conditional rules |
| 4 | Entity Resolution | Critical | `bulk_utils.go:727`, modeler pipeline | "Are entity A and entity B the same thing?" → yes/no |
| 5 | Community Summarization | Medium | `community.go:305` | Merge two entity summaries into one coherent summary |
| 6 | Community Naming | Low | `community.go:330` | Generate a 1-5 word descriptive name from a summary |
| 7 | Content Relevance Filtering | Low | `nlp_utils.go:36,78` | "Is this content relevant to topic X?" → yes/no |

### Tasks Already Covered Locally

| Task | Local Solution | Package |
|------|---------------|---------|
| Text Embeddings | Qwen3-Embedding-0.6B via go-candle / EmbedEverything | `pkg/candle/embedder.go`, `pkg/embedder/embed_everything.go` |
| Reranking | TF-IDF cosine, embedding cosine, Qwen3-Reranker via EmbedEverything | `pkg/crossencoder/local.go`, `pkg/crossencoder/embedding.go` |
| NLI / Verification | DeBERTa cross-encoder via go-candle | `pkg/verification/nli.go` |
| BM25 Search | Database-level full-text search | `pkg/search/search.go` |
| Graph Traversal (BFS) | Pure Go algorithm | `pkg/search/search.go` |
| RRF, MMR | Pure Go algorithms | `pkg/search/rerankers.go` |
| Edge Quality Filtering | Heuristic tokenization + stopwords | `pkg/types/edge_quality.go` |
| Summarization (basic) | T5/Flan-T5 via go-candle | `pkg/candle/client.go` |
| Text Generation (basic) | SmolLM2/etc via go-candle | `pkg/candle/client.go` |

## Three Operating Modes

```toml
[predicato]
# Choose one:
inference_mode = "specialized"   # Purpose-built small models per task (no LLM)
inference_mode = "lora"          # Small base LLM + LoRA adapters per task
inference_mode = "llm"           # Full LLM (current behavior, API or local)
```

---

## Mode 1: Specialized Models

Each task gets a purpose-built pre-trained model. No training required — all models are publicly available on HuggingFace Hub and auto-download on first use.

### Model Selection

| Task | Model | HF ID | Size | Architecture | Inference Via |
|------|-------|-------|------|-------------|--------------|
| Entity Extraction | GLiNER2 | `fastino/gliner2-multi-v1` | ~300MB | Transformer span classifier | go-gline-rs (native FFI) |
| Relationship Extraction | GLiNER2 | `fastino/gliner2-multi-v1` | ~300MB | Same model, relation mode | go-gline-rs (native FFI) |
| Extended Extraction | GLiNER2 | `fastino/gliner2-multi-v1` | ~300MB | Same model, extended mode | go-gline-rs (native FFI) |
| Entity Resolution | STS Cross-Encoder | `cross-encoder/stsb-distilroberta-base` | ~80MB | DeBERTa/RoBERTa regression | go-candle `SeqClassificationPipeline` |
| Community Summarization | Flan-T5 | `google/flan-t5-base` | ~250MB | T5 seq2seq | go-candle `T5Pipeline` (already exists) |
| Community Naming | Flan-T5 | `google/flan-t5-small` | ~80MB | T5 seq2seq | go-candle `T5Pipeline` (already exists) |
| Content Relevance | Relevance Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~80MB | MiniLM classification | go-candle `SeqClassificationPipeline` |

**Total additional model weight**: ~870MB (GLiNER2 shared across 3 tasks, T5 shared across 2)

### Implementation Work

#### 1. Finish GLiNER2 Native Binding (tasks 1-3)

The `pkg/gliner2/native_client.go` stub exists. Need to complete the go-gline-rs FFI binding so GLiNER2 runs in-process without an HTTP server.

**Files to create/modify:**
- `go-gline-rs/` — Rust FFI binding for GLiNER2 (similar pattern to go-candle)
- `predicato/pkg/gliner2/native_client.go` — complete the stub
- `predicato/pkg/candle/client.go` — wire native GLiNER2 as default when no HTTP endpoint configured

**Eliminates:** HTTP server dependency for entity/relation/extended extraction.

#### 2. Entity Resolution via Cross-Encoder (task 4)

Replace the LLM "are these the same entity?" call with a cross-encoder similarity score.

**Approach:**
- Load `cross-encoder/stsb-distilroberta-base` via `SeqClassificationPipeline`
- Input: `"entity A description [SEP] entity B description"`
- Output: similarity score (0.0–1.0)
- If score > configurable threshold (default 0.85), entities are the same

**Files to create:**
- `predicato/pkg/modeler/cross_encoder_resolver.go` — new entity resolver using cross-encoder
- Modify `predicato/pkg/modeler/default.go` — use cross-encoder resolver when no LLM configured

**Note:** The `SeqClassificationPipeline` we built supports DeBERTa. The STS model outputs 1 logit (regression) instead of 3 (classification). Need to handle both in the Go wrapper — if `num_classes == 1`, treat as regression score.

#### 3. Community Summarization + Naming via T5 (tasks 5-6)

The T5 pipeline already exists and works. Just need to wire it as the fallback when no LLM summarizer is configured.

**Files to modify:**
- `predicato/community/community.go` — in `summarizePair()`, fall back to T5 with prompt `"summarize: Summary 1: {a}. Summary 2: {b}."` when no Chat-capable NLP client
- `predicato/community/community.go` — in `generateCommunityName()`, fall back to T5 with prompt `"title: {summary}"`

#### 4. Content Relevance via Cross-Encoder (task 7)

Replace the LLM "is this relevant?" call with a relevance cross-encoder.

**Approach:**
- Load `cross-encoder/ms-marco-MiniLM-L-6-v2` via `SeqClassificationPipeline`
- Input: `"query [SEP] document"`
- Output: relevance score (0.0–1.0)
- If score > threshold (default 0.5), content is relevant

**Files to modify:**
- `predicato/nlp_utils.go` — add cross-encoder fallback in `AnalyzeContentRelevance()`

### go-candle Changes Needed

The `SeqClassificationPipeline` currently uses DeBERTa (3-class NLI). Need to also support:
- **STS regression models** (1 output logit) — for entity resolution
- **MiniLM cross-encoder** (1 output logit) — for relevance scoring
- **BERT-based models** — the sequence classification Rust code currently only loads DeBERTa; need to add BERT/RoBERTa/MiniLM model detection

**File:** `go-candle/candle_binding/src/sequence_classification.rs`
- Add model type detection from `config.json` (`model_type` field)
- Support `bert`, `roberta`, `distilroberta`, `minilm` alongside `deberta-v2`
- Handle 1-logit regression output (no softmax, just sigmoid or raw score)

---

## Mode 2: Small LLM + LoRA Adapters

One base model with task-specific LoRA adapters that get swapped per pipeline step. All local, all auto-downloaded.

### Base Model Options

| Model | Params | GGUF Q4 Size | Strengths | License |
|-------|--------|-------------|-----------|---------|
| `Qwen/Qwen3-4B` | 4B | ~2.5GB | Best quality/param ratio, strong structured output, multilingual | Apache 2.0 |
| `google/gemma-3-4b-it` | 4B | ~2.5GB | Strong instruction following, good at classification | Gemma license |
| `ibm-granite/granite-3.3-2b-instruct` | 2B | ~1.5GB | Smallest viable, enterprise-optimized, excellent license | Apache 2.0 |

**Recommendation:** Start with Granite 3.3-2B for minimum resource usage, upgrade to Qwen3-4B if quality is insufficient.

### LoRA Adapters

| Adapter | Task | Training Data Source | Output Format |
|---------|------|---------------------|---------------|
| `lora-ner` | Entity Extraction | CoNLL-2003, BC5CDR, NCBI Disease + distilled from high-quality LLM runs | `[{name, type}]` JSON |
| `lora-re` | Relationship Extraction | DocRED, SciERC, ChemProt + distilled | `[{subject, predicate, object, subject_type, object_type}]` JSON |
| `lora-ext` | Extended Extraction | Distilled from predicato's own pipeline using GPT-4o/Claude | Extended triple JSON with context fields |
| `lora-dedup` | Entity Resolution | Generated pairs from predicato entity store (positive + hard negatives) | `"YES"` / `"NO"` |
| `lora-summ` | Community Summarization | Multi-News, WCEP + generated from predicato communities | Merged summary text |
| `lora-title` | Community Naming | PubMed/arXiv title generation | 1-5 word title |
| `lora-rel` | Content Relevance | MS MARCO relevance judgments | `"YES"` / `"NO"` |

### LoRA Infrastructure

#### go-candle: Adapter Loading

The text generation pipeline needs runtime LoRA adapter loading:

```rust
// New FFI functions needed in go-candle:
fn load_lora_adapter(wrapper: *mut TextGenerationPipelineWrapper,
                     adapter_path: *const c_char,
                     adapter_name: *const c_char) -> c_int;

fn set_active_adapter(wrapper: *mut TextGenerationPipelineWrapper,
                      adapter_name: *const c_char) -> c_int;

fn unload_adapter(wrapper: *mut TextGenerationPipelineWrapper,
                  adapter_name: *const c_char) -> c_int;
```

Go side:
```go
pipeline.LoadAdapter("/models/adapters/lora-ner", "ner")
pipeline.SetActiveAdapter("ner")
result := pipeline.Generate("Extract entities: ...")
pipeline.SetActiveAdapter("re")
result = pipeline.Generate("Extract relations: ...")
```

**Rust implementation:** Use `candle-lora` crate or manual weight injection. LoRA adds low-rank matrices to attention layers — the adapter files are small (5-50MB safetensors).

#### Training Pipeline (Python, offline)

Located in `modeling/lora_training/`:

```
modeling/lora_training/
├── train_adapter.py          # Unified training script (uses unsloth/peft + trl)
├── generate_training_data.py # Distillation: run high-quality LLM, capture I/O pairs
├── evaluate_adapter.py       # Compare adapter vs full LLM on held-out test set
├── configs/
│   ├── ner.yaml
│   ├── re.yaml
│   ├── ext.yaml
│   ├── dedup.yaml
│   ├── summ.yaml
│   ├── title.yaml
│   └── rel.yaml
└── README.md
```

**Distillation workflow:**
1. Run predicato ingestion with a high-quality LLM (Claude, GPT-4o) on your medical corpus
2. Capture every `(input_text, task_prompt, output)` tuple — predicato already logs these
3. Filter for high-confidence outputs (extraction confidence > 0.8)
4. Fine-tune LoRA adapters using `unsloth` (4-bit QLoRA, ~30min per adapter on 1 GPU)
5. Export adapters to safetensors
6. Evaluate: adapter output vs. LLM output on held-out set

### predicato Integration

```go
// pkg/nlp/lora_client.go
type LoRAClient struct {
    pipeline    *gocandle.TextGenerationPipeline
    adapters    map[string]string  // task -> adapter name
    mu          sync.Mutex
}

func (c *LoRAClient) ExtractEntities(ctx context.Context, text string, types []string) ([]ExtractedEntity, error) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.pipeline.SetActiveAdapter(c.adapters["ner"])
    prompt := formatNERPrompt(text, types)
    result, err := c.pipeline.Generate(prompt, generateOpts)
    return parseEntities(result)
}

func (c *LoRAClient) ExtractRelations(ctx context.Context, text string, types []string) ([]ExtractedRelation, error) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.pipeline.SetActiveAdapter(c.adapters["re"])
    // ...
}
```

---

## Configuration

```toml
[predicato]
# Mode selection — determines how ML tasks are handled:
#   "specialized" — purpose-built small models per task (no LLM, ~870MB total)
#   "lora"        — small base LLM + LoRA adapters per task (~2-3GB total)
#   "llm"         — full LLM via API or local server (current behavior)
inference_mode = "specialized"

# --- Specialized mode settings ---
[predicato.specialized_models]
# Models auto-download from HF Hub on first use.
# Override defaults only if you want different models.
# ner_model = "fastino/gliner2-multi-v1"                    # Entity + relation + extended extraction
# entity_resolver_model = "cross-encoder/stsb-distilroberta-base"  # Entity dedup
# summarizer_model = "google/flan-t5-base"                  # Community summarization
# relevance_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Content relevance scoring
# cache_dir = ""                                            # HF cache directory

# --- LoRA mode settings ---
# [predicato.lora]
# base_model = "ibm-granite/granite-3.3-2b-instruct"
# adapter_dir = "/models/adapters/"                # Directory containing adapter subdirs
# Adapters auto-discovered from adapter_dir. Override mapping:
# ner_adapter = "lora-ner"
# re_adapter = "lora-re"
# ext_adapter = "lora-ext"
# dedup_adapter = "lora-dedup"
# summ_adapter = "lora-summ"
# title_adapter = "lora-title"
# rel_adapter = "lora-rel"
```

---

## Implementation Roadmap

### Phase 1: Specialized Models (Config 1) — No Training Required

| Step | Work | Effort | Unblocks |
|------|------|--------|----------|
| 1a | Add BERT/RoBERTa/MiniLM support to go-candle `sequence_classification.rs` | 1 day | Entity resolution + relevance |
| 1b | Handle 1-logit regression output in seq classification pipeline | 0.5 day | STS models |
| 2 | Build cross-encoder entity resolver in `pkg/modeler/` | 2 days | LLM-free entity resolution |
| 3 | Wire T5 fallback for community summarization + naming | 1 day | LLM-free communities |
| 4 | Wire cross-encoder fallback for content relevance | 1 day | LLM-free relevance filtering |
| 5 | Finish GLiNER2 native FFI binding (go-gline-rs) | 3-5 days | LLM-free extraction without HTTP |
| 6 | Add `inference_mode` config + factory wiring | 1 day | End-to-end config |
| 7 | Integration tests: full ingestion pipeline with no LLM | 1 day | Validation |

**Total: ~10-12 days**

### Phase 2: LoRA Adapters (Config 2) — Requires Training

| Step | Work | Effort | Unblocks |
|------|------|--------|----------|
| 8 | Add LoRA adapter loading to go-candle text generation | 3-5 days | Adapter runtime |
| 9 | Build distillation pipeline in `modeling/lora_training/` | 2 days | Training data |
| 10 | Train + evaluate 7 adapters | 1-2 weeks | Adapter quality |
| 11 | Build `pkg/nlp/lora_client.go` with adapter routing | 2 days | Integration |
| 12 | Add LoRA config + factory wiring | 1 day | End-to-end config |

**Total: ~3-4 weeks**

### Phase 3: Polish

| Step | Work | Effort |
|------|------|--------|
| 13 | Benchmark: specialized vs LoRA vs full LLM (quality + latency + memory) | 2-3 days |
| 14 | Documentation + examples for each mode | 1 day |
| 15 | CI: test all three modes in pipeline | 1 day |

---

## Resource Requirements

| Mode | Total Model Weight | RAM (inference) | GPU | First-run Download |
|------|-------------------|----------------|-----|-------------------|
| Specialized | ~870MB | ~2GB | None (CPU) | ~870MB from HF Hub |
| LoRA (Granite 2B) | ~1.5GB base + ~200MB adapters | ~3GB | None (CPU, slower) / 1 GPU (fast) | ~1.7GB from HF Hub |
| LoRA (Qwen3 4B) | ~2.5GB base + ~200MB adapters | ~5GB | None (CPU, slow) / 1 GPU (fast) | ~2.7GB from HF Hub |
| Full LLM | Varies | Varies | Depends | N/A (API) |

## Open Questions

1. **GLiNER2 native binding priority** — this is the largest single piece of work. The HTTP client works today; native just eliminates the sidecar process. Should we defer this and ship Config 1 with GLiNER2 HTTP first?

2. **LoRA base model choice** — Granite 3.3-2B is smallest but may not produce good enough structured output for extraction. Need to benchmark before committing.

3. **Adapter hosting** — should adapters be hosted on our own HF org (`soundprediction/predicato-lora-ner`) for auto-download, or bundled in the repo?

4. **Medical domain adapters** — the general adapters may not be good enough for medical NER/RE. May need domain-specific training data from StatPearls, PubMed, etc.
