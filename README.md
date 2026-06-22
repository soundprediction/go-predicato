# Predicato

A temporal knowledge graph framework with a fully local ML stack - no API keys required.

The core service is written in **Go** and provides an HTTP API for building and querying knowledge graphs. It can also be imported into other Go applications as a library. A **Python client library** is also provided for convenient integration with Python applications and data pipelines.

## What Makes Predicato Different

Most agentic memory libraries require external services (language models, vector databases, graph databases). Predicato has implemented embedded alternatives for all components so it can run without any external dependencies.

Predicato is **modular by design** - every component can run locally OR connect to external services. Start with the internal stack for development, then swap in cloud services for production without changing your code.

## Design

Predicato implements a **two-layer architecture** that separates raw fact extraction from graph modeling:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Episodes                                 │
│              (documents, conversations, events)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Entity Extraction                             │
│         (GLiNER for NER, NLP models for relationships)           │
│                              │                                   │
│            ┌─────────────────┴──────────────────┐                │
│            ▼                                    ▼                │
│   Standard Extraction              Extended Extraction           │
│   • Entities (nodes)               • Contextual triples          │
│   • Relationships (triples)        • Conditional rules           │
│   • Embeddings                     • Temporal/spatial context    │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────────┐
│      Fact Store         │     │        GraphModeler              │
│  (PostgreSQL/DoltGres)  │     │    (pluggable interface)         │
│                         │     │                                  │
│  • Extracted nodes      │     │  ┌───────────────────────────┐   │
│  • Knowledge triples    │     │  │  • ResolveEntities        │   │
│  • Conditional rules    │     │  │  • ResolveRelationships   │   │
│  • Source documents     │     │  │  • BuildCommunities       │   │
│  • Vector embeddings    │     │  └───────────────────────────┘   │
│                         │     │            │                     │
│  ExtractOnly=true       │     │            ▼                     │
│  stops here ───────────►│     │  ┌───────────────────────────┐   │
│                         │     │  │    Knowledge Graph        │   │
│  ┌───────────────────┐  │     │  │  (CozoDB/DuckDB/Ladybug)  │   │
│  │   RAG Search      │  │     │  │  • Resolved entities      │   │
│  │ (VectorChord/JSONB)│  │     │  │  • Temporal relationships │   │
│  └───────────────────┘  │     │  │  • Communities            │   │
│                         │     │  │                           │   │
└─────────────────────────┘     │  └───────────────────────────┘   │
                                └─────────────────────────────────┘
```

### Why Two Layers?

Entity extraction is the expensive step — it requires LLM calls and embedding generation for every chunk of every document. By persisting the raw extraction results in the Fact Store, this work is done once and never repeated. The graph can then be built, torn down, and rebuilt with different parameters without re-extracting.

**Fact Store (Layer 1)** - Stores raw extractions exactly as they were found:
- **Extracted nodes** (entities) with types, descriptions, and embeddings
- **Knowledge triples** (subject-predicate-object) with contextual fields: condition, temporal, location, certainty, scope, source attribution
- **Conditional rules** (IF-THEN-UNLESS patterns) from extended extraction
- Preserves source provenance (which document, which chunk, which model)
- Uses PostgreSQL/VectorChord for production-grade hybrid search

**Knowledge Graph (Layer 2)** - Stores resolved, interconnected knowledge:
- Entity resolution merges duplicates ("Bob Smith" = "Robert Smith")
- Temporal modeling tracks when facts were valid vs. when recorded
- Community detection groups related entities
- Graph traversal finds multi-hop relationships

This separation enables:
1. **Multiple graph views** - Generate different graph representations from the same extracted facts (different resolution thresholds, entity type filters, or custom `GraphModeler` implementations) without re-running extraction
2. **Incremental updates** - Re-process only changed documents
3. **Simpler RAG** - Use `SearchFacts()` when you don't need graph features
4. **Audit trail** - Track exactly what was extracted from each source

### Bi-Temporal Model

Every fact in Predicato has two time dimensions:

| Dimension | Field | Meaning |
|-----------|-------|---------|
| **Transaction Time** | `created_at` | When the fact was recorded in the system |
| **Valid Time** | `valid_from`, `valid_to` | When the fact was true in the real world |

This enables queries like:
- "What did we know about X as of last Tuesday?" (transaction time)
- "What was true about X during Q3 2024?" (valid time)
- "Show me facts that were recorded wrong and later corrected" (both)

### Pipeline Architecture

Predicato supports two ingestion modes:

**End-to-End Pipeline** (default):
```
Episode -> Extract (nodes + triples) -> Resolve -> Graph
```

**Decoupled Pipeline** (with FactStore):
```
Episode -> Extract -> FactStore    (ExtractOnly=true)
                         |
                         v         (nodes, triples, rules persisted)
         FactStore -> GraphModeler -> Graph
```

**Extended Pipeline** (with contextual enrichment):
```
Episode -> Extract -> Extended Extraction -> FactStore
                      (context + rules)        |
                                               v
                            FactStore -> GraphModeler -> Graph
```

The decoupled mode enables:
- **Custom graph modeling**: Implement `GraphModeler` interface to customize entity resolution, relationship handling, and community detection
- **Batch processing**: Extract facts in bulk, then promote to graph on schedule
- **Re-processing**: Re-model the same facts with different parameters
- **Validation**: Test custom modelers before production use

### Entity Resolution

When adding episodes, Predicato automatically:
1. Extracts entities using GLiNER, GLInER2 (API), or NLP model prompts
2. Generates embeddings for each entity
3. Compares against existing entities (cosine similarity)
4. Merges duplicates above a threshold (default: 0.85)
5. Creates temporal edges between resolved entities

### Knowledge Triples & Extended Extraction

Every relationship extracted by Predicato is stored as a **knowledge triple** — a unified `(subject, predicate, object)` record with typed endpoints and optional context fields:

```
┌──────────────────────────────────────────────────────────────────┐
│  Subject: "Lisinopril"    SubjectType: "Drug"                    │
│  Predicate: "treats"                                             │
│  Object: "Hypertension"   ObjectType: "Disease"                  │
│                                                                  │
│  Context:                                                        │
│    Condition:    "when first-line therapy is appropriate"         │
│    Temporal:     "ongoing"                                        │
│    Certainty:    "established"                                    │
│    Scope:        "adults"                                        │
│    Source:       "StatPearls Hypertension chapter"                │
│    Confidence:   0.95                                            │
└──────────────────────────────────────────────────────────────────┘
```

**Standard extraction** produces triples with subject, predicate, object, types, and descriptions from every chunk. **Extended extraction** (`ExtendedExtraction: true`) makes a second LLM pass to enrich triples with contextual fields and extract conditional rules:

```go
client.Add(ctx, episodes, &predicato.AddEpisodeOptions{
    ExtendedExtraction: true,  // Enable contextual enrichment
})
```

Extended extraction adds:
- **Contextual fields** on triples: `condition`, `temporal`, `location`, `certainty`, `scope`, `source_attribution`
- **Conditional rules**: IF-THEN-UNLESS patterns (e.g., "IF patient has hypertension THEN prescribe ACE inhibitor UNLESS contraindicated")

All fields are stored as flat columns in the fact store — no nested JSON. This makes them directly queryable via SQL and searchable via the hybrid search pipeline.

### Custom Graph Modeling

Override the default resolution logic by implementing `GraphModeler`:

```go
type GraphModeler interface {
    ResolveEntities(ctx, input) (*EntityResolutionOutput, error)
    ResolveRelationships(ctx, input) (*RelationshipResolutionOutput, error)
    BuildCommunities(ctx, input) (*CommunityOutput, error)
}

// Use with AddEpisodeOptions
client.AddEpisode(ctx, episode, &predicato.AddEpisodeOptions{
    GraphModeler: myCustomModeler,
})

// Or set as default
client, _ := predicato.NewClient(db, llm, embedder, &predicato.Config{
    DefaultGraphModeler: myCustomModeler,
})
```

Validate custom modelers before use:

```go
result, _ := client.ValidateModeler(ctx, myCustomModeler)
if !result.Valid {
    log.Fatalf("Modeler validation failed: %v", result.EntityResolution.Error)
}
```

## Components

| Component | Internal (No API) | External Services |
|-----------|-----------------|---------------------|
| **Graph Database** | CozoDB (embedded), DuckDB+DuckPGQ (embedded), Ladybug (embedded) | Neo4j, Memgraph |
| **Embeddings** | go-candle | OpenAI compatible APIs, AWS bedrock, Gemini |
| **Reranking** | go-candle | Jina, Cohere |
| **Text Generation** | go-candle (SmolLM2) | OpenAI compatible APIs |
| **Entity Extraction** | GLiNER (ONNX) | GLiNER2 (API) | LLM-based extraction |
| **Fact Storage** | DoltGres (embedded) | PostgreSQL + VectorChord |

**Serving:** an HTTP/MCP server (`predicato server`) and a **gRPC server**
(`predicato serve-grpc`) — run the graph as a separate process / on its own
machine, query it remotely, and scale it as a pool. See
[docs/GRPC_SERVER.md](docs/GRPC_SERVER.md).

**Why choose Predicato:**
- **Security** - Don't expose your data to external services
- **Run offline** - Embedded database + local ML models = no network required
- **Swap components freely** - Same code works with local models or cloud APIs
- **Bi-temporal knowledge** - Track when facts were recorded AND when they were valid
- **Hybrid search** - Semantic + BM25 keyword + graph traversal in one query
- **Production hardened** - WAL recovery, circuit breakers, cost tracking, telemetry

## Quick Start (Internal Stack)

No API keys. No external services. Just Go and CGO.

```go
package main

import (
    "context"
    "log"
    "time"

    "github.com/soundprediction/predicato"
    candleAdapter "github.com/soundprediction/predicato/pkg/candle"
    "github.com/soundprediction/predicato/pkg/driver"
)

func main() {
    ctx := context.Background()

    // Embedded graph database — CozoDB (recommended), DuckDB, or Ladybug
    db, _ := driver.NewCozoDriver("./knowledge.cozo", 1024)
    defer db.Close()

    // Local text generation (SmolLM2, no API)
    candleClient, _ := candleAdapter.NewClient(&candleAdapter.CandleNLPConfig{
        TextGenModelID: "HuggingFaceTB/SmolLM2-360M-Instruct",
    })
    llmClient := candleAdapter.NewLLMAdapter(candleClient, "text_generation")
    defer candleClient.Close()

    // Local embeddings (no API)
    embedderClient, _ := candleAdapter.NewCandleEmbedderClient(&candleAdapter.CandleEmbedderConfig{
        Model:      "qwen/qwen3-embedding-0.6b",
        Dimensions: 1024,
    })
    defer embedderClient.Close()

    // Local reranking (no API) — uses embedder for cosine-similarity reranking
    reranker := candleAdapter.NewCandleRerankerClient(embedderClient)

    // Create client
    client, _ := predicato.NewClient(db, llmClient, embedderClient, &predicato.Config{
        GroupID:  "my-app",
        TimeZone: time.UTC,
    }, nil)
    defer client.Close(ctx)

    // Add knowledge
    client.Add(ctx, []predicato.Episode{{
        ID:        "meeting-1",
        Name:      "Team Standup",
        Content:   "Alice mentioned the API redesign is blocked on the auth team.",
        Reference: time.Now(),
        CreatedAt: time.Now(),
        GroupID:   "my-app",
    }})

    // Search with reranking
    results, _ := client.Search(ctx, "API redesign status", nil)

    // Rerank for better relevance
    passages := make([]string, len(results.Nodes))
    for i, node := range results.Nodes {
        passages[i] = node.Summary
    }
    ranked, _ := reranker.Rank(ctx, "API redesign status", passages)

    log.Printf("Top result: %s (score: %.2f)", ranked[0].Passage, ranked[0].Score)
}
```

First run downloads models (~1.7GB total). Subsequent runs use cached models.

## Quick Start (External APIs)

The same interfaces work with cloud services or different embedded backends - just swap the implementations:

```go
package main

import (
    "context"
    "log"
    "os"
    "time"

    "github.com/soundprediction/predicato"
    "github.com/soundprediction/predicato/pkg/driver"
    "github.com/soundprediction/predicato/pkg/embedder"
    "github.com/soundprediction/predicato/pkg/nlp"
)

func main() {
    ctx := context.Background()

    // Choose your graph database:
    // Option A: CozoDB embedded (recommended)
    // db, _ := driver.NewCozoDriver("./knowledge.cozo", 1024)
    // Option B: DuckDB + DuckPGQ embedded
    // db, _ := driver.NewDuckPGQDriver("./knowledge.duckdb", 1024)
    // Option C: Ladybug embedded (legacy)
    // db, _ := driver.NewLadybugDriver("./knowledge.db", 1024)
    // Option D: Neo4j (external)
    db, _ := driver.NewNeo4jDriver(
        os.Getenv("NEO4J_URI"),
        os.Getenv("NEO4J_USER"),
        os.Getenv("NEO4J_PASSWORD"),
    )
    defer db.Close(ctx)

    // OpenAI for LLM and embeddings
    apiKey := os.Getenv("OPENAI_API_KEY")
    llmClient, _ := nlp.NewOpenAIClient(apiKey, nlp.Config{Model: "gpt-4o-mini"})
    embedderClient := embedder.NewOpenAIEmbedder(apiKey, embedder.Config{
        Model: "text-embedding-3-small",
    })

    client, _ := predicato.NewClient(db, llmClient, embedderClient, &predicato.Config{
        GroupID:  "my-app",
        TimeZone: time.UTC,
    }, nil)
    defer client.Close(ctx)

    // Same API as internal stack
    client.Add(ctx, []predicato.Episode{{
        ID:      "meeting-1",
        Content: "Alice mentioned the API redesign is blocked.",
        // ...
    }})
}
```

## Installation

```bash
go get github.com/soundprediction/predicato
```

### Building with Embedded Graph Databases

Predicato supports three embedded graph databases. Each requires CGO and uses a build tag:

| Driver | Build Tag | Library |
|--------|-----------|---------|
| **CozoDB** (recommended) | `system_cozo` | [cozo-lib-go](https://github.com/cozodb/cozo-lib-go) |
| **DuckDB + DuckPGQ** | `system_duckpgq` | [go-duckdb](https://github.com/marcboeker/go-duckdb) + [duckpgq](https://github.com/cwida/duckpgq-extension) |
| **Ladybug** (legacy) | `system_ladybug` | Custom embedded graph DB |

```bash
# Clone the repository
git clone https://github.com/soundprediction/predicato
cd predicato

# Build with CozoDB (recommended)
go build -tags system_cozo ./...

# Build with DuckDB + DuckPGQ
go build -tags system_duckpgq ./...

# Build with Ladybug (requires native library download)
go generate ./cmd/main.go
make build

# Run tests
make test

# Build CLI binary
make build-cli
```

#### Ladybug Manual Build (without Make)

```bash
# Step 1: Download Ladybug library
go generate ./cmd/main.go

# Step 2: Build with CGO flags
export CGO_LDFLAGS="-L$(pwd)/cmd/lib-ladybug -Wl,-rpath,$(pwd)/cmd/lib-ladybug"
go build -tags system_ladybug ./...
```

### Building Without CGO

Many packages work without CGO dependencies:

```bash
# Build core packages (no CGO required)
go build ./pkg/factstore/...
go build ./pkg/embedder/...
go build ./pkg/nlp/...

# Run pure Go tests
make test-nocgo
```

### Prerequisites

**Embedded graph databases (CozoDB, DuckDB, or Ladybug):**
- Go 1.21+
- GCC (for CGO compilation)
- Make (recommended for Ladybug)
- ~4GB RAM for local models

**External APIs only (no CGO needed):**
- Go 1.21+
- API keys for your chosen providers

## Examples

| Example | Description |
|---------|-------------|
| [`examples/basic/`](examples/basic/) | Full internal stack - CozoDB + Candle + Reranking |
| [`examples/chat/`](examples/chat/) | Interactive chat with local models |
| [`examples/external_apis/`](examples/external_apis/) | Neo4j + OpenAI integration |

## Architecture

```
predicato/
├── pkg/driver/        # Graph databases (CozoDB, DuckDB+DuckPGQ, Ladybug, Neo4j, Memgraph)
├── pkg/candle/        # Local ML models via go-candle (embeddings, reranking, text gen, NER, translation)
├── pkg/embedder/      # Embedding providers (Candle, OpenAI, Gemini)
├── pkg/crossencoder/  # Reranking (Candle, Jina, LLM-based)
├── pkg/nlp/           # LLM clients (OpenAI-compatible APIs)
├── pkg/search/        # Hybrid search (semantic + BM25 + graph traversal)
├── pkg/factstore/     # Fact storage (PostgreSQL/DoltGres + VectorChord)
└── pkg/types/         # Core types (nodes, triples, edges, episodes)
```

## Internal Services Stack

We rely on Go bindings for ML models implemented in Rust. In particular we use [go-candle](https://github.com/soundprediction/go-candle) (HuggingFace candle, pure Rust FFI) for embeddings, reranking, and text generation, and [go-gline-rs](https://github.com/fbilhaut/gline-rs) for GLiNER NER. All models are pure Rust with no runtime dependencies (no libtorch, no ONNX Runtime). Predicato will automatically download models on first use and cache to `~/.cache/huggingface/`.

Here is an example configuration and the model sizes involved:

| Component | Model | Download Size |
|-----------|-------|---------------|
| Embeddings | `qwen/qwen3-embedding-0.6b` | ~600MB |
| Reranking | Embedding-based cosine similarity | (shares embedder) |
| Text Generation | SmolLM2-360M-Instruct | ~350MB |

Models download automatically on first use and cache to `~/.cache/huggingface/`.

## Key Features

**Temporal Knowledge Graph**
- Bi-temporal model: `created_at` (when recorded) vs `valid_from/valid_to` (when true)
- Automatic invalidation of contradicting facts
- Historical queries: "what did we know about X as of date Y?"

**Hybrid Search**
- Semantic similarity (cosine distance on embeddings)
- BM25 keyword matching
- Graph traversal (BFS expansion through relationships)
- 5 reranking strategies: RRF, MMR, cross-encoder, node distance, episode mentions

**Production Ready**
- Circuit breakers with provider fallback
- Token usage tracking and cost calculation
- Error telemetry with DB persistence

## Fact Storage & RAG

Predicato includes a **fact storage system** for extracted entities, knowledge triples, and conditional rules that can be used independently for RAG (Retrieval-Augmented Generation) without requiring graph queries.

The fact store persists:
- **`extracted_nodes`** — entities with names, types, descriptions, and embeddings
- **`extracted_triples`** — knowledge triples with subject/predicate/object, typed endpoints, contextual fields, and embeddings
- **`extracted_rules`** — conditional rules (IF-THEN-UNLESS patterns)
- **`sources`** — original source documents with metadata

### PostgreSQL Backend

The fact storage uses PostgreSQL-compatible databases with **VectorChord** for native vector similarity search:

| Mode | Database | Use Case |
|------|----------|----------|
| **Embedded** | DoltGres | Development, single-node deployment |
| **External** | PostgreSQL + VectorChord | Production, managed databases (RDS, Cloud SQL) |

If no external PostgreSQL is configured, Predicato automatically uses **DoltGres** (embedded PostgreSQL-compatible database with git-like versioning).

### Configuration

```go
// Option 1: Automatic embedded DoltGres (no config needed)
client, _ := predicato.NewClient(db, llm, embedder, &predicato.Config{
    GroupID: "my-app",
}, nil)

// Option 2: External PostgreSQL with VectorChord
client, _ := predicato.NewClient(db, llm, embedder, &predicato.Config{
    GroupID: "my-app",
    FactStoreConfig: &factstore.FactStoreConfig{
        Type:             "postgres",
        ConnectionString: "postgres://user:pass@localhost:5432/facts?sslmode=disable",
    },
}, nil)
```

### RAG Search (without Graph)

For simpler RAG use cases that don't need relationship traversal:

```go
// Search extracted facts directly (no graph queries)
results, _ := client.SearchFacts(ctx, "API design patterns", &types.SearchConfig{
    Limit:    10,
    MinScore: 0.7,
})

for _, node := range results.Nodes {
    fmt.Printf("Found: %s (score: %.2f)\n", node.Name, node.Score)
}
```

This performs hybrid search (vector similarity + keyword matching) using VectorChord and PostgreSQL full-text search.

## Post-Hoc Verification (Hallucination Detection)

Predicato includes a verification pipeline (`pkg/verification/`) that checks LLM-generated claims against source facts from the knowledge graph. It runs entirely on local models with no LLM calls required.

### Pipeline

1. **Claim splitting** — breaks the LLM response into individual sentences, filters short fragments, and detects meta-statements ("I don't have information about...") that should not be flagged
2. **Evidence matching** — batch-embeds all claims and source facts, computes cosine similarity to find the best-matching source for each claim
3. **NLI entailment check** — runs a DeBERTa cross-encoder to classify each claim as entailment/neutral/contradiction against its best evidence
4. **Aggregation** — computes an overall trust score (fraction of supported claims) and a pass/fail verdict

### Usage

```go
// Create an NLI client (LLM-free, ~80MB model auto-downloaded from HF Hub)
nli, _ := verification.NewCandleNLIClient(&verification.CandleNLIConfig{
    Model: "cross-encoder/nli-deberta-v3-xsmall",
})
defer nli.Close()

// Attach to a predicato client
client.SetVerifier(nli)

// After generating an LLM response with RAG context:
result, _ := client.VerifyResponse(ctx, llmResponse, ragFacts, nil)
fmt.Printf("Trust: %.0f%%, Pass: %v\n", result.TrustScore*100, result.Pass)
for _, c := range result.Claims {
    fmt.Printf("  [%s] %s (confidence: %.2f)\n", c.Verdict, c.Text, c.Confidence)
}
```

### NLI Backends

| Backend | Constructor | Latency | Dependencies |
|---------|-------------|---------|-------------|
| DeBERTa cross-encoder | `NewCandleNLIClient()` | ~50ms/claim | CGO (go-candle FFI), model from HF Hub |
| LLM prompt | `NewLLMNLIClient(nlpClient)` | ~500ms/claim | Any `nlp.Client` implementation |

### Custom Thresholds

```go
result, _ := client.VerifyResponse(ctx, response, facts, &verification.VerifyOptions{
    GateThreshold:          0.30,  // Min similarity to run NLI (default: 0.25)
    SupportThreshold:       0.45,  // Min similarity for "supported" (default: 0.40)
    ContradictionThreshold: 0.60,  // NLI contradiction probability threshold (default: 0.50)
    EntailmentThreshold:    0.60,  // NLI entailment probability threshold (default: 0.50)
    TrustPassScore:         0.80,  // Min trust score to pass (default: 0.70)
})
```

## CLI & Server

```bash
# Build CLI
make build-cli

# Start HTTP server
./bin/predicato server --port 8080

# Import parquet files directly into a graph (no PostgreSQL needed)
./bin/predicato parquet-import \
  --driver duckpgq \
  --input-dir ./parquet-files \
  --output ./knowledge_graph.duckdb \
  --group-id wikidata

# Import into Ladybug instead
./bin/predicato parquet-import \
  --driver ladybug \
  --input-dir ./parquet-files \
  --output ./knowledge_graph.db

# Convert between graph drivers
./bin/predicato convert-graph \
  --source-driver duckpgq --source-uri ./graph.duckdb \
  --dest-driver ladybug --dest-uri ./graph.db

# API endpoints
POST /api/v1/ingest/messages  # Add content
POST /api/v1/ingest/extract   # Extract entities (two-stage)
POST /api/v1/ingest/promote   # Promote to graph (two-stage)
POST /api/v1/search           # Search knowledge graph
POST /api/v1/search/facts     # Search fact store (cosine similarity)
GET  /api/v1/episodes/:id     # Get episodes
```

### Parquet Import

The `parquet-import` command loads predicato-format parquet files directly into a knowledge graph, bypassing PostgreSQL. This is ideal for batch/HPC workloads where you want a self-contained graph database file.

**Supported drivers:**
- **DuckPGQ**: Uses DuckDB's native `read_parquet()` for zero-copy columnar loading. Best for large datasets on high-memory nodes.
- **Ladybug**: Uses Ladybug's native `COPY FROM` parquet import. Generates intermediate parquet files matching Ladybug's schema.

**Expected input files:**
- `nodes.parquet` — entities with embeddings
- `extracted_triples.parquet.gz` — triples with embeddings
- `extracted_rules.parquet.gz` — rules (optional)
- These are produced by the wikidata pipeline (`duckdb_to_predicato.py` or `wikidata_to_predicato.py`).

**DuckPGQ-specific options:**
- `--threads N` — DuckDB thread count
- `--memory-limit SIZE` — DuckDB memory limit (e.g., `900GB`)

Both drivers also support `BulkLoadFromPostgres()` for loading directly from a PostgreSQL fact store.

### Topic-Scoped Graph Loading

The full wikidata knowledge graph is enormous (100M+ nodes, billions of edges). For most
applications a smaller, domain-specific graph is more useful. The `topic-import` and
`topic-explore` commands let you create a focused graph around any topic by leveraging the
pre-computed embeddings already present in the parquet files.

#### Discovering the data with `topic-explore`

Before importing, use `topic-explore` to inspect the parquet files without writing any
output database:

```bash
# Print entity type distribution, top predicates, and embedding coverage:
./bin/predicato topic-explore --input-dir ./wikidata-output

# Rank the top-20 triples by semantic similarity to a topic (requires embedder config):
./bin/predicato topic-explore --input-dir ./wikidata-output --topic "diabetes mellitus"
```

Example output (no `--topic`):
```
=== Parquet Statistics ===
Nodes:   94325811
Triples: 312456789
Rules:   8234567

--- Top Entity Types ---
  DISEASE                                  12345678
  DRUG                                      9876543
  ...

--- Embedding Coverage ---
  Nodes:   92.3%
  Triples: 98.1%
  Rules:   95.4%
```

#### Building a topic-specific graph with `topic-import`

`topic-import` loads only the triples and rules whose embeddings have cosine similarity
≥ `--threshold` to the topic vector, then inserts only the nodes referenced by those
triples/rules.

```bash
# Embed the topic at runtime (requires PREDICATO_EMBEDDING_API_KEY or config):
./bin/predicato topic-import \
  --input-dir ./wikidata-output \
  --output   ./diabetes.duckdb \
  --topic    "diabetes mellitus type 2 treatment" \
  --threshold 0.65 \
  --group-id  wikidata \
  --embedding-dim 1024

# Use a pre-computed vector (no API key needed — safe for offline/HPC):
./bin/predicato topic-import \
  --input-dir    ./wikidata-output \
  --output       ./diabetes.duckdb \
  --topic-vector ./diabetes_vector.json \
  --threshold    0.65
```

The `--topic-vector` flag accepts a JSON file containing a `[]float32` array of exactly
`--embedding-dim` floats, e.g.:
```json
[0.012, -0.034, 0.091, ...]
```

**Key flags:**
| Flag | Default | Description |
|------|---------|-------------|
| `--topic` | | Topic string embedded at runtime via configured embedder |
| `--topic-vector` | | Path to pre-computed `[]float32` JSON vector |
| `--threshold` | `0.65` | Minimum cosine similarity for triple/rule inclusion |
| `--max-nodes` | `0` (unlimited) | Hard cap on nodes inserted |
| `--max-edges` | `0` (unlimited) | Hard cap on edges inserted |
| `--force` | | Overwrite existing output file |

**Calibrating the threshold:**
- `0.0` — include all triples/rules with non-null embeddings (same as full import)
- `0.5` — broad domain coverage
- `0.65` — default; focused on the topic with some contextual neighbours
- `0.8` — tight focus; may miss important context

Use `topic-explore --topic "..."` to see the distribution of similarity scores before
committing to a threshold.

> [!NOTE]
> **GLiNER2 Support**: Use the `--gliner2` flag to enable the GLiNER2 entity extraction provider.
> Predicato will automatically start the local GLiNER2 Python service (port 11435) if it is not already running.
> This requires `uv` or `python3` to be available in your path. Configured models (default `fastino/gliner2-multi-v1`) will be downloaded on first use.

## Python Client

Predicato includes an official Python client for interacting with the HTTP server.

### Installation

```bash
pip install predicato
# or with uv
uv add predicato
```

### Quick Start

```python
from predicato import PredicatoClient

# Connect to the server
with PredicatoClient(base_url="http://localhost:8080") as client:
    # Add content
    result = client.add_episode(
        name="Team Meeting",
        content="Alice mentioned the API redesign is blocked on auth.",
        group_id="my-project",
        source="meeting-notes",
    )

    # Search the knowledge graph
    results = client.search(
        query="API redesign status",
        group_id="my-project",
        limit=10,
    )

    for node in results.nodes:
        print(f"- {node.name} ({node.entity_type})")
```

### Two-Stage Ingestion

For more control over entity extraction and graph building:

```python
from predicato import PredicatoClient

with PredicatoClient(base_url="http://localhost:8080") as client:
    # Stage 1: Extract entities and relationships
    extraction = client.extract_to_facts(
        name="Medical Article",
        content="Hypertension is treated with ACE inhibitors like lisinopril...",
        group_id="medical-kb",
        entity_types={
            "Disease": {"description": "A medical condition"},
            "Drug": {"description": "A medication"},
        },
    )

    print(f"Extracted {len(extraction.extracted_nodes)} entities")

    # Inspect/modify extracted entities before promoting...

    # Stage 2: Promote to graph with entity resolution
    result = client.promote_to_graph(source_id=extraction.source_id)

    print(f"Resolved {len(result.nodes)} entities in graph")
```

### Async Support

```python
import asyncio
from predicato import AsyncPredicatoClient

async def main():
    async with AsyncPredicatoClient(base_url="http://localhost:8080") as client:
        result = await client.add_episode(
            name="Meeting Notes",
            content="...",
            group_id="my-project",
        )

asyncio.run(main())
```

### Fact Store Search (RAG)

Search the structured fact store directly using cosine similarity:

```python
from predicato import PredicatoClient

with PredicatoClient(base_url="http://localhost:8080") as client:
    # Search extracted entities with vector similarity
    results = client.search_facts(
        query="hypertension treatment options",
        group_id="medical-kb",
        limit=20,
        min_score=0.5,  # Cosine similarity threshold
    )

    # Results include similarity scores
    for node, score in zip(results.nodes, results.node_scores):
        print(f"- {node.name} ({node.type}): {score:.2f}")
```

This is useful for RAG applications that need semantic search without full graph traversal.

See [`python/`](python/) for full documentation and [`python/examples/`](python/examples/) for more examples including StatPearls medical article ingestion.

## Documentation

- [Getting Started](docs/GETTING_STARTED.md)
- [Fact Store RAG](docs/FACTSTORE_RAG.md)
- [Extended Fact Storage](docs/EXTENDED_FACT_STORAGE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Ladybug Setup](docs/ladybug_SETUP.md)
- [FAQ](docs/FAQ.md)

## License

Apache 2.0

## Acknowledgments

Inspired by [Graphiti](https://github.com/getzep/graphiti) by Zep.
