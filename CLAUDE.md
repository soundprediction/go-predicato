# Predicato Project Instructions

## Overview

The `predicato` package is a temporal knowledge graph framework for Go, designed for building and querying dynamic knowledge graphs that evolve over time. It provides real-time incremental updates without batch recomputation, with hybrid search combining semantic embeddings, keyword search, and graph traversal.

## Architecture

### Two-Stage Design

Predicato separates knowledge extraction from graph modeling into two distinct, persisted stages:

1. **Fact Store (extraction)** — LLM-powered entity and relationship extraction produces raw nodes, edges, and embeddings that are persisted to PostgreSQL (with VectorChord) via the `FactsDB` interface. This step is expensive (LLM calls, embedding generation) and its output is stored durably so it never needs to be repeated.

2. **Knowledge Graph (modeling)** — The persisted facts are promoted into a graph database (LadybugDB, Neo4j, Memgraph) through entity resolution, deduplication, temporal modeling, and community detection via the `GraphModeler` interface.

The key insight: because the extraction results are persisted independently of any particular graph, you can generate **multiple graph views** from the same extracted facts — with different resolution thresholds, entity type filters, or custom `GraphModeler` implementations — without re-running extraction. The `ExtractOnly` option (`AddEpisodeOptions.ExtractOnly = true`) stops after stage 1, and `PromoteToGraph()` runs stage 2 on demand.

Key code paths in `ingestion.go` / `ingestion_factstore.go`:
- `AddEpisode()` → checks for `factStore` → calls `ExtractToFacts()` then `PromoteToGraph()`
- `ExtractToFacts()` — chunks content, extracts entities/edges via LLM, saves to `FactsDB`
- `PromoteToGraph()` — reads from `FactsDB`, runs `GraphModeler` (entity resolution, edge resolution, communities), writes to graph driver

### Core Library (root package)
- **Entry point**: `predicato.NewClient(driver, nlProcessor, embedder, config)`
- **Key interfaces** (following Interface Segregation Principle):
  - `EpisodeManager`: Add, remove, retrieve episodes
  - `GraphQuerier`: Read-only search and query operations
  - `GraphModeler`: Pluggable entity resolution, relationship resolution, and community detection
  - `FactsDB`: Intermediate storage for raw extracted knowledge
  - `Predicato`: Composed interface combining all capabilities
- **Core files**: `interfaces.go`, `ingestion.go`, `ingestion_factstore.go`, `retrieval.go`, `graph_ops.go`

### Package Structure (`pkg/`)
- **`driver/`**: Graph database abstraction layer
  - CozoDB (embedded, `system_cozo`), DuckDB+DuckPGQ (embedded, `system_duckpgq`), LadybugDB (embedded, `system_ladybug`), Neo4j, Memgraph, FalkorDB
- **`factstore/`**: Persistent storage for extracted facts/entities
  - PostgreSQL with VectorChord extension
  - DuckDB for analytical workloads
  - Hybrid search: vector similarity + keyword/fulltext + RRF merging
- **`nlp/`**: LLM client interfaces (OpenAI-compatible)
- **`candle/`**: Local ML models via go-candle (embeddings, reranking, text gen, NER, translation)
- **`embedder/`**: Embedding model clients (Candle, OpenAI, Gemini)
- **`search/`**: Search configuration and reranking
- **`crossencoder/`**: Cross-encoder reranking support (Candle, Jina, LLM-based)
- **`prompts/`**: LLM prompt templates for extraction, deduplication, summarization
- **`modeler/`**: Knowledge graph modeling pipeline
- **`types/`**: Core type definitions (Node, Edge, Episode, etc.)
- **`gliner/`**, **`gliner2/`**: Named entity recognition via GLiNER
- **`community/`**: Community detection in knowledge graphs
- **`checkpoint/`**: Graph state checkpointing
- **`server/`**: HTTP REST API server
- **`config/`**: Configuration management
- **`logger/`**: Structured logging with color handler
- **`cache/`**: Caching utilities
- **`cost/`**: LLM cost tracking
- **`alert/`**: Alerting utilities
- **`telemetry/`**: OpenTelemetry integration
- **`models/`**: Database model definitions
- **`utils/`**: Shared utilities and maintenance operations

### CLI (`cmd/`)
- **`cmd/main.go`**: Main entry point (downloads LadybugDB via `go generate`)
- **`cmd/predicato/`**: Cobra CLI commands:
  - `server`: Start the HTTP REST API server
  - `mcp`: Start the Model Context Protocol (MCP) server
  - `version`: Print version information
- **`cmd/mcp-server/`**: Standalone MCP server (Firebase Genkit-based)
- **`cmd/gliner2-api/`**: GLiNER2 NER API server (Gin-based)

### Python Client (`python/`)
- Python client library for the predicato REST API
- Dependencies: httpx, pydantic, typing-extensions
- Optional extras: `crawler` (beautifulsoup4, openai), `server` (fastapi, uvicorn, gliner2)
- Tooling: ruff, black, mypy, pytest

### Examples (`examples/`)
- `basic/`: Basic knowledge graph usage
- `chat/`: Chat interface example
- `factstore_rag/`: Fact store RAG pipeline
- `external_apis/`: External API integration
- `prompts/`: Prompt usage examples

## Building

**IMPORTANT**: This project uses CGO and requires the LadybugDB native library. Use the Makefile for building:

```bash
# Build everything (recommended)
make build

# Build the CLI binary
make build-cli

# Run all tests (CGO + non-CGO)
make test

# Run only non-CGO tests (no native library needed)
make test-nocgo

# Run only CGO tests
make test-cgo

# Run server
make run-server

# Run server in debug mode
make run-server-debug

# Development workflow (fmt, vet, test)
make dev

# All available targets
make help
```

### Manual Build Steps

If you need to build manually:

```bash
# Download the ladybug native library (required before building)
go generate ./cmd/main.go

# Build with the system_ladybug tag
go build -tags system_ladybug ./...

# Build the CLI
go build -tags system_ladybug -o bin/predicato ./cmd/main.go
```

**Note**: The `go generate` command downloads `liblbug.so` to `cmd/lib-ladybug/`. If you see linker errors like `cannot find -llbug`, run `go generate ./cmd/main.go` first.

**Known Issue**: Building `./...` (all packages including examples) may fail with linker errors because the examples need their own copy of `liblbug.so`. Use `make build-cli` to build the main CLI, or run `go generate` in each example directory before building.

### CGO vs Non-CGO Packages

Some packages require CGO (LadybugDB native library), others are pure Go:
- **CGO required**: `pkg/driver`, `pkg/checkpoint`, `pkg/modeler`, `pkg/utils`, `pkg/factstore`
- **Pure Go**: `pkg/embedder`, `pkg/nlp`, `pkg/prompts`, `pkg/logger`, `pkg/types`

Files that import CGO-dependent packages use `//go:build cgo` build tags.

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs:
1. **Lint & Format**: gofmt check + `go vet` (CGO_ENABLED=0)
2. **Test (No CGO)**: Pure Go package tests (CGO_ENABLED=0)
3. **Test (CGO)**: Tests requiring LadybugDB native library
4. **Build**: Full build with CGO

## Technical Considerations

- **Language**: Go 1.25.5
- **Type Safety**: Leverage Go's type system for compile-time safety
- **Concurrency**: Use goroutines and channels effectively
- **Error Handling**: Explicit error handling with clear messages and proper wrapping
- **Resource Management**: Cleanup via defer and Close() methods
- **Testing**: Comprehensive tests required

## Formatting & Linting

**Always run `gofmt` before committing** — CI checks formatting and will reject unformatted code:

```bash
# Check for formatting issues
gofmt -l .

# Auto-fix formatting
gofmt -w .
```

This project also uses `betteralign` to optimize struct field ordering for memory efficiency and `golangci-lint` with the `fieldalignment` govet pass. Configuration is in `.golangci.yml`.

**After modifying or creating structs**, run the struct alignment linter:

```bash
# Check for alignment issues
betteralign ./...

# Auto-fix alignment issues
betteralign --apply ./...
```

**IMPORTANT**: `betteralign --apply` reorders struct fields. If any struct literals use positional (non-named) initialization, the reordering will break them. After running `--apply`, always verify with `go build ./...` and fix any struct literals to use named fields.

## Quality Standards

- Code should compile without warnings
- All public functions should have proper documentation
- Error messages should be clear and actionable
- Performance should be reasonable for the intended use cases
- Memory usage should be efficient and not leak resources
- Struct fields should be ordered for optimal memory alignment (run `betteralign ./...` to check)
