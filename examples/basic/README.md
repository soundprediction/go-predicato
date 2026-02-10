# Basic Example - Internal Services Stack

This example demonstrates Predicato's fully local setup using only internal services - **no API keys or external services required**.

## What This Example Shows

- Creating a Predicato client with the internal services stack
- Using Ladybug embedded database (no server required)
- Using Candle SmolLM2 for text generation (local, no API)
- Using Candle for embeddings (qwen/qwen3-embedding-0.6b)
- Using Candle for reranking (embedding-based cosine similarity)
- Adding episodes to the knowledge graph
- Searching and reranking results

## Prerequisites

- **Go 1.21+**
- **GCC** (for CGO compilation)
- **~4GB RAM** minimum (for loading ML models)
- **~2GB disk space** (for model downloads)

No API keys needed!

## Model Downloads

First run will automatically download models to `~/.cache/huggingface/`:

| Component | Model | Size |
|-----------|-------|------|
| Embeddings | qwen/qwen3-embedding-0.6b | ~600MB |
| Reranking | Embedding-based cosine similarity | (shares embedder) |
| Text Generation | SmolLM2-360M-Instruct | ~350MB |

## Build & Run

```bash
# From the repository root
cd examples/basic

# Download native library (first time only)
go generate

# Build
go build -o basic_example .

# Run
./basic_example
```

Or use Make from the repository root:

```bash
make build
cd examples/basic
go run .
```

## Expected Output

```
================================================================================
Predicato Basic Example - Internal Services Stack
================================================================================

This example uses predicato's internal services:
  - Ladybug: embedded graph database (no server required)
  - Candle SmolLM2: local text generation (no API required)
  - Candle: local embeddings with qwen/qwen3-embedding-0.6b
  - Candle: local reranking via embedding-based cosine similarity

No API keys or external services needed!

[1/5] Setting up Ladybug embedded graph database...
      Ladybug driver created (embedded database at ./example_graph.db)
[2/4] Setting up Candle SmolLM2 for text generation...
      Candle SmolLM2 text generation model loaded
[3/4] Setting up Candle embedder with qwen/qwen3-embedding-0.6b...
      Candle embedder created (reranking shares embedder)
[4/4] Creating Predicato client...
      Predicato client created (group: example-group)

================================================================================
All components initialized successfully!
================================================================================

Adding sample episodes to the knowledge graph...
Added 3 episodes to the knowledge graph

Searching the knowledge graph for: "API design and deadlines"
Found 3 nodes and 5 edges

Search results (before reranking):
----------------------------------
  1. Meeting with Alice (episodic)
     Had a productive meeting with Alice about the new project...
  2. Project Research (episodic)
     Researched various approaches for implementing the API...

Reranking results with Candle embedding-based reranker...

Search results (after reranking):
---------------------------------
  1. (score: 0.892) Had a productive meeting with Alice about the new project...
  2. (score: 0.756) Researched various approaches for implementing the API...

Demonstrating text generation with Candle SmolLM2...
Prompt: The advantages of using a knowledge graph are
Generated: that it can be used to represent the relationships between...

================================================================================
Example completed successfully!
================================================================================
```

## Files Created

After running, you'll see:
- `./example_graph.db` - Ladybug database directory

## Troubleshooting

### "cannot find -llbug"

The Ladybug native library hasn't been downloaded:

```bash
go generate
```

### CGO errors

Ensure GCC is installed:

```bash
# Ubuntu/Debian
sudo apt install build-essential

# macOS
xcode-select --install
```

### Out of memory

The example requires ~4GB RAM. Close other applications or try a machine with more memory.

### Model download fails

Check your internet connection and disk space. Models download to `~/.cache/huggingface/`.

## Next Steps

- See `examples/chat/` for an interactive chat application
- See `examples/external_apis/` for using cloud services (OpenAI, Neo4j)
- Read the [Getting Started Guide](../../docs/GETTING_STARTED.md)
