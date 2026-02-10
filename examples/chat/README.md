# Interactive Chat Example (Internal Services Stack)

This example demonstrates how to build an interactive chat application using predicato with **only internal services** - no external APIs required!

## Internal Services Stack

| Component | Service | Model |
|-----------|---------|-------|
| **Database** | Ladybug | Embedded graph database |
| **Embeddings** | go-candle | `qwen/qwen3-embedding-0.6b` |
| **Reranking** | go-candle | Embedding-based cosine similarity |
| **Text Generation** | go-candle | SmolLM2-360M-Instruct |

## Features

- **Dual Knowledge Stores**: Separates global knowledge (shared facts) from user-specific episodic memory
- **Local Text Generation**: Uses Candle SmolLM2 for responses (no API required)
- **Local Embeddings**: Uses qwen/qwen3-embedding-0.6b for semantic search
- **Reranking**: Uses zhiqing/Qwen3-Reranker-0.6B-ONNX to improve search result quality
- **Conversation Continuity**: Uses `AddToEpisode` to maintain a single episode per chat session
- **UUID v7 Episode IDs**: Leverages time-sortable UUIDs for natural episode ordering
- **Interactive Commands**: Supports history viewing and direct knowledge base queries

**No API keys or external services required!**

## Architecture

The example creates two separate Predicato clients:

1. **Global Predicato Client** (optional):
   - Shared knowledge base across all users
   - Read-only for chat purposes
   - Used for contextual information retrieval
   - Search results are reranked for better relevance

2. **User Predicato Client**:
   - User-specific episodic memory
   - Stores conversation history
   - One episode per chat session

## Prerequisites

### Required
- Go 1.21 or later
- CGO enabled (required for Rust FFI bindings)
- ~4GB RAM minimum

### CGO Setup

CGO is required for the Rust-based ML libraries. Ensure you have:

**macOS:**
```bash
xcode-select --install
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install build-essential
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install gcc gcc-c++
```

### Verify CGO is enabled
```bash
go env CGO_ENABLED
# Should output: 1
```

## First Run - Model Downloads

On first run, the example will automatically download the required models:

| Model | Size | Purpose |
|-------|------|---------|
| `qwen/qwen3-embedding-0.6b` | ~600MB | Text embeddings + reranking |
| SmolLM2-360M-Instruct | ~350MB | Text generation |

**Total: ~950MB**

Models are cached after the first download.

## Usage

### Setup (First Time Only)

Download the Ladybug native library:

```bash
cd examples/chat
go generate ./...
```

### Basic Usage

Run with default settings (creates user database in `./user_dbs/`):

```bash
go run main.go
```

### With Custom User ID

```bash
go run main.go --user-id bob
```

### With Custom Database Paths

```bash
go run main.go \
  --user-id alice \
  --global-db /path/to/global/knowledge.ladybugdb \
  --user-db-dir /path/to/user/databases
```

### Without Global Knowledge Base

```bash
go run main.go --skip-global
```

## Interactive Commands

Once the chat is running, you can use these commands:

- `<your question>` - Ask the assistant a question
- `history` - View conversation history
- `search <query>` - Search the global knowledge base directly (with reranking)
- `exit` or `quit` - End the chat session

## Example Session

```
================================================================================
Predicato Interactive Chat - Internal Services Stack
================================================================================

This chat uses predicato's internal services:
  - Ladybug: embedded graph database (no server required)
  - Candle SmolLM2: local text generation (no API required)
  - Candle: local embeddings with qwen/qwen3-embedding-0.6b
  - Candle: local reranking via embedding-based cosine similarity

No API keys or external services needed!
User ID: alice

Initializing internal services...
(First run will download models, please wait...)

[1/3] Setting up Candle SmolLM2 for text generation...
      Candle SmolLM2 loaded
[2/3] Setting up Candle embedder with qwen/qwen3-embedding-0.6b...
      Candle embedder loaded (reranking shares embedder)
[3/3] Setting up Predicato clients...
      User database initialized at ./user_dbs/user_alice.ladybugdb

All components initialized successfully!

======================================================================
Predicato Interactive Chat
======================================================================

Commands:
  Type your question and press Enter
  Type 'exit' or 'quit' to end the session
  Type 'history' to view conversation history
  Type 'search <query>' to search the global knowledge base
======================================================================

You: What is a knowledge graph?
Created episode: 01930e1c-3a4f-7b2a-8c5d-1e2f3a4b5c6d