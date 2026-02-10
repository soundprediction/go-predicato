# Getting Started with Predicato

This guide will help you get started with Predicato, a temporal knowledge graph library for Go with a fully local ML stack.

## What Makes Predicato Different

Most agentic memory libraries require external services (OpenAI, Pinecone, Neo4j). Predicato is **modular by design** - every component can run locally OR connect to external services. Start with the internal stack for development, then swap in cloud services for production without changing your code.

| Component | Internal (No API) | External (Cloud) |
|-----------|-----------------|---------------------|
| **Graph Database** | Ladybug (embedded) | Neo4j, Memgraph |
| **Embeddings** | go-candle | OpenAI, Voyage, Gemini |
| **Reranking** | go-candle | Jina, Cohere |
| **Text Generation** | go-candle (SmolLM2) | OpenAI, Anthropic, Ollama |
| **Entity Extraction** | GLiNER (ONNX) | NLP model-based extraction |
| **Fact Storage** | DoltGres (embedded) | PostgreSQL + VectorChord |

## Prerequisites

- Go 1.21 or later
- GCC (for CGO compilation - required for internal stack)
- Make (optional, for convenience)

## Installation

```bash
go get github.com/soundprediction/predicato
```

## Quick Start: Internal Stack (Recommended)

**No API keys. No external services. Just Go and CGO.**

This example uses:
- Ladybug embedded database
- Local embeddings (qwen3-embedding via go-candle)
- Local text generation (SmolLM2 via go-candle)
- Local reranking (embedding-based cosine similarity)

### Step 1: Set Up Your Project

Create a new directory with these files:

**cgo.go** (required for native libraries):
```go
package main

//go:generate sh -c "curl -sL https://raw.githubusercontent.com/LadybugDB/go-ladybug/refs/heads/master/download_lbug.sh | bash -s -- -out lib-ladybug"

/*
#cgo darwin LDFLAGS: -L${SRCDIR}/lib-ladybug -Wl,-rpath,${SRCDIR}/lib-ladybug
#cgo linux LDFLAGS: -L${SRCDIR}/lib-ladybug -Wl,-rpath,${SRCDIR}/lib-ladybug
#cgo windows LDFLAGS: -L${SRCDIR}/lib-ladybug
*/
import "C"
```

**main.go**:
```go
package main

import (
    "context"
    "fmt"
    "log"
    "time"

    "github.com/soundprediction/predicato"
    candleAdapter "github.com/soundprediction/predicato/pkg/candle"
    "github.com/soundprediction/predicato/pkg/driver"
    "github.com/soundprediction/predicato/pkg/types"
)

func main() {
    ctx := context.Background()

    // 1. Embedded graph database (no server required)
    db, err := driver.NewLadybugDriver("./knowledge.db", 1)
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close(ctx)

    // 2. Local text generation (SmolLM2, no API)
    candleClient, err := candleAdapter.NewClient(&candleAdapter.CandleNLPConfig{
        TextGenModelID: "HuggingFaceTB/SmolLM2-360M-Instruct",
    })
    if err != nil {
        log.Fatal(err)
    }
    nlpClient := candleAdapter.NewLLMAdapter(candleClient, "text_generation")
    defer candleClient.Close()

    // 3. Local embeddings (no API)
    embedderClient, err := candleAdapter.NewCandleEmbedderClient(&candleAdapter.CandleEmbedderConfig{
        Model:      "qwen/qwen3-embedding-0.6b",
        Dimensions: 1024,
    })
    if err != nil {
        log.Fatal(err)
    }
    defer embedderClient.Close()

    // 4. Local reranking (no API) — uses embedder for cosine-similarity reranking
    reranker := candleAdapter.NewCandleRerankerClient(embedderClient)

    // 5. Create Predicato client
    client, err := predicato.NewClient(db, nlpClient, embedderClient, &predicato.Config{
        GroupID:  "my-app",
        TimeZone: time.UTC,
    }, nil)
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close(ctx)

    fmt.Println("Predicato client created with fully local stack!")

    // 6. Add knowledge
    _, err = client.Add(ctx, []types.Episode{{
        ID:        "meeting-1",
        Name:      "Team Standup",
        Content:   "Alice mentioned the API redesign is blocked on the auth team. Bob suggested we prioritize the database migration first.",
        Reference: time.Now(),
        CreatedAt: time.Now(),
        GroupID:   "my-app",
    }}, nil)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println("Episode added!")

    // 7. Search the knowledge graph
    results, err := client.Search(ctx, "API redesign status", nil)
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Found %d nodes\n", len(results.Nodes))

    // 8. Rerank for better relevance
    if len(results.Nodes) > 0 {
        passages := make([]string, len(results.Nodes))
        for i, node := range results.Nodes {
            passages[i] = node.Summary
        }
        ranked, err := reranker.Rank(ctx, "API redesign status", passages)
        if err != nil {
            log.Fatal(err)
        }
        if len(ranked) > 0 {
            fmt.Printf("Top result: %s (score: %.2f)\n", ranked[0].Passage, ranked[0].Score)
        }
    }
}
```

### Step 2: Download Native Libraries and Build

```bash
# Download the Ladybug native library
go generate ./cgo.go

# Build and run
go build -tags system_ladybug -o myapp .
./myapp
```

First run downloads ML models (~1.7GB total). Subsequent runs use cached models.

---

## Quick Start: External APIs

The same interfaces work with cloud services - just swap the implementations:

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
    "github.com/soundprediction/predicato/pkg/types"
)

func main() {
    ctx := context.Background()

    // Option A: Still use embedded Ladybug (recommended for simplicity)
    db, err := driver.NewLadybugDriver("./knowledge.db", 1)
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close(ctx)

    // Option B: Or use Neo4j for production scale
    // db, err := driver.NewNeo4jDriver(
    //     os.Getenv("NEO4J_URI"),
    //     os.Getenv("NEO4J_USER"),
    //     os.Getenv("NEO4J_PASSWORD"),
    // )

    // OpenAI for NLP model and embeddings
    apiKey := os.Getenv("OPENAI_API_KEY")
    nlpClient, err := nlp.NewOpenAIClient(apiKey, nlp.Config{Model: "gpt-4o-mini"})
    if err != nil {
        log.Fatal(err)
    }
    
    embedderClient := embedder.NewOpenAIEmbedder(apiKey, embedder.Config{
        Model:      "text-embedding-3-small",
        Dimensions: 1536,
    })

    // Same API as internal stack
    client, err := predicato.NewClient(db, nlpClient, embedderClient, &predicato.Config{
        GroupID:  "my-app",
        TimeZone: time.UTC,
    }, nil)
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close(ctx)

    // Add and search - identical API
    client.Add(ctx, []types.Episode{{
        ID:        "doc-1",
        Name:      "Project Update",
        Content:   "The new feature is ready for testing.",
        Reference: time.Now(),
        CreatedAt: time.Now(),
        GroupID:   "my-app",
    }}, nil)

    results, _ := client.Search(ctx, "feature status", nil)
    for _, node := range results.Nodes {
        log.Printf("Found: %s", node.Name)
    }
}
```

Run with:
```bash
export OPENAI_API_KEY=sk-your-key
go generate ./cgo.go
go build -tags system_ladybug -o myapp .
./myapp
```

---

## Quick Start: Local LLM with Ollama

For a middle ground - local NLP model, but simpler setup than the full internal stack:

**1. Install Ollama:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2
```

**2. Use in code:**
```go
package main

import (
    "context"
    "log"
    "time"

    "github.com/soundprediction/predicato"
    "github.com/soundprediction/predicato/pkg/driver"
    "github.com/soundprediction/predicato/pkg/nlp"
)

func main() {
    ctx := context.Background()

    // Embedded database
    db, err := driver.NewLadybugDriver("./knowledge.db", 1)
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close(ctx)

    // Ollama for local NLP (OpenAI-compatible API)
    nlpClient, err := nlp.NewOpenAIClient("ollama", nlp.Config{
        Model:   "llama3.2",
        BaseURL: "http://localhost:11434/v1",
    })
    if err != nil {
        log.Fatal(err)
    }

    // Create client (embedder optional)
    client, err := predicato.NewClient(db, nlpClient, nil, &predicato.Config{
        GroupID:  "my-app",
        TimeZone: time.UTC,
    }, nil)
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close(ctx)

    log.Println("Ready with Ollama!")
}
```

---

## Core Concepts

### Episodes

Episodes are temporal data units representing events, conversations, documents, or any time-bound information:

```go
episodes := []types.Episode{
    {
        ID:        "meeting-001",
        Name:      "Weekly Team Standup",
        Content:   "Alice reported progress on the API. Bob mentioned database issues.",
        Reference: time.Now().Add(-2 * time.Hour), // When it happened
        CreatedAt: time.Now(),                      // When recorded
        GroupID:   "team-alpha",
        Metadata: map[string]interface{}{
            "meeting_type": "standup",
            "duration":     "30min",
        },
    },
}

_, err := client.Add(ctx, episodes, nil)
```

### Nodes

Nodes represent entities in your knowledge graph:
- **EntityNode**: People, places, concepts, objects
- **EpisodicNode**: Events, meetings, conversations
- **CommunityNode**: Groups of related entities

### Edges

Edges represent relationships between nodes:
- **EntityEdge**: "Alice works with Bob"
- **EpisodicEdge**: "Meeting occurred in Conference Room A"
- **CommunityEdge**: "Engineering Team includes Alice and Bob"

### Search

Hybrid search combining semantic similarity, keywords, and graph traversal:

```go
// Basic search
results, err := client.Search(ctx, "database connection issues", nil)

// Advanced search with configuration
searchConfig := &types.SearchConfig{
    Limit:              10,
    CenterNodeDistance: 3,
    MinScore:           0.1,
    IncludeEdges:       true,
    Rerank:             true,
}

results, err = client.Search(ctx, "API integration progress", searchConfig)

for _, node := range results.Nodes {
    fmt.Printf("Found: %s (%s)\n", node.Name, node.Type)
}
```

### Fact Store RAG Search (New)

For simpler RAG use cases that don't need graph traversal:

```go
// Configure fact store (PostgreSQL or DoltGres)
config := &predicato.Config{
    GroupID: "my-app",
    FactStoreConfig: &factstore.FactStoreConfig{
        Type:             factstore.FactStoreTypeDoltGres, // or FactStoreTypePostgres
        ConnectionString: "postgres://localhost:5432/facts",
    },
}

client, _ := predicato.NewClient(db, nlpClient, embedderClient, config, nil)

// Direct RAG search on facts (no graph queries)
results, err := client.SearchFacts(ctx, "API redesign", &types.SearchConfig{
    Limit: 10,
})

for i, node := range results.Nodes {
    fmt.Printf("%d. %s (score: %.2f)\n", i+1, node.Name, results.NodeScores[i])
}
```

---

## Configuration Options

### Embedder Configuration

```go
// Internal (local) — via go-candle
embedderClient, _ := candleAdapter.NewCandleEmbedderClient(&candleAdapter.CandleEmbedderConfig{
    Model:      "qwen/qwen3-embedding-0.6b",
    Dimensions: 1024,
})

// External (OpenAI)
embedderClient := embedder.NewOpenAIEmbedder(apiKey, embedder.Config{
    Model:      "text-embedding-3-large",
    Dimensions: 3072,
})

// External (Voyage)
embedderClient := embedder.NewVoyageEmbedder(&embedder.VoyageConfig{
    APIKey: voyageKey,
    Model:  "voyage-3",
})
```

### Search Configuration

```go
searchConfig := &types.SearchConfig{
    Limit:              20,    // Max results
    CenterNodeDistance: 2,     // Graph traversal depth
    MinScore:           0.0,   // Minimum relevance
    IncludeEdges:       true,  // Include relationships
    Rerank:             false, // Apply reranking
    NodeConfig: &types.NodeSearchConfig{
        SearchMethods: []string{"cosine_similarity", "bm25"},
    },
}
```

---

## Build Instructions

### Building Without CGO (Limited Features)

Some packages work without CGO:

```bash
go build ./pkg/factstore/...
go build ./pkg/embedder/...
go build ./pkg/nlp/...
go test ./pkg/factstore/...
```

### Building With CGO (Full Features)

```bash
# Download native libraries
go generate ./cgo.go

# Build with Ladybug support
go build -tags system_ladybug -o myapp .

# Or use Make
make build
```

See [AGENTS.md](../AGENTS.md) for detailed build instructions.

---

## Multi-tenancy

Use GroupID to isolate data:

```go
// User-specific
userConfig := &predicato.Config{
    GroupID: fmt.Sprintf("user-%s", userID),
}

// Organization-specific
orgConfig := &predicato.Config{
    GroupID: fmt.Sprintf("org-%s", orgID),
}
```

---

## Best Practices

### 1. Resource Management

Always close clients and drivers:

```go
defer client.Close(ctx)
defer db.Close(ctx)
defer embedderClient.Close()
```

### 2. Context Usage

Use context for timeouts:

```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

results, err := client.Search(ctx, query, nil)
```

### 3. Error Handling

```go
if err != nil {
    switch {
    case errors.Is(err, predicato.ErrNodeNotFound):
        // Handle missing node
    case errors.Is(err, predicato.ErrInvalidEpisode):
        // Handle invalid input
    default:
        log.Printf("Error: %v", err)
    }
}
```

### 4. Batch Processing

```go
const batchSize = 100

for i := 0; i < len(allEpisodes); i += batchSize {
    end := min(i+batchSize, len(allEpisodes))
    batch := allEpisodes[i:end]
    
    if _, err := client.Add(ctx, batch, nil); err != nil {
        log.Printf("Batch %d failed: %v", i/batchSize, err)
    }
}
```

---

## Next Steps

- Read the [API Reference](API_REFERENCE.md) for detailed documentation
- Check the [Examples](EXAMPLES.md) for more use cases
- See [FAQ](FAQ.md) for common questions
- Explore the [examples/](../examples/) directory

## Troubleshooting

### "cannot find -llbug"

The Ladybug native library hasn't been downloaded:
```bash
go generate ./cgo.go
```

### CGO-related errors

Ensure GCC is installed:
```bash
# Ubuntu/Debian
sudo apt install build-essential

# macOS
xcode-select --install
```

### Model download slow

First run downloads ~1.7GB of models. Use a fast connection or pre-download:
```bash
# Models are cached in ~/.cache/huggingface/
```

### Getting Help

- Check the [FAQ](FAQ.md)
- Open an issue on GitHub
- Review [AGENTS.md](../AGENTS.md) for build details
