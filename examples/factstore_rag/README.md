# FactStore RAG Example

This example demonstrates using Predicato's factstore for RAG (Retrieval-Augmented Generation) applications without requiring a graph database.

## What This Example Shows

- Configuring a PostgreSQL factstore with VectorChord (production)
- Storing extracted knowledge (entities and relationships)
- Performing hybrid search (vector + keyword) with RRF fusion
- Building RAG context from search results

## Prerequisites

### For Production (PostgreSQL with VectorChord)

- PostgreSQL 15+
- [VectorChord extension](https://github.com/tensorchord/VectorChord)

```bash
# Quick start with Docker
docker run -d --name postgres-vectorchord \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  tensorchord/vchord-postgres:pg17-v0.2.1

# Create database
psql -h localhost -U postgres -c "CREATE DATABASE facts;"
```

### For Development (Demo Mode)

No external dependencies - the example runs in demo mode showing the API.

## Build & Run

### With PostgreSQL

```bash
# Set connection string
export FACTSTORE_POSTGRES_URL='postgres://postgres:password@localhost:5432/facts?sslmode=disable'

# Build and run
cd examples/factstore_rag
go build -o factstore_example .
./factstore_example
```

### Demo Mode (No Database)

```bash
cd examples/factstore_rag
go run .
```

## Expected Output

### With PostgreSQL

```
================================================================================
Predicato FactStore RAG Example
================================================================================

This example demonstrates:
  - Storing extracted knowledge in a factstore
  - Performing hybrid search (vector + keyword)
  - Using search results for RAG applications

[1/4] Connecting to PostgreSQL with VectorChord...
      Connected to PostgreSQL (VectorChord enabled)
[2/4] Initializing database schema...
      Schema initialized
[3/4] Storing extracted knowledge...
      Stored 4 nodes and 4 edges
[4/4] Performing hybrid search...

================================================================================
Search Query: "Who founded the company?"
================================================================================

Found 4 nodes:
----------------------------------------
  1. [0.850] Alice Smith (person)
     Co-founder and CEO of Acme Corporation

  2. [0.820] Bob Johnson (person)
     Co-founder and head of engineering at Acme Corporation

  3. [0.750] Acme Corporation (organization)
     A technology company specializing in cloud computing and AI solutions...

  4. [0.600] CloudAI (product)
     Flagship AI product with over 10,000 customers worldwide

Found 4 edges:
----------------------------------------
  1. [0.900] Alice Smith -[FOUNDED]-> Acme Corporation
     Alice Smith co-founded Acme Corporation in 2010

  2. [0.880] Bob Johnson -[FOUNDED]-> Acme Corporation
     Bob Johnson co-founded Acme Corporation in 2010

================================================================================
RAG Context Generation
================================================================================

Based on the search results, you can construct RAG context:

Relevant entities:
- Alice Smith (person): Co-founder and CEO of Acme Corporation
- Bob Johnson (person): Co-founder and head of engineering at Acme Corporation
- Acme Corporation (organization): A technology company...

Relationships:
- Alice Smith FOUNDED Acme Corporation: Alice Smith co-founded Acme Corporation in 2010
- Bob Johnson FOUNDED Acme Corporation: Bob Johnson co-founded Acme Corporation in 2010

================================================================================
Example completed successfully!
================================================================================
```

### Demo Mode

```
================================================================================
Predicato FactStore RAG Example
================================================================================

[1/4] Using in-memory factstore (no PostgreSQL configured)...
      Set FACTSTORE_POSTGRES_URL for production PostgreSQL backend
      Example: FACTSTORE_POSTGRES_URL='postgres://user:pass@localhost:5432/facts'

      (Skipping database operations - demo mode)

================================================================================
FactStore RAG API Overview (Demo Mode)
================================================================================

To use the factstore, you would:

1. Create a factstore connection:
   db, err := factstore.NewPostgresDB(connString, 1024)

2. Initialize the schema:
   db.Initialize(ctx)

3. Store extracted knowledge:
   db.SaveSource(ctx, source)
   db.SaveExtractedKnowledge(ctx, sourceID, nodes, edges)

4. Search with hybrid (vector + keyword) search:
   results, err := db.HybridSearch(ctx, query, embedding, config)

================================================================================
Demo completed!
================================================================================
```

## Code Overview

### FactStore Configuration

```go
// PostgreSQL with VectorChord
db, err := factstore.NewPostgresDB(connectionString, 1024)
```

### Storing Knowledge

```go
// Save source document
source := &factstore.Source{
    ID:      "doc-001",
    Name:    "Company Overview",
    Content: "Full document text...",
    GroupID: "my-tenant",
}
db.SaveSource(ctx, source)

// Save extracted entities and relationships
db.SaveExtractedKnowledge(ctx, source.ID, nodes, edges)
```

### Hybrid Search

```go
config := &factstore.FactSearchConfig{
    GroupID:  "my-tenant",
    Limit:    10,
    MinScore: 0.5,
    SearchMethods: []factstore.SearchMethod{
        factstore.VectorSearch,
        factstore.KeywordSearch,
    },
}

results, err := db.HybridSearch(ctx, query, queryEmbedding, config)
// results.Nodes - Matching entities
// results.Edges - Matching relationships
// results.NodeScores, results.EdgeScores - Relevance scores
```

### Building RAG Context

```go
context := "Relevant entities:\n"
for _, node := range results.Nodes {
    context += fmt.Sprintf("- %s (%s): %s\n", node.Name, node.Type, node.Description)
}
// Use context in LLM prompt
```

## Integration with Predicato Client

For full integration with the Predicato client:

```go
import "github.com/soundprediction/predicato"

// Configure factstore in Predicato config
config := &predicato.Config{
    GroupID: "my-group",
    FactStoreConfig: &factstore.FactStoreConfig{
        Type:                factstore.FactStoreTypePostgres,
        ConnectionString:    "postgres://...",
        EmbeddingDimensions: 1024,
    },
}

client, _ := predicato.NewClient(driver, nlp, embedder, config, nil)

// Use SearchFacts for integrated search
results, err := client.SearchFacts(ctx, "search query", searchConfig)
```

## Next Steps

- See [docs/FACTSTORE_RAG.md](../../docs/FACTSTORE_RAG.md) for detailed setup guide
- See `examples/basic/` for full Predicato client with graph database
- See `examples/external_apis/` for using OpenAI embeddings
