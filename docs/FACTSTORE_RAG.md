# FactStore RAG (Retrieval-Augmented Generation) Guide

This guide covers setting up and using the factstore for RAG workloads with Predicato.

## Overview

The factstore provides persistent storage for extracted facts and entities, supporting hybrid search that combines:
- **Vector similarity search** using embeddings
- **Keyword/fulltext search** 
- **Reciprocal Rank Fusion (RRF)** for result merging

## Backend Options

### 1. PostgreSQL with VectorChord (Recommended for Production)

Best performance for vector search using native PostgreSQL vector operations with VectorChord's optimized indexing.

**Requirements:**
- PostgreSQL 15+ 
- [VectorChord extension](https://github.com/tensorchord/VectorChord)

**Installation:**

```bash
# Using Docker (recommended)
docker run -d --name postgres-vectorchord \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  tensorchord/vchord-postgres:pg17-v0.2.1

# Or install from source - see VectorChord documentation
# https://github.com/tensorchord/VectorChord#installation
```

**Enable Extension:**

```sql
-- Connect to your database and enable VectorChord
CREATE EXTENSION IF NOT EXISTS vchord CASCADE;
-- Note: vchord depends on the vector extension which will be created automatically
```

**Usage:**

```go
import "github.com/soundprediction/predicato/pkg/factstore"

config := &factstore.FactStoreConfig{
    Type:                factstore.FactStoreTypePostgres,
    ConnectionString:    "postgres://user:pass@localhost:5432/facts?sslmode=disable",
    EmbeddingDimensions: 1024, // Match your embedding model
}

db, err := factstore.NewFactsDB(config)
if err != nil {
    return err
}
defer db.Close()

// Initialize tables and indexes
if err := db.Initialize(ctx); err != nil {
    return err
}
```

### 2. DoltGres (Development/Testing)

Uses in-memory vector search. Good for development but not recommended for production with large datasets.

**Note:** DoltGres does not support the VectorChord extension. Vector operations are performed in-memory using Go's cosine similarity calculations.

**Usage:**

```go
config := &factstore.FactStoreConfig{
    Type:                factstore.FactStoreTypeDoltGres,
    ConnectionString:    "postgres://root:@localhost:3307/facts",
    EmbeddingDimensions: 1024,
}

db, err := factstore.NewFactsDB(config)
```

## Migration from DoltGres to PostgreSQL

If you're upgrading from DoltGres to PostgreSQL with VectorChord:

### Step 1: Export Data from DoltGres

```go
// Connect to DoltGres
doltDB, _ := factstore.NewDoltGresFactsDB(doltConnString, 1024)

// Get all data
sources, _ := doltDB.GetAllSources(ctx, 0)
nodes, _ := doltDB.GetAllNodes(ctx, 0)
triples, _ := doltDB.GetAllTriples(ctx, 0)
```

### Step 2: Import to PostgreSQL

```go
// Connect to PostgreSQL with VectorChord
pgDB, _ := factstore.NewPostgresDB(pgConnString, 1024)
pgDB.Initialize(ctx)

// Import sources
for _, source := range sources {
    pgDB.SaveSource(ctx, source)
}

// Import nodes and triples (grouped by source)
sourceNodes := groupNodesBySource(nodes)
sourceTriples := groupTriplesBySource(triples)

for sourceID, nodeList := range sourceNodes {
    tripleList := sourceTriples[sourceID]
    pgDB.SaveExtractedKnowledge(ctx, sourceID, nodeList, tripleList)
}
```

### Step 3: Rebuild Vector Indexes

After importing data, rebuild the vector indexes for optimal performance:

```sql
-- For PostgreSQL with VectorChord
-- Drop and recreate the VChordrq index (optimized for high-dimensional vectors)
DROP INDEX IF EXISTS extracted_nodes_embedding_idx;
CREATE INDEX extracted_nodes_embedding_idx 
ON extracted_nodes USING vchordrq (embedding vector_cosine_ops);

DROP INDEX IF EXISTS extracted_triples_embedding_idx;
CREATE INDEX extracted_triples_embedding_idx
ON extracted_triples USING vchordrq (embedding vector_cosine_ops);

-- Analyze tables for query optimization
ANALYZE extracted_nodes;
ANALYZE extracted_triples;
ANALYZE sources;
```

## Search Configuration

### Basic Search

```go
config := &factstore.FactSearchConfig{
    GroupID:   "my-tenant",
    Limit:     10,
    MinScore:  0.7,  // Minimum similarity threshold (0.0-1.0)
    NodeTypes: []string{"person", "organization"},
}

// Vector search with embedding
nodes, scores, err := db.SearchNodes(ctx, "", embedding, config)

// Keyword search
nodes, scores, err := db.SearchNodes(ctx, "search query", nil, config)

// Hybrid search (recommended)
results, err := db.HybridSearch(ctx, "search query", embedding, config)
```

### Time-Range Filtering

```go
config := &factstore.FactSearchConfig{
    TimeRange: &factstore.TimeRange{
        Start: time.Now().AddDate(0, -1, 0), // Last month
        End:   time.Now(),
    },
}
```

### Search Methods

Control which search methods to use:

```go
config := &factstore.FactSearchConfig{
    SearchMethods: []factstore.SearchMethod{
        factstore.VectorSearch,   // Embedding-based similarity
        factstore.KeywordSearch,  // Full-text search
    },
}
```

## Performance Considerations

### VectorChord vs In-Memory Search

| Aspect | VectorChord | In-Memory (DoltGres) |
|--------|-------------|---------------------|
| Search Speed | O(log n) with VChordrq | O(n) scan |
| Memory Usage | Index-based | Loads all embeddings |
| Max Results | Unlimited | 10,000 (configurable) |
| Recommended For | Production | Development/Testing |

### In-Memory Search Limits

When using DoltGres (no VectorChord), in-memory search is limited to prevent excessive memory usage:

```go
// Maximum results processed in-memory (defined in postgres.go)
const MaxInMemorySearchResults = 10000
```

A warning is logged when this limit is reached:
```
WARNING: In-memory vector search hit limit of 10000 results. 
Consider using VectorChord for better performance.
```

### Index Tuning for VectorChord

For large datasets, tune the VChordrq index parameters:

```sql
-- Create VChordrq index with custom parameters
CREATE INDEX extracted_nodes_embedding_idx 
ON extracted_nodes 
USING vchordrq (embedding vector_cosine_ops)
WITH (residual_quantization = true);

-- For very large datasets, use IVF parameters
CREATE INDEX extracted_nodes_embedding_idx 
ON extracted_nodes 
USING vchordrq (embedding vector_cosine_ops)
WITH (lists = 100);  -- Adjust based on dataset size
```

### Connection Pooling

The factstore configures connection pooling automatically:

```go
// Default settings (configurable via MaxConnections)
db.SetMaxOpenConns(25)
db.SetMaxIdleConns(5)
db.SetConnMaxLifetime(5 * time.Minute)
```

## Embedding Model Compatibility

The `EmbeddingDimensions` config must match your embedding model:

| Model | Dimensions |
|-------|------------|
| qwen3-embedding (default) | 1024 |
| text-embedding-3-small | 1536 |
| text-embedding-3-large | 3072 |
| voyage-3 | 1024 |
| jina-embeddings-v3 | 1024 |

```go
config := &factstore.FactStoreConfig{
    EmbeddingDimensions: 1536,  // For OpenAI text-embedding-3-small
}
```

## Schema Reference

### Sources Table

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(255) | Primary key |
| name | TEXT | Source name/title |
| content | TEXT | Full source content |
| group_id | VARCHAR(255) | Multi-tenant group ID |
| metadata | JSONB | Additional metadata |
| created_at | TIMESTAMP | Creation timestamp |

### Extracted Nodes Table

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(255) | Primary key |
| source_id | VARCHAR(255) | FK to sources |
| group_id | VARCHAR(255) | Multi-tenant group ID |
| name | TEXT | Entity name |
| type | VARCHAR(50) | Entity type (person, org, etc.) |
| description | TEXT | Entity description |
| embedding | vector(N)/JSONB | Vector embedding |
| chunk_index | INT | Position in source |
| created_at | TIMESTAMP | Creation timestamp |

### Extracted Triples Table

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(255) | Primary key |
| source_id | VARCHAR(255) | FK to sources |
| group_id | VARCHAR(255) | Multi-tenant group ID |
| subject | TEXT | Subject entity name |
| subject_type | VARCHAR(50) | Subject entity type |
| predicate | TEXT | Relationship/predicate |
| object | TEXT | Object entity name |
| object_type | VARCHAR(50) | Object entity type |
| description | TEXT | Triple description |
| condition | TEXT | Conditional context (extended extraction) |
| temporal | TEXT | Temporal context (extended extraction) |
| location | TEXT | Location context (extended extraction) |
| certainty | TEXT | Certainty level (extended extraction) |
| scope | TEXT | Scope qualifier (extended extraction) |
| source_attribution | TEXT | Source attribution (extended extraction) |
| confidence | FLOAT | Extraction confidence score |
| embedding | vector(N)/JSONB | Vector embedding |
| chunk_index | INT | Position in source |
| model | VARCHAR(255) | Model used for extraction |
| created_at | TIMESTAMP | Creation timestamp |

### Extracted Rules Table

| Column | Type | Description |
|--------|------|-------------|
| id | VARCHAR(255) | Primary key |
| source_id | VARCHAR(255) | FK to sources |
| antecedent | TEXT | IF condition |
| consequent | TEXT | THEN result |
| exception | TEXT | UNLESS exception (optional) |
| rule_type | TEXT | Type of rule |
| scope | TEXT | Scope qualifier |
| source_attribution | TEXT | Source attribution |
| confidence | FLOAT | Extraction confidence score |
| embedding | vector(N)/JSONB | Vector embedding |
| chunk_index | INT | Position in source |
| model | VARCHAR(255) | Model used for extraction |
| created_at | TIMESTAMP | Creation timestamp |

For detailed documentation on the extended extraction fields (condition, temporal, location, etc.) and conditional rules, see [Extended Fact Storage](EXTENDED_FACT_STORAGE.md).

## Troubleshooting

### "could not access file 'vchord'" Error

The VectorChord extension is not installed. See installation instructions above.

### Slow Vector Search with DoltGres

This is expected behavior. Migrate to PostgreSQL with VectorChord for production workloads.

### "embedding dimensions mismatch" Error

Ensure `EmbeddingDimensions` matches your embedding model output:

```go
// Check your embedding model output
embedding := embedder.Embed(ctx, "test")
fmt.Printf("Dimensions: %d\n", len(embedding))
```

### Memory Issues with Large Datasets

If using DoltGres with large datasets:
1. Reduce search limits
2. Add filters (GroupID, TimeRange, NodeTypes)
3. Migrate to PostgreSQL with VectorChord

## See Also

- [GETTING_STARTED.md](./GETTING_STARTED.md) - Quick start guide
- [EXTENDED_FACT_STORAGE.md](./EXTENDED_FACT_STORAGE.md) - Extended fact storage (contextual triples & conditional rules)
- [API_REFERENCE.md](./API_REFERENCE.md) - Full API documentation
- [pkg/factstore/doc.go](../pkg/factstore/doc.go) - Package documentation
