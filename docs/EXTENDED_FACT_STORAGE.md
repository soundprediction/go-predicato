use# Extended Fact Storage

Detailed documentation for Predicato's extended fact storage scheme: contextual knowledge triples and conditional rules.

## Overview

Predicato's **extended fact storage** enriches the standard extraction pipeline with contextual metadata on knowledge triples and conditional rules. Where standard extraction produces bare `(subject, predicate, object)` triples, extended extraction adds qualifiers — *when* a fact is true, *where* it applies, *under what conditions*, and *how certain* it is — plus conditional rules expressed as IF-THEN-UNLESS patterns.

Extended extraction is optional and additive. Standard extraction always runs first; extended extraction enriches the results in a second pass.

## Architecture

Extended fact storage sits within Predicato's two-layer design:

```
Episode (document/conversation)
    │
    ▼
┌──────────────────────────────────────────────┐
│  Standard Extraction (always runs)            │
│  • Entities (ExtractedNode)                   │
│  • Relationships (ExtractedTriple)            │
│  • Embeddings                                 │
└──────────────────────────────────────────────┘
    │
    ▼  (if ExtendedExtraction: true)
┌──────────────────────────────────────────────┐
│  Extended Extraction (second pass)            │
│  • Contextual fields on triples               │
│    (condition, temporal, location,             │
│     certainty, scope, source_attribution)      │
│  • Conditional rules (IF-THEN-UNLESS)          │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  Fact Store (PostgreSQL/DoltGres)             │
│  • extracted_nodes table                      │
│  • extracted_triples table (with context)     │
│  • extracted_rules table                      │
│  • sources table                              │
└──────────────────────────────────────────────┘
    │
    ▼  (PromoteToGraph)
┌──────────────────────────────────────────────┐
│  Knowledge Graph (Ladybug/Neo4j/Memgraph)     │
│  • Resolved entities                          │
│  • Temporal relationships                     │
│  • Communities                                │
└──────────────────────────────────────────────┘
```

## Knowledge Triples

Every relationship in Predicato is stored as an `ExtractedTriple` — a unified record combining entity-relationship data with contextual fields.

### The ExtractedTriple Type

Defined in `pkg/types/extraction.go`:

```go
type ExtractedTriple struct {
    ID                string    // Unique identifier
    SourceID          string    // FK to sources table
    GroupID           string    // Multi-tenant group ID
    Subject           string    // Subject entity name
    SubjectType       string    // Subject entity type (e.g., "Drug", "Person")
    Predicate         string    // Relationship (e.g., "treats", "works_at")
    Object            string    // Object entity name
    ObjectType        string    // Object entity type (e.g., "Disease", "Organization")
    Description       string    // Human-readable description
    Condition         string    // Under what conditions this is true
    Temporal          string    // When this is true (e.g., "ongoing", "2020-2023")
    Location          string    // Where this applies
    Certainty         string    // How certain (e.g., "established", "hypothetical")
    Scope             string    // Who/what this applies to (e.g., "adults", "EU markets")
    SourceAttribution string    // Where this information came from
    Model             string    // Model used for extraction
    Embedding         []float32 // Vector embedding
    Confidence        float64   // Extraction confidence score (0.0-1.0)
    ChunkIndex        int       // Position in source document
    CreatedAt         time.Time // Creation timestamp
}
```

### Standard vs Extended Fields

| Field | Standard Extraction | Extended Extraction |
|-------|:---:|:---:|
| Subject, Predicate, Object | Yes | Yes |
| SubjectType, ObjectType | Yes | Yes |
| Description | Yes | Yes |
| Embedding, Confidence | Yes | Yes |
| **Condition** | - | Yes |
| **Temporal** | - | Yes |
| **Location** | - | Yes |
| **Certainty** | - | Yes |
| **Scope** | - | Yes |
| **SourceAttribution** | - | Yes |

All fields are stored as flat columns — no nested JSON. This makes them directly queryable via SQL and searchable via the hybrid search pipeline.

### Example

```
Subject:     "Lisinopril"         SubjectType: "Drug"
Predicate:   "treats"
Object:      "Hypertension"       ObjectType:  "Disease"

Context (extended):
  Condition:    "when first-line therapy is appropriate"
  Temporal:     "ongoing"
  Certainty:    "established"
  Scope:        "adults"
  Source:       "StatPearls Hypertension chapter"
  Confidence:   0.95
```

## Conditional Rules

Extended extraction also produces **conditional rules** — structured IF-THEN-UNLESS patterns that capture domain logic.

### The ExtractedRule Type

Defined in `pkg/types/extraction.go`:

```go
type ExtractedRule struct {
    ID                string    // Unique identifier
    SourceID          string    // FK to sources table
    Antecedent        string    // IF condition
    Consequent        string    // THEN result
    Exception         string    // UNLESS exception (optional)
    RuleType          string    // Type of rule (e.g., "clinical", "policy")
    Scope             string    // Who/what this rule applies to
    SourceAttribution string    // Where this rule came from
    Model             string    // Model used for extraction
    Embedding         []float32 // Vector embedding
    Confidence        float64   // Extraction confidence score (0.0-1.0)
    ChunkIndex        int       // Position in source document
    CreatedAt         time.Time // Creation timestamp
}
```

### Example

```
Antecedent:  "patient has hypertension AND is not pregnant"
Consequent:  "prescribe ACE inhibitor as first-line therapy"
Exception:   "history of angioedema or bilateral renal artery stenosis"
RuleType:    "clinical"
Scope:       "adults"
Confidence:  0.92
```

## Enabling Extended Extraction

Enable extended extraction by setting `ExtendedExtraction: true` in `AddEpisodeOptions`:

```go
_, err := client.Add(ctx, episodes, &predicato.AddEpisodeOptions{
    ExtendedExtraction: true,
})
```

### Requirements

Extended extraction requires an NLP client that supports the `TaskExtendedExtraction` capability. Currently this is provided by:

- **GLiNER2** (`pkg/gliner2/`) — the primary extended extraction provider
- Any NLP client implementing `ExtractExtended(ctx, text, entityTypes, relationTypes)`

If `ExtendedExtraction: true` is set but the configured NLP client does not support it, Predicato logs a warning and falls back to standard extraction.

## The Extraction Pipeline

When `ExtendedExtraction: true`, the pipeline runs as follows:

1. **Chunk** — Episode content is split into chunks
2. **Standard extraction** — Each chunk is processed for entities and relationships
   - Entities are extracted (via GLiNER or NLP prompts) → `ExtractedNode`
   - Relationships are extracted → `ExtractedTriple` (standard fields only)
3. **Embed** — Embeddings are generated for nodes and triples
4. **Extended extraction** — Each chunk gets a second pass:
   - Existing triples are enriched with context fields (condition, temporal, location, certainty, scope, source_attribution)
   - New contextual triples may be extracted
   - Conditional rules are extracted → `ExtractedRule`
5. **Embed rules** — Embeddings are generated for rules
6. **Persist** — All results are saved to the fact store:
   - `SaveExtractedKnowledge(sourceID, nodes, triples)`
   - `SaveExtractedRules(sourceID, rules)`

The key code path is in `ingestion_factstore.go`:
```
ExtractToFacts()
  → chunk content
  → extract entities per chunk
  → extract relationships per chunk → ExtractedTriple
  → embed nodes and triples
  → if ExtendedExtraction:
      → ExtractExtended() per chunk
      → enrich triples with context
      → extract rules
      → embed rules
  → SaveExtractedKnowledge()
  → SaveExtractedRules()
```

## Database Schema

### `extracted_triples` Table

```sql
CREATE TABLE extracted_triples (
    id                VARCHAR(255) PRIMARY KEY,
    source_id         VARCHAR(255) REFERENCES sources(id),
    group_id          VARCHAR(255),
    subject           TEXT,
    subject_type      VARCHAR(50),
    predicate         TEXT,
    object            TEXT,
    object_type       VARCHAR(50),
    description       TEXT,
    condition         TEXT,
    temporal          TEXT,
    location          TEXT,
    certainty         TEXT,
    scope             TEXT,
    source_attribution TEXT,
    confidence        FLOAT,
    embedding         vector(1024),
    chunk_index       INT,
    model             VARCHAR(255),
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Indexes:
- `idx_triples_source` — on `source_id`
- `idx_triples_group` — on `group_id`
- `idx_triples_fts` — GIN full-text search on `subject || predicate || object || description`

### `extracted_rules` Table

```sql
CREATE TABLE extracted_rules (
    id                VARCHAR(255) PRIMARY KEY,
    source_id         VARCHAR(255) REFERENCES sources(id),
    antecedent        TEXT,
    consequent        TEXT,
    exception         TEXT,
    rule_type         TEXT,
    scope             TEXT,
    source_attribution TEXT,
    confidence        FLOAT,
    embedding         vector(1024),
    chunk_index       INT,
    model             VARCHAR(255),
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## FactsDB Interface

The `FactsDB` interface (defined in `pkg/factstore/facts.go`) provides the storage methods for extended facts:

```go
type FactsDB interface {
    // Standard methods
    SaveExtractedKnowledge(ctx, sourceID, nodes, triples) error
    GetExtractedTriples(ctx, sourceID) ([]*ExtractedTriple, error)

    // Extended extraction methods
    SaveExtractedRules(ctx, sourceID, rules) error
    GetExtractedRules(ctx, sourceID) ([]*ExtractedRule, error)

    // Search methods
    SearchTriples(ctx, query, embedding, config) ([]*ExtractedTriple, []float64, error)
    HybridSearch(ctx, query, embedding, config) (*FactSearchResults, error)
}
```

All implementations (PostgreSQL, DoltGres, MySQL, SQLite, DuckDB) support these methods.

## Searching Extended Facts

### Hybrid Search

The `HybridSearch` method returns both nodes and triples, with relevance scores:

```go
results, err := factsDB.HybridSearch(ctx, "hypertension treatment", embedding, &factstore.FactSearchConfig{
    GroupID:  "medical-kb",
    Limit:    20,
    MinScore: 0.5,
})

for i, triple := range results.Triples {
    fmt.Printf("Triple: %s %s %s (score: %.2f)\n",
        triple.Subject, triple.Predicate, triple.Object,
        results.TripleScores[i])
    if triple.Condition != "" {
        fmt.Printf("  Condition: %s\n", triple.Condition)
    }
    if triple.Temporal != "" {
        fmt.Printf("  Temporal: %s\n", triple.Temporal)
    }
}
```

### Triple Search

Search triples directly with vector and/or keyword search:

```go
triples, scores, err := factsDB.SearchTriples(ctx, "treats hypertension", embedding, &factstore.FactSearchConfig{
    GroupID:  "medical-kb",
    Limit:    10,
    MinScore: 0.7,
})
```

### Time-Range Filtering

Filter by when facts were extracted:

```go
config := &factstore.FactSearchConfig{
    TimeRange: &factstore.TimeRange{
        Start: time.Now().AddDate(0, -1, 0), // Last month
        End:   time.Now(),
    },
}
```

### SQL Queries on Context Fields

Because context fields are flat columns, you can query them directly in SQL:

```sql
-- Find all triples with temporal constraints
SELECT subject, predicate, object, temporal, certainty
FROM extracted_triples
WHERE temporal IS NOT NULL AND temporal != ''
  AND group_id = 'medical-kb';

-- Find high-certainty drug-disease relationships
SELECT subject, predicate, object, certainty, confidence
FROM extracted_triples
WHERE certainty = 'established'
  AND subject_type = 'Drug'
  AND confidence > 0.8;
```

## Graph Promotion

Facts flow from the Fact Store to the Knowledge Graph via `PromoteToGraph()`:

```
Fact Store (extracted_triples, extracted_rules)
    │
    ▼  PromoteToGraph()
GraphModeler
    ├── ResolveEntities()      — merge duplicate entities
    ├── ResolveRelationships() — merge duplicate relationships
    └── BuildCommunities()     — detect entity clusters
    │
    ▼
Knowledge Graph (Ladybug/Neo4j/Memgraph)
```

Promotion reads from the fact store and writes resolved entities and relationships to the graph database. The contextual fields on triples are preserved through promotion, so graph edges retain their condition, temporal, location, and other qualifiers.

```go
// Extract only (no graph promotion)
_, err := client.Add(ctx, episodes, &predicato.AddEpisodeOptions{
    ExtractOnly:        true,
    ExtendedExtraction: true,
})

// Later: promote to graph
err = client.PromoteToGraph(ctx, sourceID)
```

## See Also

- [FACTSTORE_RAG.md](./FACTSTORE_RAG.md) — Backend setup (PostgreSQL/DoltGres), search configuration, performance tuning
- [GETTING_STARTED.md](./GETTING_STARTED.md) — Quick start guide
- `pkg/types/extraction.go` — Type definitions for ExtractedTriple, ExtractedRule
- `pkg/factstore/facts.go` — FactsDB interface definition
- `ingestion_factstore.go` — Extraction pipeline implementation
