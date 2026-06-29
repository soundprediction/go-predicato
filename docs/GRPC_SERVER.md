# gRPC Server Mode

Predicato can run as a standalone **gRPC server** so the knowledge graph executes
in a separate process — or on a different machine — from the services that query
it (e.g. humn). This is the building block for scaling the graph independently of
the API tier (a pool of read replicas behind one logical address).

For Go service-to-service use; the existing HTTP/MCP server is unchanged and
remains the path for non-Go clients.

## Run it

```bash
# same graph / NLP / embedder configuration as the HTTP `server` command
predicato serve-grpc --addr :50071
```

Configuration comes from the config file, environment, or flags (`--db-driver`,
`--db-uri`, `--nlp-*`, `--embedding-*`, `--reranker-*`, `--gliner2*`, …)
exactly like `server`.

For graph-result cross-encoder reranking, start every predicato worker with a
Jina-compatible `/v1/rerank` service:

```bash
predicato serve-grpc \
  --addr :50072 \
  --db-driver ladybug \
  --db-uri /data/topic-graphs/ladybug/canonical.ladybug \
  --embedding-provider openai \
  --embedding-base-url http://GPU_PRIVATE_IP:9099/v1 \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-api-key "$EMBEDDING_API_KEY" \
  --reranker-provider reranker \
  --reranker-base-url http://GPU_PRIVATE_IP:9099/v1 \
  --reranker-model zhiqing/Qwen3-Reranker-0.6B-ONNX
```

Equivalent environment variables are `RERANKER_PROVIDER`,
`RERANKER_BASE_URL`, `RERANKER_MODEL`, and `RERANKER_API_KEY`.

## What it exposes

The service `predicato.v1.GraphService` (`proto/predicato/v1/graph.proto`):

| RPC | Maps to |
|-----|---------|
| `Search` | `Predicato.Search` — hybrid search |
| `AddEpisode` | `Predicato.AddEpisode` — ingest one episode |
| `GetNode` / `GetEdge` | fetch one element by uuid |
| `GetStats` | graph statistics (when the engine supports it) |
| `Health` | liveness probe (LB readiness) |

The wire messages mirror predicato's Go types (`Node`, `Edge`, `SearchConfig`,
`SearchResults`, `Episode`). **Embedding vectors are intentionally omitted** from
the wire — callers use names/facts/types/sources, not the vectors — keeping
payloads small. Ingest options that aren't serializable (a `GraphModeler`, error
policy) are not remoted.

## Use it from Go

```go
import grpcsvc "github.com/soundprediction/predicato/pkg/grpcsvc"

cli, _ := grpcsvc.Dial("graph-host:50071")     // see Pools below for multi-host
defer cli.Close()
res, _ := cli.Search(ctx, "fatigue, cold intolerance", &types.SearchConfig{Limit: 8})
```

`grpcsvc.Client` presents predicato's Go types (not the generated protos), so it
drops into code that already calls `Search`/`AddEpisode`.

- **Server side:** `grpcsvc.NewServer(p).Register(grpcServer)` wraps any
  `predicato.Predicato`; `grpcsvc.Serve(p, addr, opts...)` is the one-call helper.

## Pools (horizontal scaling)

`Dial` load-balances (round-robin) across a **pool** of graph servers. The target
may be:

- a single `host:port`;
- a comma-separated **static** pool `h1:50071,h2:50071,…`;
- a scheme target like `dns:///predicato.svc:50071` — a **dynamic** pool a DNS /
  headless service resolves to all instance IPs, re-resolved as the pool
  autoscales.

gRPC's default `pick_first` would pin to one instance; `Dial` adds the
`round_robin` service config so traffic spreads across the pool.

**Statelessness:** the graph is read-mostly for the DDx use case, so serve `Search`
from a pool of read replicas (each instance with the same read-only graph) and
route `AddEpisode` (writes) to a single writer/primary that replicates out.

## Regenerating the stubs

```bash
# requires protoc, protoc-gen-go, protoc-gen-go-grpc on PATH
./proto/generate.sh
```

## Wiring from humn

```toml
[predicato]
grpc_endpoint = "dns:///predicato.svc:50071"   # pool; or "host:50071", or "h1:port,h2:port"
search_node_reranker = "cross_encoder"         # asks predicato to use its configured reranker
```

When set, humn routes its knowledge-graph queries to the remote pool instead of an
in-process embedded graph.
