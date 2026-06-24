//go:build cgo && system_ladybug

// copy-name-embeddings makes a generalization subgraph (G_g) self-sufficient for
// embedding-based name resolution: it copies Entity.name_embedding from a fully
// embedded source graph into the target G_g (joined by name — the G_g's canonical
// names are a subset of the source's) and builds an HNSW vector index on the target.
//
// The G_g is emitted without embeddings (the generalization builder carries names +
// relations only), so reasoning hosts otherwise have to open the large source graph
// just to resolve a name. After this runs, the reasoner can resolve AND reason over
// the same small graph. No re-embedding — the source vectors are reused verbatim, so
// they stay in the same embedding space as the query embedder.
//
// Idempotent: only entities whose name_embedding is empty/wrong-dim are written, and
// the index is (re)created at the end. Resumable — re-running resumes the remainder.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/soundprediction/predicato/pkg/driver"
)

func main() {
	srcPath := flag.String("src", "", "source graph with populated name_embedding (required)")
	dstPath := flag.String("dst", "", "target generalization graph to fill + index (required)")
	dims := flag.Int("dims", 1024, "embedding dimensions")
	chunkSz := flag.Int("chunk", 50000, "entities pulled per target scan")
	ckptEvery := flag.Int("checkpoint-every", 2000, "checkpoint every N writes")
	bufBytes := flag.Uint64("buffer-pool-bytes", 6<<30, "ladybug buffer pool size")
	skipIndex := flag.Bool("skip-index", false, "copy embeddings but do not create the vector index")
	flag.Parse()
	if *srcPath == "" || *dstPath == "" {
		log.Fatal("--src and --dst are required")
	}
	ctx := context.Background()

	index := loadSourceEmbeddings(ctx, *srcPath, *dims, *bufBytes)
	log.Printf("loaded %d source name embeddings", len(index))
	if len(index) == 0 {
		log.Fatal("source graph has no name embeddings")
	}

	dcfg := driver.DefaultLadybugDriverConfig()
	dcfg.DBPath = *dstPath
	dcfg.ReadOnly = false
	dcfg.BufferPoolSize = *bufBytes
	d, err := driver.NewLadybugDriverWithConfig(dcfg)
	if err != nil {
		log.Fatalf("open dst %s: %v", *dstPath, err)
	}
	defer d.Close()

	copyInto(ctx, d, index, *dims, *chunkSz, *ckptEvery)

	if !*skipIndex {
		createIndex(ctx, d)
	}
}

// loadSourceEmbeddings reads name -> name_embedding from the source graph.
func loadSourceEmbeddings(ctx context.Context, path string, dims int, buf uint64) map[string][]float32 {
	cfg := driver.DefaultLadybugDriverConfig()
	cfg.DBPath = path
	cfg.ReadOnly = true
	cfg.BufferPoolSize = buf
	s, err := driver.NewLadybugDriverWithConfig(cfg)
	if err != nil {
		log.Fatalf("open src %s: %v", path, err)
	}
	defer s.Close()

	rows := query(ctx, s, fmt.Sprintf(
		"MATCH (n:Entity) WHERE size(n.name_embedding) = %d RETURN n.name AS name, n.name_embedding AS emb", dims), nil)
	out := make(map[string][]float32, len(rows))
	for _, r := range rows {
		name, _ := r["name"].(string)
		if strings.TrimSpace(name) == "" {
			continue
		}
		vec := toFloat32(r["emb"])
		if len(vec) != dims {
			continue
		}
		out[strings.ToLower(strings.TrimSpace(name))] = vec
	}
	return out
}

func copyInto(ctx context.Context, d *driver.LadybugDriver, index map[string][]float32, dims, chunkSz, ckptEvery int) {
	miss := fmt.Sprintf("(n.name_embedding IS NULL OR size(n.name_embedding) <> %d)", dims)
	total := count(ctx, d, fmt.Sprintf("MATCH (n:Entity) WHERE %s RETURN count(n) AS c", miss))
	log.Printf("%d target entities without a %d-dim embedding", total, dims)
	selChunk := fmt.Sprintf("MATCH (n:Entity) WHERE %s RETURN n.uuid AS uuid, n.name AS name LIMIT %d", miss, chunkSz)
	setQ := "MATCH (n:Entity {uuid:$u}) SET n.name_embedding = $vec"

	written, matched := 0, 0
	start := time.Now()
	for {
		rows := query(ctx, d, selChunk, nil)
		if len(rows) == 0 {
			break
		}
		progressed := false
		for _, r := range rows {
			uuid, _ := r["uuid"].(string)
			name, _ := r["name"].(string)
			vec, ok := index[strings.ToLower(strings.TrimSpace(name))]
			if !ok {
				continue // no source vector for this name; leave empty (drops out next scan? no — keep marker)
			}
			matched++
			if _, _, _, err := d.ExecuteQuery(ctx, setQ, map[string]any{"u": uuid, "vec": vec}); err != nil {
				log.Printf("SET %s failed: %v", uuid, err)
				continue
			}
			written++
			progressed = true
			if written%ckptEvery == 0 {
				checkpoint(ctx, d)
				log.Printf("%d written  %d matched  %.0f/s", written, matched, float64(written)/time.Since(start).Seconds())
			}
		}
		checkpoint(ctx, d)
		// If a whole chunk produced no writes, the remaining rows are all unmatched —
		// they would re-appear forever (still empty), so stop.
		if !progressed {
			log.Printf("chunk had no source matches; %d entities have no source embedding", len(rows))
			break
		}
	}
	checkpoint(ctx, d)
	log.Printf("COPY DONE: %d embeddings written (%d names matched)", written, matched)
}

func createIndex(ctx context.Context, d *driver.LadybugDriver) {
	if _, _, _, err := d.ExecuteQuery(ctx, "LOAD EXTENSION 'vector'", nil); err != nil {
		log.Printf("LOAD vector warning: %v", err)
	}
	log.Printf("creating HNSW index name_emb_idx on Entity.name_embedding …")
	if _, _, _, err := d.ExecuteQuery(ctx, "CALL CREATE_VECTOR_INDEX('Entity', 'name_emb_idx', 'name_embedding')", nil); err != nil {
		log.Fatalf("CREATE_VECTOR_INDEX failed: %v", err)
	}
	checkpoint(ctx, d)
	log.Printf("INDEX CREATED")
}

func toFloat32(v any) []float32 {
	switch t := v.(type) {
	case []float32:
		return t
	case []float64:
		out := make([]float32, len(t))
		for i, f := range t {
			out[i] = float32(f)
		}
		return out
	case []any:
		out := make([]float32, 0, len(t))
		for _, e := range t {
			switch f := e.(type) {
			case float32:
				out = append(out, f)
			case float64:
				out = append(out, float32(f))
			default:
				return nil
			}
		}
		return out
	default:
		return nil
	}
}

func count(ctx context.Context, d *driver.LadybugDriver, q string) int {
	rows := query(ctx, d, q, nil)
	if len(rows) == 0 {
		return 0
	}
	switch v := rows[0]["c"].(type) {
	case int64:
		return int(v)
	case int:
		return v
	case float64:
		return int(v)
	}
	return 0
}

func query(ctx context.Context, d *driver.LadybugDriver, q string, params map[string]any) []map[string]any {
	res, _, _, err := d.ExecuteQuery(ctx, q, params)
	if err != nil {
		log.Fatalf("query failed: %v\n%s", err, q)
	}
	rows, ok := res.([]map[string]any)
	if !ok {
		log.Fatalf("unexpected result type %T", res)
	}
	return rows
}

func checkpoint(ctx context.Context, d *driver.LadybugDriver) {
	if _, _, _, err := d.ExecuteQuery(ctx, "CHECKPOINT", nil); err != nil {
		log.Printf("checkpoint warning: %v", err)
	}
}
