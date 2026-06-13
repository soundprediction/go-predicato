//go:build cgo && system_ladybug

// backfill-embeddings backfills fact_embedding on RelatesToNode_ edges (and
// optionally name_embedding on Entity nodes) in a ladybug graph, in place.
//
// Read strategy: ladybug's primary-key index is a hash index (no range scans),
// so a keyset cursor would force a full table scan per batch. Instead we read in
// large chunks with a plain LIMIT over the "missing embedding" filter; embedded
// rows drop out of the filter so successive chunks advance. Within a chunk,
// embedding runs in parallel across one or more endpoints while the (single)
// ladybug connection writes results back via point-lookup SET (uuid = hash idx).
//
// Robustness: bad batches (endpoint error or vector-count mismatch after
// retries) are skipped and logged, never fatal — a later resumable run picks
// them up. Idempotent + resumable: only NULL / wrong-dimension rows are touched.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/embedder"
)

type spec struct{ label, table, textProp, embProp string }

type job struct {
	uuids []string
	texts []string
}
type result struct {
	uuids []string
	vecs  [][]float32
	err   error
}

func main() {
	dbPath := flag.String("db", "", "path to .ladybug database (required)")
	target := flag.String("target", "edges", "edges | nodes | both")
	endpoints := flag.String("endpoints", "http://localhost:13305/api/v1", "comma-separated OpenAI-compatible embeddings base URLs")
	model := flag.String("model", "Qwen3-Embedding-0.6B-GGUF", "embedding model id")
	dims := flag.Int("dims", 1024, "embedding dimensions")
	batch := flag.Int("batch", 256, "embedding request size")
	chunkSz := flag.Int("chunk", 100000, "rows pulled per DB scan")
	workersPer := flag.Int("workers-per-endpoint", 2, "concurrent embed workers per endpoint")
	bufBytes := flag.Uint64("buffer-pool-bytes", 16<<30, "ladybug buffer pool size in bytes")
	ckptEvery := flag.Int("checkpoint-every", 40, "checkpoint every N written sub-batches")
	maxChunks := flag.Int("max-chunks", 0, "stop after N chunks (0=unlimited; for sampling)")
	flag.Parse()
	if *dbPath == "" {
		log.Fatal("--db required")
	}
	ctx := context.Background()

	var urls []string
	for _, u := range strings.Split(*endpoints, ",") {
		if u = strings.TrimSpace(u); u != "" {
			urls = append(urls, u)
		}
	}
	embs := make([]*embedder.OpenAIEmbedder, len(urls))
	for i, u := range urls {
		embs[i] = embedder.NewOpenAIEmbedder("local-no-key", embedder.Config{Model: *model, BaseURL: u, Dimensions: *dims})
		ok := false
		for a := 0; a < 5 && !ok; a++ {
			if v, err := embs[i].EmbedSingle(ctx, "connectivity probe"); err == nil && len(v) == *dims {
				ok = true
			} else {
				time.Sleep(time.Duration(a+1) * 2 * time.Second)
			}
		}
		if !ok {
			log.Fatalf("endpoint %s did not return a %d-dim embedding after retries", u, *dims)
		}
		log.Printf("endpoint OK: %s", u)
	}

	cfg := driver.DefaultLadybugDriverConfig()
	cfg.DBPath = *dbPath
	cfg.ReadOnly = false
	cfg.BufferPoolSize = *bufBytes
	d, err := driver.NewLadybugDriverWithConfig(cfg)
	if err != nil {
		log.Fatalf("open %s: %v", *dbPath, err)
	}
	defer d.Close()

	var specs []spec
	if *target == "edges" || *target == "both" {
		specs = append(specs, spec{"edges", "RelatesToNode_", "fact", "fact_embedding"})
	}
	if *target == "nodes" || *target == "both" {
		specs = append(specs, spec{"nodes", "Entity", "name", "name_embedding"})
	}
	for _, s := range specs {
		backfill(ctx, d, embs, s, *dims, *batch, *chunkSz, *workersPer, *ckptEvery, *maxChunks)
	}
}

func backfill(ctx context.Context, d *driver.LadybugDriver, embs []*embedder.OpenAIEmbedder, s spec, dims, batch, chunkSz, workersPer, ckptEvery, maxChunks int) {
	miss := fmt.Sprintf("(r.%s IS NULL OR size(r.%s) <> %d)", s.embProp, s.embProp, dims)
	total := count(ctx, d, fmt.Sprintf("MATCH (r:%s) WHERE %s RETURN count(r) AS c", s.table, miss))
	log.Printf("[%s] %d rows missing %d-dim %s", s.label, total, dims, s.embProp)
	if total == 0 {
		return
	}
	selChunk := fmt.Sprintf("MATCH (r:%s) WHERE %s RETURN r.uuid AS uuid, r.%s AS text LIMIT %d", s.table, miss, s.textProp, chunkSz)
	setQ := fmt.Sprintf("MATCH (r:%s {uuid:$u}) SET r.%s = $vec", s.table, s.embProp)

	nWorkers := workersPer * len(embs)
	jobCh := make(chan job, nWorkers*2)
	resCh := make(chan result, nWorkers*2)
	for w := 0; w < nWorkers; w++ {
		e := embs[w%len(embs)]
		go func(e *embedder.OpenAIEmbedder) {
			for j := range jobCh {
				var vecs [][]float32
				var err error
				for attempt := 0; attempt < 4; attempt++ {
					vecs, err = e.Embed(ctx, j.texts)
					if err == nil && len(vecs) == len(j.texts) {
						break
					}
					if err == nil {
						err = fmt.Errorf("embed returned %d vecs for %d texts", len(vecs), len(j.texts))
					}
					time.Sleep(time.Duration(attempt+1) * time.Second)
				}
				resCh <- result{j.uuids, vecs, err}
			}
		}(e)
	}

	var feedWg sync.WaitGroup
	done, skipped, written := 0, 0, 0
	start := time.Now()
	for c := 0; maxChunks == 0 || c < maxChunks; c++ {
		rows := query(ctx, d, selChunk, nil)
		if len(rows) == 0 {
			break
		}
		var subs []job
		for i := 0; i < len(rows); i += batch {
			end := i + batch
			if end > len(rows) {
				end = len(rows)
			}
			j := job{uuids: make([]string, 0, end-i), texts: make([]string, 0, end-i)}
			for _, r := range rows[i:end] {
				u, _ := r["uuid"].(string)
				t, _ := r["text"].(string)
				j.uuids = append(j.uuids, u)
				j.texts = append(j.texts, t)
			}
			subs = append(subs, j)
		}
		feedWg.Add(1)
		go func(subs []job) {
			defer feedWg.Done()
			for _, j := range subs {
				jobCh <- j
			}
		}(subs)

		for r := 0; r < len(subs); r++ {
			res := <-resCh
			if res.err != nil || len(res.vecs) != len(res.uuids) {
				log.Printf("[%s] skipping batch of %d (will retry on a later run): %v", s.label, len(res.uuids), res.err)
				skipped += len(res.uuids)
				continue
			}
			bad := false
			for i, u := range res.uuids {
				if len(res.vecs[i]) != dims {
					bad = true
					break
				}
				if _, _, _, err := d.ExecuteQuery(ctx, setQ, map[string]any{"u": u, "vec": res.vecs[i]}); err != nil {
					log.Printf("[%s] SET %s failed (skipping): %v", s.label, u, err)
					bad = true
					break
				}
			}
			if bad {
				skipped += len(res.uuids)
				continue
			}
			done += len(res.uuids)
			written++
			if written%ckptEvery == 0 {
				checkpoint(ctx, d)
				el := time.Since(start).Seconds()
				rate := float64(done) / el
				eta := time.Duration(0)
				if rate > 0 {
					eta = time.Duration(float64(total-done)/rate) * time.Second
				}
				log.Printf("[%s] %d/%d  %.0f/s  ETA %s  (skipped %d)", s.label, done, total, rate, eta.Round(time.Second), skipped)
			}
		}
		checkpoint(ctx, d)
		log.Printf("[%s] chunk %d complete: %d done, %d skipped", s.label, c, done, skipped)
	}
	feedWg.Wait()
	close(jobCh)
	checkpoint(ctx, d)
	log.Printf("[%s] FINISHED: %d embedded this run, %d skipped", s.label, done, skipped)
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
		log.Printf("checkpoint warning (driver flushes on close): %v", err)
	}
}
