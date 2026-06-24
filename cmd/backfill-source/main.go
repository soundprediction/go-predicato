//go:build cgo && system_ladybug

// backfill-source augments an existing ladybug knowledge graph IN PLACE with the
// upstream source provenance that was dropped when the graph was first imported
// from an aggregator (PrimeKG's DuckDB kept "name + type only, no xref IDs").
//
// It joins each Entity to PrimeKG's raw nodes.tab by name (names are ~99.9% unique;
// the rare collision is disambiguated by type) and writes, into the entity's
// `attributes` JSON, the resolvable upstream reference:
//   - xrefs:       ["GO:0051581"]              (canonical CURIE)
//   - source_urls: ["https://amigo.geneontology.org/amigo/term/GO:0051581"]
//
// Non-destructive and idempotent: only entities whose xrefs are still empty
// (attributes contains `"xrefs": []`) are touched, so re-runs resume cleanly and
// nothing already-sourced is overwritten. Reads in LIMIT chunks over that filter
// (embedded rows drop out of the filter, so successive chunks advance), writing
// back via point-lookup SET on the uuid hash index. Modeled on backfill-embeddings.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/soundprediction/predicato/pkg/driver"
)

// pkgEntry is one PrimeKG node: its source DB, the id within that DB, and its type.
type pkgEntry struct {
	source string
	id     string
	ntype  string
}

func main() {
	dbPath := flag.String("db", "", "path to .ladybug database to augment (required)")
	nodesTab := flag.String("primekg-nodes", "", "path to PrimeKG raw nodes.tab (required)")
	chunkSz := flag.Int("chunk", 50000, "entities pulled per DB scan")
	ckptEvery := flag.Int("checkpoint-every", 2000, "checkpoint every N written rows")
	bufBytes := flag.Uint64("buffer-pool-bytes", 8<<30, "ladybug buffer pool size in bytes")
	dryRun := flag.Bool("dry-run", false, "report matches without writing")
	maxChunks := flag.Int("max-chunks", 0, "stop after N chunks (0=unlimited; for sampling)")
	flag.Parse()
	if *dbPath == "" || *nodesTab == "" {
		log.Fatal("--db and --primekg-nodes are required")
	}
	ctx := context.Background()

	index := loadPrimeKG(*nodesTab)
	log.Printf("loaded %d PrimeKG node names", len(index))

	cfg := driver.DefaultLadybugDriverConfig()
	cfg.DBPath = *dbPath
	cfg.ReadOnly = *dryRun
	cfg.BufferPoolSize = *bufBytes
	d, err := driver.NewLadybugDriverWithConfig(cfg)
	if err != nil {
		log.Fatalf("open %s: %v", *dbPath, err)
	}
	defer d.Close()

	backfill(ctx, d, index, *chunkSz, *ckptEvery, *maxChunks, *dryRun)
}

// loadPrimeKG reads nodes.tab (node_index, node_id, node_type, node_name, node_source)
// into a name-keyed index. Values are slices so a colliding name keeps all sources
// for type-based disambiguation at join time.
func loadPrimeKG(path string) map[string][]pkgEntry {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("open %s: %v", path, err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	index := make(map[string][]pkgEntry, 130000)
	first := true
	for sc.Scan() {
		if first { // header
			first = false
			continue
		}
		cols := strings.Split(sc.Text(), "\t")
		if len(cols) < 5 {
			continue
		}
		id := strings.Trim(cols[1], `"`)
		ntype := strings.Trim(cols[2], `"`)
		name := strings.Trim(cols[3], `"`)
		source := strings.Trim(cols[4], `"`)
		if name == "" || id == "" || source == "" {
			continue
		}
		key := strings.ToLower(strings.TrimSpace(name))
		index[key] = append(index[key], pkgEntry{source: source, id: id, ntype: ntype})
	}
	if err := sc.Err(); err != nil {
		log.Fatalf("scan %s: %v", path, err)
	}
	return index
}

func backfill(ctx context.Context, d *driver.LadybugDriver, index map[string][]pkgEntry, chunkSz, ckptEvery, maxChunks int, dryRun bool) {
	// Un-backfilled entities still carry the import's empty `"xrefs": []`.
	miss := `r.attributes IS NOT NULL AND r.attributes CONTAINS '"xrefs": []'`
	total := count(ctx, d, fmt.Sprintf("MATCH (r:Entity) WHERE %s RETURN count(r) AS c", miss))
	log.Printf("%d entities without xrefs", total)
	if total == 0 {
		return
	}
	selChunk := fmt.Sprintf("MATCH (r:Entity) WHERE %s RETURN r.uuid AS uuid, r.name AS name, r.attributes AS attrs LIMIT %d", miss, chunkSz)
	setQ := "MATCH (r:Entity {uuid:$u}) SET r.attributes = $a"

	matched, unmatched, written := 0, 0, 0
	start := time.Now()
	for c := 0; maxChunks == 0 || c < maxChunks; c++ {
		rows := query(ctx, d, selChunk, nil)
		if len(rows) == 0 {
			break
		}
		for _, row := range rows {
			uuid, _ := row["uuid"].(string)
			name, _ := row["name"].(string)
			attrs, _ := row["attrs"].(string)
			entry, ok := resolve(index, name, attrs)
			if !ok {
				unmatched++
				if dryRun {
					continue
				}
				// Mark as processed (empty-but-checked) so re-runs don't rescan it:
				// flip "xrefs": [] to "xrefs":[] (compact) by rewriting attributes.
				if newAttrs, changed := setXrefs(attrs, nil, nil); changed {
					_, _, _, _ = d.ExecuteQuery(ctx, setQ, map[string]any{"u": uuid, "a": newAttrs})
				}
				continue
			}
			matched++
			curie := toCURIE(entry)
			url := toURL(entry)
			if dryRun {
				if matched <= 12 {
					log.Printf("  %q -> %s  %s", name, curie, url)
				}
				continue
			}
			newAttrs, changed := setXrefs(attrs, []string{curie}, []string{url})
			if !changed {
				continue
			}
			if _, _, _, err := d.ExecuteQuery(ctx, setQ, map[string]any{"u": uuid, "a": newAttrs}); err != nil {
				log.Printf("SET %s failed (skipping): %v", uuid, err)
				continue
			}
			written++
			if written%ckptEvery == 0 {
				checkpoint(ctx, d)
				el := time.Since(start).Seconds()
				rate := float64(written) / el
				log.Printf("%d written  %d matched  %d unmatched  %.0f/s", written, matched, unmatched, rate)
			}
		}
		if dryRun {
			log.Printf("dry-run chunk %d: %d matched, %d unmatched (no writes)", c, matched, unmatched)
			break
		}
		checkpoint(ctx, d)
		log.Printf("chunk %d complete: %d written, %d matched, %d unmatched", c, written, matched, unmatched)
	}
	if !dryRun {
		checkpoint(ctx, d)
	}
	log.Printf("FINISHED: %d written, %d matched, %d unmatched of %d", written, matched, unmatched, total)
}

// resolve picks the PrimeKG entry for an entity name, disambiguating a colliding
// name by matching the graph entity_type (from attributes) to PrimeKG's node_type.
func resolve(index map[string][]pkgEntry, name, attrs string) (pkgEntry, bool) {
	entries := index[strings.ToLower(strings.TrimSpace(name))]
	if len(entries) == 0 {
		return pkgEntry{}, false
	}
	if len(entries) == 1 {
		return entries[0], true
	}
	want := normType(gjsonString(attrs, "entity_type"))
	for _, e := range entries {
		if want != "" && normType(e.ntype) == want {
			return e, true
		}
	}
	return entries[0], true // fall back to first; names collide for <0.1% of nodes
}

// setXrefs writes xrefs/source_urls into the attributes JSON, returning the new
// string and whether it changed. Passing nil/empty still flips the empty-array
// marker to compact form so the idempotency filter no longer matches the row.
func setXrefs(attrs string, xrefs, urls []string) (string, bool) {
	var m map[string]any
	if err := json.Unmarshal([]byte(attrs), &m); err != nil || m == nil {
		return attrs, false
	}
	if len(xrefs) > 0 {
		m["xrefs"] = xrefs
	} else {
		m["xrefs"] = []string{}
	}
	if len(urls) > 0 {
		m["source_urls"] = urls
	}
	out, err := json.Marshal(m)
	if err != nil {
		return attrs, false
	}
	s := string(out)
	if s == attrs {
		return attrs, false
	}
	return s, true
}

// gjsonString pulls a top-level string field out of a small JSON object without a
// full typed struct (attributes is an open map).
func gjsonString(js, key string) string {
	var m map[string]any
	if json.Unmarshal([]byte(js), &m) != nil {
		return ""
	}
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func normType(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.ReplaceAll(s, " ", "_")
	s = strings.ReplaceAll(s, "/", "_")
	return s
}

// toCURIE renders the canonical compact identifier for an upstream reference.
func toCURIE(e pkgEntry) string {
	switch e.source {
	case "GO":
		return "GO:" + pad7(e.id)
	case "MONDO", "MONDO_grouped":
		return "MONDO:" + pad7(e.id)
	case "HPO":
		return "HP:" + pad7(e.id)
	case "UBERON":
		return "UBERON:" + pad7(e.id)
	case "NCBI":
		return "NCBIGene:" + e.id
	case "DrugBank":
		return "DrugBank:" + e.id
	case "REACTOME":
		return "Reactome:" + e.id
	case "CTD":
		return "CTD:" + e.id
	default:
		return e.source + ":" + e.id
	}
}

// toURL renders a resolvable link to the upstream source record.
func toURL(e pkgEntry) string {
	switch e.source {
	case "GO":
		return "https://amigo.geneontology.org/amigo/term/GO:" + pad7(e.id)
	case "MONDO", "MONDO_grouped":
		return "https://monarchinitiative.org/MONDO:" + pad7(e.id)
	case "HPO":
		return "https://hpo.jax.org/app/browse/term/HP:" + pad7(e.id)
	case "UBERON":
		return "http://purl.obolibrary.org/obo/UBERON_" + pad7(e.id)
	case "NCBI":
		return "https://www.ncbi.nlm.nih.gov/gene/" + e.id
	case "DrugBank":
		return "https://go.drugbank.com/drugs/" + e.id
	case "REACTOME":
		return "https://reactome.org/content/detail/" + e.id
	case "CTD":
		return "https://ctdbase.org/detail.go?type=chem&acc=" + e.id
	default:
		return ""
	}
}

// pad7 zero-pads a bare numeric ontology id to the conventional 7 digits
// (GO 51581 -> 0051581); leaves non-numeric ids untouched.
func pad7(id string) string {
	for _, r := range id {
		if r < '0' || r > '9' {
			return id
		}
	}
	if len(id) >= 7 {
		return id
	}
	return strings.Repeat("0", 7-len(id)) + id
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
