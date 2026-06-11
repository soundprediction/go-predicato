//go:build system_ladybug

package driver

import (
	"context"
	"os"
	"strconv"
	"strings"
	"testing"
)

// TestBundledVectorSearchOffline proves the driver seeds the bundled vector
// extension and the indexed search works WITHOUT the registry-installed copy.
// Gated on TEST_LADYBUG_DB (a ladybug graph with a fact_emb_idx) and TEST_QVEC
// (a file containing a "[f,f,...]" 1024-dim query-vector literal).
func TestBundledVectorSearchOffline(t *testing.T) {
	db := os.Getenv("TEST_LADYBUG_DB")
	qf := os.Getenv("TEST_QVEC")
	if db == "" || qf == "" {
		t.Skip("set TEST_LADYBUG_DB and TEST_QVEC to run")
	}
	raw, err := os.ReadFile(qf)
	if err != nil {
		t.Fatal(err)
	}
	var emb []float32
	for _, tok := range strings.Split(strings.Trim(strings.TrimSpace(string(raw)), "[]"), ",") {
		f, err := strconv.ParseFloat(strings.TrimSpace(tok), 32)
		if err == nil {
			emb = append(emb, float32(f))
		}
	}
	if len(emb) != 1024 {
		t.Fatalf("expected 1024-dim query vector, got %d", len(emb))
	}

	// Point HOME at a clean temp dir so lbug's extension home starts empty —
	// the search then MUST rely on the driver seeding the bundle (hermetic;
	// never touches the developer's real ~/.lbdb).
	t.Setenv("HOME", t.TempDir())

	d, err := NewLadybugDriverWithConfig(&LadybugDriverConfig{
		DBPath: db, ReadOnly: true, BufferPoolSize: 8 << 30, MaxConcurrentQueries: 1,
	})
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer d.Close()

	edges, err := d.SearchEdgesByEmbedding(context.Background(), emb, "canonical", 5)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(edges) == 0 {
		t.Fatal("0 edges — bundled vector extension/index path not used")
	}
	t.Logf("OK: %d edges via indexed vector search (offline-seeded extension)", len(edges))
	for i, e := range edges {
		if i < 3 {
			t.Logf("  %s", e.Fact)
		}
	}
}
