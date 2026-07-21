package router

import (
	"os"
	"path/filepath"
	"testing"
)

// The broad "canonical" graph is auto-flagged as the routing fallback even
// without a companion .yaml; ordinary topics are not.
func TestDiscoverTopicGraphs_CanonicalIsFallback(t *testing.T) {
	dir := t.TempDir()
	for _, slug := range []string{"canonical", "pregnancy"} {
		if err := os.WriteFile(filepath.Join(dir, slug+".md"), []byte("# "+slug+"\nDescription.\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		// findTopicDatabase accepts a <slug>.ladybug file.
		if err := os.WriteFile(filepath.Join(dir, slug+".ladybug"), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	configs, _, err := DiscoverTopicGraphs(dir)
	if err != nil {
		t.Fatalf("DiscoverTopicGraphs: %v", err)
	}
	byName := make(map[string]ClientConfig, len(configs))
	for _, c := range configs {
		byName[c.Name] = c
	}

	if c, ok := byName["canonical"]; !ok || !c.Fallback {
		t.Fatalf("canonical should be discovered and marked Fallback; ok=%v cfg=%+v", ok, c)
	}
	if c, ok := byName["pregnancy"]; !ok || c.Fallback {
		t.Fatalf("pregnancy should be discovered and NOT marked Fallback; ok=%v fallback=%v", ok, c.Fallback)
	}
}

func TestContainsClient(t *testing.T) {
	names := []string{"a", "canonical", "b"}
	if !containsClient(names, "canonical") {
		t.Error("expected containsClient=true for a present name")
	}
	if containsClient(names, "missing") {
		t.Error("expected containsClient=false for an absent name")
	}
}
