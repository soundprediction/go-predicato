//go:build system_ladybug

package driver

import (
	"os"
	"path/filepath"
	"testing"
)

// TestSeedBundledExtensionCopies verifies the vendored extension is copied into
// lbug's home extension dir (offline) when absent, and is a no-op when present.
func TestSeedBundledExtensionCopies(t *testing.T) {
	tmp := t.TempDir()
	bundle := filepath.Join(tmp, "extensions")
	if err := os.MkdirAll(bundle, 0o755); err != nil {
		t.Fatal(err)
	}
	want := []byte("FAKE-VECTOR-EXTENSION")
	if err := os.WriteFile(filepath.Join(bundle, "libvector.lbug_extension"), want, 0o644); err != nil {
		t.Fatal(err)
	}
	home := filepath.Join(tmp, "home")
	if err := os.MkdirAll(home, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PREDICATO_LADYBUG_EXTENSION_DIR", bundle)
	t.Setenv("HOME", home)

	seedBundledExtension("vector", "0.17.0")

	dst := filepath.Join(home, ".lbdb", "extension", "0.17.0", lbugPlatformDir(), "vector", "libvector.lbug_extension")
	got, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("extension not seeded to %s: %v", dst, err)
	}
	if string(got) != string(want) {
		t.Fatalf("seeded content mismatch")
	}

	// no-op when already present (and no bundle): overwrite dst, ensure unchanged
	t.Setenv("PREDICATO_LADYBUG_EXTENSION_DIR", filepath.Join(tmp, "nonexistent"))
	seedBundledExtension("vector", "0.17.0")
	if got2, _ := os.ReadFile(dst); string(got2) != string(want) {
		t.Fatalf("existing extension should not be modified")
	}
}

// TestBundledExtensionFileFound verifies the bundled file is located via the env dir.
func TestBundledExtensionFileFound(t *testing.T) {
	tmp := t.TempDir()
	if err := os.WriteFile(filepath.Join(tmp, "libfts.lbug_extension"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PREDICATO_LADYBUG_EXTENSION_DIR", tmp)
	if got := bundledExtensionFile("fts"); got == "" {
		t.Fatal("expected to locate bundled fts extension")
	}
}
