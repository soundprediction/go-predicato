//go:build system_duckpgq

package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/soundprediction/predicato/pkg/driver"
)

func main() {
	dir := "/data/topic-graphs/duckpgq"
	if len(os.Args) > 1 {
		dir = os.Args[1]
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to read dir: %v\n", err)
		os.Exit(1)
	}

	ctx := context.Background()
	failures := 0

	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".duckdb") {
			continue
		}
		name := strings.TrimSuffix(e.Name(), ".duckdb")
		path := filepath.Join(dir, e.Name())

		d, err := driver.NewDuckPGQDriver(path, 1024)
		if err != nil {
			fmt.Printf("FAIL  %-35s  open: %v\n", name, err)
			failures++
			continue
		}

		if err := d.EnsureFTSIndex(ctx); err != nil {
			fmt.Printf("FAIL  %-35s  fts: %v\n", name, err)
			failures++
		} else {
			fmt.Printf("OK    %-35s  FTS indices created\n", name)
		}
		d.Close()
	}

	if failures > 0 {
		fmt.Printf("\n%d failures\n", failures)
		os.Exit(1)
	}
	fmt.Println("\nAll FTS indices created successfully")
}
