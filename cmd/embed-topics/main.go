// embed-topics reads a topic_list.tsv and writes <slug>_vector.json files using
// the go-embedeverything library (Qwen3-Embedding-0.6B).
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/soundprediction/predicato/pkg/embedder"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage: embed-topics <topic_list.tsv> <output-dir>\n")
		os.Exit(1)
	}
	tsvPath := os.Args[1]
	outDir := os.Args[2]

	// Parse TSV.
	f, err := os.Open(tsvPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error opening %s: %v\n", tsvPath, err)
		os.Exit(1)
	}
	defer f.Close()

	type topic struct {
		slug, phrase string
	}
	var topics []topic
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "\t", 2)
		if len(parts) != 2 {
			continue
		}
		topics = append(topics, topic{slug: parts[0], phrase: parts[1]})
	}
	fmt.Printf("Found %d topics\n", len(topics))

	// Create embedder.
	emb, err := embedder.NewEmbedEverythingClient(&embedder.EmbedEverythingConfig{
		Config: &embedder.Config{
			Model:      "Qwen/Qwen3-Embedding-0.6B",
			Dimensions: 1024,
		},
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating embedder: %v\n", err)
		os.Exit(1)
	}
	defer emb.Close()

	os.MkdirAll(outDir, 0o755)

	for i, t := range topics {
		outPath := fmt.Sprintf("%s/%s_vector.json", outDir, t.slug)
		if _, err := os.Stat(outPath); err == nil {
			fmt.Printf("  [%d/%d] %s — already exists, skipping\n", i+1, len(topics), t.slug)
			continue
		}

		vec, err := emb.EmbedSingle(nil, t.phrase)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  [%d/%d] %s — ERROR: %v\n", i+1, len(topics), t.slug, err)
			continue
		}

		data, _ := json.Marshal(vec)
		if err := os.WriteFile(outPath, data, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "  [%d/%d] %s — write error: %v\n", i+1, len(topics), t.slug, err)
			continue
		}
		fmt.Printf("  [%d/%d] %s — %d dims\n", i+1, len(topics), t.slug, len(vec))
	}
	fmt.Println("Done.")
}
