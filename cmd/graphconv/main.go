//go:build system_duckpgq

package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/soundprediction/predicato/pkg/graphconv"
)

func main() {
	direction := flag.String("direction", "", "conversion direction: to-icebug or from-icebug (required)")
	source := flag.String("source", "", "source DuckDB file path (required)")
	dest := flag.String("dest", "", "destination DuckDB file path (required)")
	embeddingDim := flag.Int("embedding-dim", 1024, "embedding dimensions")
	csrTable := flag.String("csr-table", "predicato", "icebug table prefix")
	groupID := flag.String("group-id", "", "optional group_id filter")
	directed := flag.Bool("directed", true, "whether graph is directed")
	schemaOut := flag.String("schema-out", "", "write schema.cypher to this path (to-icebug only)")

	flag.Parse()

	if *direction == "" || *source == "" || *dest == "" {
		fmt.Fprintln(os.Stderr, "Usage: graphconv --direction {to-icebug,from-icebug} --source <path> --dest <path>")
		flag.PrintDefaults()
		os.Exit(1)
	}

	opts := graphconv.Options{
		Source:       *source,
		Dest:         *dest,
		EmbeddingDim: *embeddingDim,
		CSRTable:     *csrTable,
		GroupID:      *groupID,
		Directed:     *directed,
		SchemaOut:    *schemaOut,
	}

	var err error
	switch *direction {
	case "to-icebug":
		fmt.Printf("Converting DuckPGQ -> icebug: %s -> %s\n", *source, *dest)
		err = graphconv.ToIcebug(opts)
	case "from-icebug":
		fmt.Printf("Converting icebug -> DuckPGQ: %s -> %s\n", *source, *dest)
		err = graphconv.FromIcebug(opts)
	default:
		fmt.Fprintf(os.Stderr, "unknown direction: %s (use to-icebug or from-icebug)\n", *direction)
		os.Exit(1)
	}

	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
