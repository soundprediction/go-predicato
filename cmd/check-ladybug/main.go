//go:build cgo

package main

import (
	"context"
	"database/sql"
	"fmt"
	"os"

	_ "github.com/marcboeker/go-duckdb"
)

func main() {
	path := "/data/topic-graphs/duckpgq/diabetes-type2.duckdb"
	if len(os.Args) > 1 {
		path = os.Args[1]
	}

	db, err := sql.Open("duckdb", path+"?access_mode=READ_ONLY")
	if err != nil {
		fmt.Fprintf(os.Stderr, "open: %v\n", err)
		os.Exit(1)
	}
	defer db.Close()
	db.Exec("LOAD fts")

	ctx := context.Background()

	// Check FTS tables
	rows, err := db.QueryContext(ctx, `SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'fts_%' ORDER BY table_name`)
	if err != nil {
		fmt.Printf("FTS check error: %v\n", err)
	} else {
		fmt.Println("FTS tables:")
		for rows.Next() {
			var name string
			rows.Scan(&name)
			fmt.Printf("  %s\n", name)
		}
		rows.Close()
	}

	// Check B-tree indices
	rows, err = db.QueryContext(ctx, `SELECT index_name FROM duckdb_indexes() ORDER BY index_name`)
	if err != nil {
		fmt.Printf("Index check error: %v\n", err)
	} else {
		fmt.Println("\nB-tree/HNSW indices:")
		for rows.Next() {
			var name string
			rows.Scan(&name)
			fmt.Printf("  %s\n", name)
		}
		rows.Close()
	}

	// Test BM25 search
	fmt.Println("\nBM25 node search for 'metformin insulin resistance':")
	rows, err = db.QueryContext(ctx, `SELECT uuid, name, entity_type, fts_main_entities.match_bm25(uuid, 'metformin insulin resistance') AS score FROM entities WHERE score IS NOT NULL ORDER BY score DESC LIMIT 10`)
	if err != nil {
		fmt.Printf("BM25 error: %v\n", err)
	} else {
		i := 0
		for rows.Next() {
			var uuid, name string
			var entityType sql.NullString
			var score float64
			rows.Scan(&uuid, &name, &entityType, &score)
			i++
			if len(name) > 80 {
				name = name[:80] + "..."
			}
			fmt.Printf("  %d. [%s] %s (score=%.4f)\n", i, entityType.String, name, score)
		}
		rows.Close()
		if i == 0 {
			fmt.Println("  (no results)")
		}
	}

	// Test BM25 edge search
	fmt.Println("\nBM25 edge search for 'metformin insulin resistance':")
	rows, err = db.QueryContext(ctx, `SELECT uuid, name, fact, fts_main_edges.match_bm25(uuid, 'metformin insulin resistance') AS score FROM edges WHERE score IS NOT NULL ORDER BY score DESC LIMIT 5`)
	if err != nil {
		fmt.Printf("BM25 edge error: %v\n", err)
	} else {
		i := 0
		for rows.Next() {
			var uuid, name, fact string
			var score float64
			rows.Scan(&uuid, &name, &fact, &score)
			i++
			if len(fact) > 100 {
				fact = fact[:100] + "..."
			}
			fmt.Printf("  %d. %s — %s (score=%.4f)\n", i, name, fact, score)
		}
		rows.Close()
		if i == 0 {
			fmt.Println("  (no results)")
		}
	}
}
