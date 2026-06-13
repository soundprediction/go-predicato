//go:build cgo && system_ladybug

package main

import (
	"flag"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	ladybug "github.com/LadybugDB/go-ladybug"
)

func loadVec(c *ladybug.Connection) {
	if r, err := c.Query("LOAD VECTOR;"); err == nil && r != nil {
		r.Close()
	}
}

func bench(db *ladybug.Database, query string, C, total int, pool bool) float64 {
	conns := make([]*ladybug.Connection, 0, C)
	n := 1
	if pool {
		n = C
	}
	for i := 0; i < n; i++ {
		c, err := ladybug.OpenConnection(db)
		if err != nil {
			log.Fatalf("open conn: %v", err)
		}
		loadVec(c)
		conns = append(conns, c)
	}
	var mu sync.Mutex
	var done int64
	per := total / C
	var wg sync.WaitGroup
	start := time.Now()
	for g := 0; g < C; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			conn := conns[0]
			if pool {
				conn = conns[g]
			}
			for i := 0; i < per; i++ {
				if !pool {
					mu.Lock()
				}
				r, err := conn.Query(query)
				if !pool {
					mu.Unlock()
				}
				if err == nil && r != nil {
					r.Close()
				}
				atomic.AddInt64(&done, 1)
			}
		}(g)
	}
	wg.Wait()
	return float64(done) / time.Since(start).Seconds()
}

func main() {
	dbPath := flag.String("db", "/data/topic-graphs/ladybug/canonical.ladybug", "")
	total := flag.Int("queries", 3000, "queries per level")
	flag.Parse()

	sys := ladybug.SystemConfig{BufferPoolSize: 32 << 30, MaxNumThreads: 32, ReadOnly: true, MaxDbSize: 1 << 43, EnableCompression: true}
	db, err := ladybug.OpenDatabase(*dbPath, sys)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	defer db.Close()

	c0, _ := ladybug.OpenConnection(db)
	loadVec(c0)
	qr, err := c0.Query("MATCH (e:RelatesToNode_) RETURN CAST(e.fact_embedding AS STRING) AS s LIMIT 1")
	if err != nil {
		log.Fatalf("get vec: %v", err)
	}
	t, _ := qr.Next()
	lit, _ := t.GetValue(0)
	qr.Close()
	query := fmt.Sprintf("CALL QUERY_VECTOR_INDEX('RelatesToNode_','fact_emb_idx', CAST(%v AS FLOAT[1024]), 10) RETURN node.uuid AS u, distance ORDER BY distance", lit)

	// warmup
	for i := 0; i < 50; i++ {
		if r, e := c0.Query(query); e == nil && r != nil {
			r.Close()
		}
	}
	fmt.Println("conc |  pool q/s | single+mutex q/s")
	for _, C := range []int{1, 2, 4, 8, 16, 32} {
		p := bench(db, query, C, *total, true)
		s := bench(db, query, C, *total, false)
		fmt.Printf("%4d | %9.0f | %9.0f\n", C, p, s)
	}
}
