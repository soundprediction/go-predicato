//go:build cgo && system_ladybug

package router

import (
	"github.com/soundprediction/predicato/pkg/driver"
)

func init() {
	RegisterDriverFactory(GraphDbTypeLadybug, func(c *ClientConfig, dbPath string, embDim int) (driver.GraphDriver, error) {
		cfg := driver.DefaultLadybugDriverConfig()
		cfg.DBPath = dbPath
		cfg.ReadOnly = c.ReadOnly
		// Cap the per-database max size. DefaultLadybugDriverConfig sets MaxDbSize
		// to 8 TiB, which the engine reserves as VIRTUAL address space per database
		// via mmap. The router holds many graphs open at once, so ~14 of those 8 TiB
		// reservations exhaust the ~128 TiB x86-64 user address space and every
		// subsequent OpenDatabase fails with "status 1" (surfaced misleadingly as a
		// missing-.wal error by the driver's recovery fallback).
		//
		// That ceiling — not memory — is why --max-open-graphs has to be set so low.
		// A deployment serving 31 topic graphs could keep only 10 open and thrashed
		// the LRU, paying a full cold open on every eviction. A modest cap lets
		// dozens coexist; 64 GiB is ample headroom (the largest topic graph is
		// ~1.3 GiB). humn hit and fixed this on its own driver path first — see
		// microservice/config/graphdb_ladybug.go there; the pooled router never got
		// the same fix.
		cfg.MaxDbSize = maxDbSizeForTopicGraph()
		if c.ReadOnly {
			cfg.BufferPoolSize = topicBufferPoolBytes()
			// MaxConcurrentQueries becomes the engine's MaxNumThreads, i.e. the
			// intra-query parallelism. The driver default of 1 pins every search to a
			// single core, which is the binding cost once the remote hops are cheap:
			// on a live node the embedder answers in ~9ms, the reranker in ~37ms and
			// NLI in ~0.2ms/pair, yet five searches accounted for 11s of a 14.6s
			// request. Read-only topic graphs are never written to, so widening this
			// only adds scan/probe parallelism.
			cfg.MaxConcurrentQueries = topicQueryThreads()
		}
		return driver.NewLadybugDriverWithConfig(cfg)
	})
}
