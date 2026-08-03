//go:build cgo

package router

import (
	"log/slog"
	"os"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/soundprediction/predicato/pkg/driver"
)

// fakeGraphDriver satisfies driver.GraphDriver by embedding the interface: only
// the methods the manager actually touches are implemented, and any other call
// panics loudly rather than silently misbehaving.
type fakeGraphDriver struct {
	driver.GraphDriver
}

func (f *fakeGraphDriver) Close() error { return nil }

const testGraphDbType GraphDbType = "router-test-fake"

// registerSlowFactory installs a driver factory whose opens take `delay`, and
// counts how many opens happened. Returns the counter.
func registerSlowFactory(t *testing.T, delay time.Duration) *int32 {
	t.Helper()
	var opens int32
	prev, had := driverFactories[testGraphDbType]
	RegisterDriverFactory(testGraphDbType, func(_ *ClientConfig, _ string, _ int) (driver.GraphDriver, error) {
		atomic.AddInt32(&opens, 1)
		time.Sleep(delay)
		return &fakeGraphDriver{}, nil
	})
	t.Cleanup(func() {
		if had {
			driverFactories[testGraphDbType] = prev
		} else {
			delete(driverFactories, testGraphDbType)
		}
	})
	return &opens
}

func testManager(t *testing.T, names ...string) *Manager {
	t.Helper()
	configs := make([]ClientConfig, 0, len(names))
	for _, n := range names {
		configs = append(configs, ClientConfig{
			Name:     n,
			GroupID:  n,
			ReadOnly: true,
			GraphDb:  map[string]any{"type": string(testGraphDbType), "db_path": "/nonexistent/" + n},
		})
	}
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError}))
	return NewManagerWithOptions(configs, nil, nil, logger, 0)
}

// A cold open used to run while holding the manager mutex, so one slow graph
// stalled every other graph query in the process — which is what made
// Router.SearchWithClients' goroutine fan-out serialise behind whichever graph
// happened to be cold.
func TestGetClientColdOpenDoesNotBlockOtherGraphs(t *testing.T) {
	const delay = 250 * time.Millisecond
	registerSlowFactory(t, delay)
	pm := testManager(t, "alpha", "beta")

	start := time.Now()
	var wg sync.WaitGroup
	errs := make([]error, 2)
	for i, name := range []string{"alpha", "beta"} {
		wg.Add(1)
		go func(i int, name string) {
			defer wg.Done()
			_, errs[i] = pm.GetClient(name)
		}(i, name)
	}
	wg.Wait()
	elapsed := time.Since(start)

	for i, err := range errs {
		if err != nil {
			t.Fatalf("client %d failed to open: %v", i, err)
		}
	}
	// Serialised this is 2*delay = 500ms; concurrent it is ~delay.
	if elapsed > 2*delay {
		t.Fatalf("distinct graphs still open serially: %s for 2 x %s opens", elapsed, delay)
	}
}

// Concurrent callers for the SAME graph must share one open — that collapsing is
// the only thing the manager lock was usefully providing across initializeClient.
func TestGetClientCollapsesConcurrentOpensOfSameGraph(t *testing.T) {
	opens := registerSlowFactory(t, 100*time.Millisecond)
	pm := testManager(t, "alpha")

	const n = 8
	var wg sync.WaitGroup
	clients := make([]any, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			c, err := pm.GetClient("alpha")
			if err != nil {
				t.Errorf("open %d failed: %v", i, err)
				return
			}
			clients[i] = c
		}(i)
	}
	wg.Wait()

	if got := atomic.LoadInt32(opens); got != 1 {
		t.Fatalf("want exactly 1 open for %d concurrent callers, got %d", n, got)
	}
	for i := 1; i < n; i++ {
		if clients[i] != clients[0] {
			t.Fatalf("caller %d got a different client than caller 0", i)
		}
	}
	// And exactly one driver is registered for it — a losing racer must never
	// leave a spare behind, nor clobber the winner's entry.
	if len(pm.drivers) != 1 {
		t.Fatalf("want 1 registered driver, got %d", len(pm.drivers))
	}
}
