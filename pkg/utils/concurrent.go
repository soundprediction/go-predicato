package utils

import (
	"context"
	"sync"
)

// ConcurrentExecutor manages concurrent execution of functions with a semaphore
type ConcurrentExecutor struct {
	semaphore chan struct{}
}

// NewConcurrentExecutor creates a new concurrent executor with the specified max concurrency
func NewConcurrentExecutor(maxConcurrency int) *ConcurrentExecutor {
	if maxConcurrency <= 0 {
		maxConcurrency = GetSemaphoreLimit()
	}
	return &ConcurrentExecutor{
		semaphore: make(chan struct{}, maxConcurrency),
	}
}

// Execute runs functions concurrently with semaphore control.
// Panics in goroutines are recovered and converted to PanicError.
func (e *ConcurrentExecutor) Execute(ctx context.Context, functions ...func() error) []error {
	if len(functions) == 0 {
		return nil
	}

	results := make([]error, len(functions))
	var wg sync.WaitGroup

	for i, fn := range functions {
		wg.Add(1)
		go func(index int, function func() error) {
			defer wg.Done()
			defer RecoverWithCallback(func(err error) {
				results[index] = err
			})

			// Acquire semaphore
			select {
			case e.semaphore <- struct{}{}:
				defer func() { <-e.semaphore }()
			case <-ctx.Done():
				results[index] = ctx.Err()
				return
			}

			// Execute function
			results[index] = function()
		}(i, fn)
	}

	wg.Wait()
	return results
}

// ExecuteWithResults runs functions concurrently and returns results.
// Panics in goroutines are recovered and converted to PanicError.
func ExecuteWithResults[T any](ctx context.Context, maxConcurrency int, functions ...func() (T, error)) ([]T, []error) {
	if len(functions) == 0 {
		return nil, nil
	}

	executor := NewConcurrentExecutor(maxConcurrency)
	results := make([]T, len(functions))
	errors := make([]error, len(functions))
	var wg sync.WaitGroup

	for i, fn := range functions {
		wg.Add(1)
		go func(index int, function func() (T, error)) {
			defer wg.Done()
			defer RecoverWithCallback(func(err error) {
				errors[index] = err
			})

			// Acquire semaphore
			select {
			case executor.semaphore <- struct{}{}:
				defer func() { <-executor.semaphore }()
			case <-ctx.Done():
				errors[index] = ctx.Err()
				return
			}

			// Execute function
			results[index], errors[index] = function()
		}(i, fn)
	}

	wg.Wait()
	return results, errors
}

// SemaphoreGather is equivalent to Python's semaphore_gather function
// It executes functions concurrently with a semaphore to limit concurrency
func SemaphoreGather(ctx context.Context, maxConcurrency int, functions ...func() error) []error {
	executor := NewConcurrentExecutor(maxConcurrency)
	return executor.Execute(ctx, functions...)
}

// SemaphoreGatherWithResults executes functions concurrently and returns both results and errors
func SemaphoreGatherWithResults[T any](ctx context.Context, maxConcurrency int, functions ...func() (T, error)) ([]T, []error) {
	return ExecuteWithResults(ctx, maxConcurrency, functions...)
}

// Worker represents a worker function that processes items from a channel
type Worker[T any, R any] func(ctx context.Context, item T) (R, error)

// WorkerPool manages a pool of workers processing items concurrently.
//
// Goroutine Lifecycle:
// - Worker goroutines are created when ProcessItems is called
// - Workers read from an internal items channel until it's closed
// - All workers terminate when:
//   - The items channel is exhausted and closed
//   - The context is cancelled
//
// - ProcessItems blocks until all workers complete via WaitGroup
// - Panics in workers are recovered and converted to PanicError
//
// Example:
//
//	pool := NewWorkerPool(4, func(ctx context.Context, item string) (int, error) {
//	    return len(item), nil
//	})
//	results, errors := pool.ProcessItems(ctx, []string{"a", "bb", "ccc"})
type WorkerPool[T any, R any] struct {
	worker     Worker[T, R]
	numWorkers int
}

// NewWorkerPool creates a new worker pool
func NewWorkerPool[T any, R any](numWorkers int, worker Worker[T, R]) *WorkerPool[T, R] {
	if numWorkers <= 0 {
		numWorkers = GetSemaphoreLimit()
	}
	return &WorkerPool[T, R]{
		numWorkers: numWorkers,
		worker:     worker,
	}
}

// ProcessItems processes items using the worker pool.
// Panics in worker goroutines are recovered and converted to PanicError.
func (wp *WorkerPool[T, R]) ProcessItems(ctx context.Context, items []T) ([]R, []error) {
	if len(items) == 0 {
		return nil, nil
	}

	itemsChan := make(chan struct {
		item  T
		index int
	}, len(items))

	// Send items to channel
	for i, item := range items {
		itemsChan <- struct {
			item  T
			index int
		}{item: item, index: i}
	}
	close(itemsChan)

	// Prepare result slices
	results := make([]R, len(items))
	errors := make([]error, len(items))
	var wg sync.WaitGroup
	var mu sync.Mutex // Protect errors slice during panic recovery

	// Start workers
	for i := 0; i < wp.numWorkers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case item, ok := <-itemsChan:
					if !ok {
						return
					}
					// Use a wrapper function to recover from panics
					func() {
						defer RecoverWithCallback(func(err error) {
							mu.Lock()
							errors[item.index] = err
							mu.Unlock()
						})
						results[item.index], errors[item.index] = wp.worker(ctx, item.item)
					}()
				case <-ctx.Done():
					return
				}
			}
		}()
	}

	wg.Wait()
	return results, errors
}

// Batch processes items in batches
func Batch[T any](items []T, batchSize int) [][]T {
	if batchSize <= 0 {
		batchSize = 10
	}

	var batches [][]T
	for i := 0; i < len(items); i += batchSize {
		end := i + batchSize
		if end > len(items) {
			end = len(items)
		}
		batches = append(batches, items[i:end])
	}
	return batches
}
