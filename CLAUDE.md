# Predicato Project Instructions

## Overview

The `predicato` package is a knowledge graph framework designed for building and querying dynamic knowledge graphs that evolve over time.

## Building

**IMPORTANT**: This project uses CGO and requires the LadybugDB native library. Use the Makefile for building:

```bash
# Build everything (recommended)
make build

# Build the CLI binary
make build-cli

# Run tests
make test

# Run server
make run-server
```

### Manual Build Steps

If you need to build manually:

```bash
# Download the ladybug native library (required before building)
go generate ./cmd/main.go

# Build with the system_ladybug tag
go build -tags system_ladybug ./...

# Build the CLI
go build -tags system_ladybug -o bin/predicato ./cmd/main.go
```

**Note**: The `go generate` command downloads `liblbug.so` to `cmd/lib-ladybug/`. If you see linker errors like `cannot find -llbug`, run `go generate ./cmd/main.go` first.

See the `Makefile` for all available build targets (`make help`).

**Known Issue**: Building `./...` (all packages including examples) may fail with linker errors because the examples need their own copy of `liblbug.so`. Use `make build-cli` to build the main CLI, or run `go generate` in each example directory before building.

## Context and Background

### Technical Considerations

- **Type Safety**: Leverage Go's type system to provide compile-time safety.
- **Concurrency**: Use Go's goroutines and channels effectively.
- **Error Handling**: Use Go's explicit error handling rather than exceptions, providing clear error messages and proper error wrapping.
- **Resource Management**: Properly implement cleanup and resource management using defer statements and Close() methods.
- **Testing**: Maintain comprehensive tests.

### Project Structure

Follow the existing project structure and conventions:
- Core functionality in the root package
- Supporting packages in `pkg/` subdirectories
- Maintain separation of concerns between drivers, search, LLM clients, etc.
- Use interfaces to enable dependency injection and testing

### Development Workflow

1. **Before Implementation**:
   - Understand the method's purpose and behavior
   - Check dependencies and related methods
   - Plan the Go implementation approach

2. **During Implementation**:
   - Implement full functionality, not placeholders
   - Add appropriate error handling
   - Include proper documentation
   - Consider Go-specific optimizations

3. **After Implementation**:
   - Run tests to ensure functionality
   - Verify integration with existing code

### Linting

This project uses `betteralign` to optimize struct field ordering for memory efficiency and `golangci-lint` with the `fieldalignment` govet pass. Configuration is in `.golangci.yml`.

**After modifying or creating structs**, run the struct alignment linter:

```bash
# Check for alignment issues
betteralign ./...

# Auto-fix alignment issues
betteralign --apply ./...
```

**IMPORTANT**: `betteralign --apply` reorders struct fields. If any struct literals use positional (non-named) initialization, the reordering will break them. After running `--apply`, always verify with `go build ./...` and fix any struct literals to use named fields.

### Quality Standards

- Code should compile without warnings
- All public functions should have proper documentation
- Error messages should be clear and actionable
- Performance should be reasonable for the intended use cases
- Memory usage should be efficient and not leak resources
- Struct fields should be ordered for optimal memory alignment (run `betteralign ./...` to check)

