# Go Predicato Makefile

.PHONY: build build-cli test test-cgo test-nocgo clean fmt vet lint run-example run-server deps tidy generate

# Library path for CGO tests
LIB_PATH := $(shell pwd)/cmd/lib-ladybug
CGO_LDFLAGS := -L$(LIB_PATH) -Wl,-rpath,$(LIB_PATH)

# Download Ladybug native library
generate:
	go generate ./cmd/main.go
	@# Latest LadybugDB SONAME is liblbug.so.0; create symlink so runtime linker finds it
	@test -f cmd/lib-ladybug/liblbug.so && cd cmd/lib-ladybug && ln -sf liblbug.so liblbug.so.0 || true

# Build the project (requires generate first)
build: generate
	CGO_LDFLAGS="$(CGO_LDFLAGS)" go build -tags system_ladybug ./...

# Build CLI binary
build-cli: generate
	CGO_LDFLAGS="$(CGO_LDFLAGS)" go build -tags system_ladybug -o bin/predicato ./cmd/main.go

# Build CLI for multiple platforms
build-cli-all: generate
	GOOS=linux GOARCH=amd64 CGO_LDFLAGS="$(CGO_LDFLAGS)" go build -tags system_ladybug -o bin/predicato-linux-amd64 ./cmd/main.go
	GOOS=darwin GOARCH=amd64 CGO_LDFLAGS="$(CGO_LDFLAGS)" go build -tags system_ladybug -o bin/predicato-darwin-amd64 ./cmd/main.go
	GOOS=darwin GOARCH=arm64 CGO_LDFLAGS="$(CGO_LDFLAGS)" go build -tags system_ladybug -o bin/predicato-darwin-arm64 ./cmd/main.go
	GOOS=windows GOARCH=amd64 CGO_LDFLAGS="$(CGO_LDFLAGS)" go build -tags system_ladybug -o bin/predicato-windows-amd64.exe ./cmd/main.go

# Run all tests (CGO packages require generate first)
test: generate test-cgo test-nocgo

# Run tests for packages that require CGO/Ladybug
test-cgo:
	CGO_LDFLAGS="$(CGO_LDFLAGS)" go test -tags system_ladybug ./pkg/driver/... ./pkg/checkpoint/... ./pkg/modeler/... ./pkg/utils/... ./pkg/factstore/...

# Run tests for pure Go packages (no CGO required)
test-nocgo:
	go test ./pkg/embedder/... ./pkg/nlp/... ./pkg/prompts/... ./pkg/logger/...

# Run tests with coverage
test-coverage: generate
	CGO_LDFLAGS="$(CGO_LDFLAGS)" go test -tags system_ladybug -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

# Run tests with race detection
test-race: generate
	CGO_LDFLAGS="$(CGO_LDFLAGS)" go test -tags system_ladybug -race ./...

# Clean build artifacts
clean:
	go clean ./...
	rm -f coverage.out coverage.html

# Format code
fmt:
	go fmt ./...

# Run go vet
vet:
	go vet ./...

# Install dependencies
deps:
	go mod download

# Tidy dependencies
tidy:
	go mod tidy

# Run basic example (requires environment variables)
run-example:
	cd examples/basic && go run main.go

# Run server (requires environment variables)
run-server: generate
	CGO_LDFLAGS="$(CGO_LDFLAGS)" go run -tags system_ladybug ./cmd/main.go server

# Run server with debug mode
run-server-debug: generate
	CGO_LDFLAGS="$(CGO_LDFLAGS)" go run -tags system_ladybug ./cmd/main.go server --mode debug --log-level debug

# Development workflow
dev: fmt vet test

# CI workflow
ci: deps tidy fmt vet test-race

# Install development tools
install-tools:
	go install golang.org/x/tools/cmd/goimports@latest
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Run linter (requires golangci-lint)
lint:
	golangci-lint run

# Run comprehensive checks
check: fmt vet lint test-race

# Help
help:
	@echo "Available targets:"
	@echo "  generate     - Download Ladybug native library (required first)"
	@echo "  build        - Build the project (includes generate)"
	@echo "  build-cli    - Build CLI binary"
	@echo "  build-cli-all- Build CLI for multiple platforms"
	@echo "  test         - Run all tests (includes generate)"
	@echo "  test-cgo     - Run tests requiring CGO/Ladybug"
	@echo "  test-nocgo   - Run pure Go tests (no CGO required)"
	@echo "  test-coverage- Run tests with coverage report"
	@echo "  test-race    - Run tests with race detection"
	@echo "  clean        - Clean build artifacts"
	@echo "  fmt          - Format code"
	@echo "  vet          - Run go vet"
	@echo "  lint         - Run golangci-lint"
	@echo "  deps         - Install dependencies"
	@echo "  tidy         - Tidy dependencies"
	@echo "  run-example  - Run basic example"
	@echo "  run-server   - Run server"
	@echo "  run-server-debug - Run server with debug mode"
	@echo "  dev          - Development workflow (fmt, vet, test)"
	@echo "  ci           - CI workflow (deps, tidy, fmt, vet, test-race)"
	@echo "  install-tools- Install development tools"
	@echo "  check        - Run comprehensive checks"
	@echo "  help         - Show this help"