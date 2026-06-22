#!/usr/bin/env bash
# Regenerate Go gRPC stubs from the predicato protos.
# Requires: protoc, protoc-gen-go, protoc-gen-go-grpc on PATH
#   go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
#   go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
set -euo pipefail
cd "$(dirname "$0")/.."
protoc -I proto \
  --go_out=. --go_opt=module=github.com/soundprediction/predicato \
  --go-grpc_out=. --go-grpc_opt=module=github.com/soundprediction/predicato \
  proto/predicato/v1/graph.proto
echo "generated pkg/grpcsvc/pb/*.pb.go"
