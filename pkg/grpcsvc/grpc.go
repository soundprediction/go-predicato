// Package grpcsvc serves predicato's knowledge-graph operations over gRPC so the
// engine can run as a separate process (or on a different machine) from the
// services that query it. The wire contract is defined in
// proto/predicato/v1/graph.proto and code-generated into ./pb; this package wraps
// a predicato.Predicato as the server and provides a typed Client. Large embedding
// vectors are omitted from the wire types (callers use names/facts/types/sources).
package grpcsvc

import "github.com/soundprediction/predicato/pkg/grpcsvc/pb"

// ServiceName is the fully-qualified gRPC service name (from the proto package).
var ServiceName = pb.GraphService_ServiceDesc.ServiceName
