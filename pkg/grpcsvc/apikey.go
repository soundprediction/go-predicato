package grpcsvc

import (
	"context"
	"crypto/subtle"
	"strings"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// apiKeyFromContext extracts the presented API key from the incoming metadata,
// preferring the "x-api-key" header and falling back to a bearer token in the
// "authorization" header.
func apiKeyFromContext(ctx context.Context) string {
	md, ok := metadata.FromIncomingContext(ctx)
	if !ok {
		return ""
	}
	if vals := md.Get("x-api-key"); len(vals) > 0 {
		return vals[0]
	}
	if vals := md.Get("authorization"); len(vals) > 0 {
		return strings.TrimSpace(strings.TrimPrefix(vals[0], "Bearer "))
	}
	return ""
}

// authorize returns nil when the request is allowed and an Unauthenticated error
// otherwise. It is a no-op (always allows) when key is empty, enabling a
// fail-open rollout. Health checks are always exempt.
func authorize(ctx context.Context, fullMethod, key string) error {
	if key == "" {
		return nil
	}
	if strings.HasSuffix(fullMethod, "/Health") {
		return nil
	}
	got := apiKeyFromContext(ctx)
	if subtle.ConstantTimeCompare([]byte(got), []byte(key)) == 1 {
		return nil
	}
	return status.Error(codes.Unauthenticated, "missing or invalid api key")
}

// APIKeyUnaryInterceptor enforces an x-api-key (or "authorization: Bearer <key>")
// metadata header via constant-time compare. No-op when key=="" ; Health is exempt.
func APIKeyUnaryInterceptor(key string) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		if err := authorize(ctx, info.FullMethod, key); err != nil {
			return nil, err
		}
		return handler(ctx, req)
	}
}

// APIKeyStreamInterceptor enforces an x-api-key (or "authorization: Bearer <key>")
// metadata header via constant-time compare. No-op when key=="" ; Health is exempt.
func APIKeyStreamInterceptor(key string) grpc.StreamServerInterceptor {
	return func(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		if err := authorize(ss.Context(), info.FullMethod, key); err != nil {
			return err
		}
		return handler(srv, ss)
	}
}
