package grpcsvc

import (
	"strings"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/resolver/manual"
)

// roundRobinServiceConfig makes the client load-balance across ALL addresses a
// target resolves to. gRPC's default is pick_first (pins to one address), which
// would not spread load across a pool — so a pooled target needs this.
const roundRobinServiceConfig = `{"loadBalancingConfig":[{"round_robin":{}}]}`

// dialPool builds a gRPC ClientConn that round-robins across a POOL of servers.
// target may be:
//   - a single "host:port" — a one-machine pool;
//   - a comma-separated list "h1:port,h2:port,…" — a STATIC pool (manual resolver);
//   - any gRPC target scheme, e.g. "dns:///predicato.svc:50071" — a DYNAMIC pool
//     that re-resolves, so instances added/removed by autoscaling are picked up.
//
// Defaults to an insecure transport; pass opts to add TLS, keepalive, retries.
func dialPool(target string, opts ...grpc.DialOption) (*grpc.ClientConn, error) {
	base := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultServiceConfig(roundRobinServiceConfig),
	}
	if eps := parseStaticPool(target); len(eps) > 0 {
		r := manual.NewBuilderWithScheme("pool")
		addrs := make([]resolver.Address, 0, len(eps))
		for _, e := range eps {
			addrs = append(addrs, resolver.Address{Addr: e})
		}
		r.InitialState(resolver.State{Addresses: addrs})
		base = append(base, grpc.WithResolvers(r))
		target = r.Scheme() + ":///pool"
	}
	return grpc.NewClient(target, append(base, opts...)...)
}

// parseStaticPool returns the host:port members when target is a comma-separated
// STATIC pool (and not a scheme:// target). A single host:port or a scheme target
// (dns:///, etc.) returns nil so it passes through to the normal resolver.
func parseStaticPool(target string) []string {
	if strings.Contains(target, "://") {
		return nil // scheme target (dns:///name:port, etc.) — let gRPC resolve it
	}
	if !strings.Contains(target, ",") {
		return nil // single host:port
	}
	var out []string
	for _, p := range strings.Split(target, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	if len(out) < 2 {
		return nil
	}
	return out
}
