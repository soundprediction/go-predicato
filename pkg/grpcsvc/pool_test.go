package grpcsvc

import "testing"

func TestParseStaticPool(t *testing.T) {
	cases := map[string]int{
		"host:50071":                 0, // single -> passthrough
		"dns:///predicato.svc:50071": 0, // scheme -> passthrough
		"h1:50071,h2:50071,h3:50071": 3, // static pool
		"h1:50071, h2:50071":         2, // trims spaces
		"h1:50071,":                  0, // one real member -> passthrough
	}
	for in, want := range cases {
		if got := len(parseStaticPool(in)); got != want {
			t.Errorf("parseStaticPool(%q) = %d members, want %d", in, got, want)
		}
	}
}
