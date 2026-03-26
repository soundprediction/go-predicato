package driver

import "testing"

func TestDetectCommunities_DenseCluster(t *testing.T) {
	// 5 nodes, each connected to all others — dense enough for label propagation
	neighbors := map[string][]string{
		"a": {"b", "c", "d", "e"},
		"b": {"a", "c", "d", "e"},
		"c": {"a", "b", "d", "e"},
		"d": {"a", "b", "c", "e"},
		"e": {"a", "b", "c", "d"},
	}
	names := map[string]string{"a": "Alice", "b": "Bob", "c": "Carol", "d": "Dan", "e": "Eve"}

	clusters := DetectCommunities(neighbors, names, "test")
	if len(clusters) != 1 {
		t.Fatalf("expected 1 community, got %d", len(clusters))
	}
	if len(clusters[0].MemberUUIDs) != 5 {
		t.Fatalf("expected 5 members, got %d", len(clusters[0].MemberUUIDs))
	}
}

func TestDetectCommunities_TwoDenseClusters(t *testing.T) {
	// Two dense clusters with a single bridge
	neighbors := map[string][]string{
		"a1": {"a2", "a3", "a4"},
		"a2": {"a1", "a3", "a4"},
		"a3": {"a1", "a2", "a4", "b1"}, // bridge
		"a4": {"a1", "a2", "a3"},
		"b1": {"b2", "b3", "b4", "a3"}, // bridge
		"b2": {"b1", "b3", "b4"},
		"b3": {"b1", "b2", "b4"},
		"b4": {"b1", "b2", "b3"},
	}
	names := map[string]string{
		"a1": "A1", "a2": "A2", "a3": "A3", "a4": "A4",
		"b1": "B1", "b2": "B2", "b3": "B3", "b4": "B4",
	}

	clusters := DetectCommunities(neighbors, names, "test")
	if len(clusters) < 1 {
		t.Fatalf("expected at least 1 community, got %d", len(clusters))
	}
	// Should detect at least 2 communities (one per dense cluster)
	totalMembers := 0
	for _, c := range clusters {
		totalMembers += len(c.MemberUUIDs)
	}
	if totalMembers != 8 {
		t.Errorf("expected 8 total members across communities, got %d", totalMembers)
	}
}

func TestDetectCommunities_IsolatedPair(t *testing.T) {
	// A connected pair forms a community via label fallback
	neighbors := map[string][]string{
		"a": {"b"},
		"b": {"a"},
	}
	names := map[string]string{"a": "A", "b": "B"}

	clusters := DetectCommunities(neighbors, names, "test")
	if len(clusters) != 1 {
		t.Fatalf("expected 1 community for connected pair, got %d", len(clusters))
	}
	if len(clusters[0].MemberUUIDs) != 2 {
		t.Errorf("expected 2 members, got %d", len(clusters[0].MemberUUIDs))
	}
}

func TestDetectCommunities_Empty(t *testing.T) {
	clusters := DetectCommunities(nil, nil, "test")
	if len(clusters) != 0 {
		t.Fatalf("expected 0 communities for empty graph, got %d", len(clusters))
	}
}

func TestDetectCommunities_MissingName(t *testing.T) {
	neighbors := map[string][]string{
		"a": {"b", "c", "d"},
		"b": {"a", "c", "d"},
		"c": {"a", "b", "d"},
		"d": {"a", "b", "c"},
	}
	clusters := DetectCommunities(neighbors, map[string]string{}, "test")
	if len(clusters) != 1 {
		t.Fatalf("expected 1 community, got %d", len(clusters))
	}
	if clusters[0].Name != "Community 0" {
		t.Errorf("expected fallback name 'Community 0', got %q", clusters[0].Name)
	}
}

func TestDetectCommunities_UUIDFormat(t *testing.T) {
	neighbors := map[string][]string{
		"a": {"b", "c", "d"},
		"b": {"a", "c", "d"},
		"c": {"a", "b", "d"},
		"d": {"a", "b", "c"},
	}
	names := map[string]string{"a": "A", "b": "B", "c": "C", "d": "D"}

	clusters := DetectCommunities(neighbors, names, "mygroup")
	if len(clusters) != 1 {
		t.Fatalf("expected 1 community, got %d", len(clusters))
	}
	if clusters[0].CommunityUUID != "comm_mygroup_0" {
		t.Errorf("expected UUID 'comm_mygroup_0', got %q", clusters[0].CommunityUUID)
	}
}
