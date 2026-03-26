package driver

import "fmt"

// CommunityCluster represents a detected community with its metadata.
type CommunityCluster struct {
	CommunityUUID string
	Name          string
	Content       string
	MemberUUIDs   []string
}

// DetectCommunities runs label propagation on an adjacency list and returns
// community clusters. Each node starts with its own label; labels propagate
// to the most common neighbor label. Only multi-node clusters are returned.
//
// neighbors: bidirectional adjacency list (node UUID -> neighbor UUIDs)
// entityNames: node UUID -> human-readable name (for naming communities)
// groupID: used for generating deterministic community UUIDs
func DetectCommunities(neighbors map[string][]string, entityNames map[string]string, groupID string) []CommunityCluster {
	if len(neighbors) == 0 {
		return nil
	}

	// Step 1: Label propagation.
	// Each node starts with its own UUID as its label.
	label := make(map[string]string, len(neighbors))
	for node := range neighbors {
		label[node] = node
	}

	const maxIter = 100
	for iter := 0; iter < maxIter; iter++ {
		changed := false
		for node, nbs := range neighbors {
			// Count community labels among neighbors.
			counts := make(map[string]int, len(nbs))
			for _, nb := range nbs {
				counts[label[nb]]++
			}
			// Find the label with the highest count (break ties by label string).
			bestLabel := label[node]
			bestCount := 0
			for lbl, cnt := range counts {
				if cnt > bestCount || (cnt == bestCount && lbl < bestLabel) {
					bestLabel = lbl
					bestCount = cnt
				}
			}
			if bestLabel != label[node] {
				if bestCount > 1 || bestLabel < label[node] {
					label[node] = bestLabel
					changed = true
				}
			}
		}
		if !changed {
			break
		}
	}

	// Step 2: Group nodes by their community label, keep only multi-node communities.
	clusters := make(map[string][]string)
	for node, lbl := range label {
		clusters[lbl] = append(clusters[lbl], node)
	}

	var result []CommunityCluster
	i := 0
	for _, members := range clusters {
		if len(members) <= 1 {
			continue
		}

		// Find the most-connected member for the community name.
		bestUUID := members[0]
		bestDegree := len(neighbors[bestUUID])
		for _, m := range members[1:] {
			if deg := len(neighbors[m]); deg > bestDegree {
				bestUUID = m
				bestDegree = deg
			}
		}

		name := entityNames[bestUUID]
		if name == "" {
			name = fmt.Sprintf("Community %d", i)
		}

		result = append(result, CommunityCluster{
			CommunityUUID: fmt.Sprintf("comm_%s_%d", groupID, i),
			Name:          name,
			Content:       fmt.Sprintf("Community of %d related entities", len(members)),
			MemberUUIDs:   members,
		})
		i++
	}

	return result
}
