package search

import (
	"context"
	"math"
	"sort"

	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/types"
	"github.com/soundprediction/predicato/pkg/utils"
)

// RRFResult represents the result of RRF reranking
type RRFResult struct {
	UUIDs  []string
	Scores []float64
}

// MMRResult represents the result of MMR reranking
type MMRResult struct {
	UUIDs  []string
	Scores []float64
}

// RRF (Reciprocal Rank Fusion) reranks search results by combining multiple ranked lists
func RRF(results [][]string, rankConstant int, minScore float64) ([]string, []float64) {
	if rankConstant <= 0 {
		rankConstant = DefaultRankConstant
	}

	scores := make(map[string]float64)

	// Calculate RRF scores
	for _, result := range results {
		for i, uuid := range result {
			if _, exists := scores[uuid]; !exists {
				scores[uuid] = 0
			}
			scores[uuid] += 1.0 / float64(i+rankConstant)
		}
	}

	// Create sorted list of UUID-score pairs
	type uuidScore struct {
		uuid  string
		score float64
	}

	var scoredUUIDs []uuidScore
	for uuid, score := range scores {
		if score >= minScore {
			scoredUUIDs = append(scoredUUIDs, uuidScore{uuid: uuid, score: score})
		}
	}

	// Sort by score (descending)
	sort.Slice(scoredUUIDs, func(i, j int) bool {
		return scoredUUIDs[i].score > scoredUUIDs[j].score
	})

	// Extract UUIDs and scores
	uuids := make([]string, len(scoredUUIDs))
	scoreList := make([]float64, len(scoredUUIDs))
	for i, item := range scoredUUIDs {
		uuids[i] = item.uuid
		scoreList[i] = item.score
	}

	return uuids, scoreList
}

// NodeDistanceReranker reranks nodes based on their distance from a center node
func NodeDistanceReranker(ctx context.Context, driver driver.GraphDriver, nodeUUIDs []string, centerNodeUUID string, minScore float64) ([]string, []float64, error) {
	// Filter out the center node UUID
	filteredUUIDs := make([]string, 0, len(nodeUUIDs))
	for _, uuid := range nodeUUIDs {
		if uuid != centerNodeUUID {
			filteredUUIDs = append(filteredUUIDs, uuid)
		}
	}

	scores := make(map[string]float64)
	scores[centerNodeUUID] = 0.0 // Center node has distance 0

	// For now, implement a simple distance calculation
	// In a full implementation, this would use graph traversal queries
	// to find shortest paths between nodes

	// Assign default distances (this is a simplified implementation)
	for _, uuid := range filteredUUIDs {
		if uuid != centerNodeUUID {
			// Default distance of 1 for simplicity
			scores[uuid] = 1.0
		}
	}

	// Sort by distance (lower is better)
	type uuidDistance struct {
		uuid     string
		distance float64
	}

	var sortedNodes []uuidDistance
	for _, uuid := range filteredUUIDs {
		distance := scores[uuid]
		if distance == 0 {
			distance = math.Inf(1) // Infinite distance if not connected
		}
		sortedNodes = append(sortedNodes, uuidDistance{uuid: uuid, distance: distance})
	}

	sort.Slice(sortedNodes, func(i, j int) bool {
		return sortedNodes[i].distance < sortedNodes[j].distance
	})

	// Add center node at the beginning if it was in the original list
	containsCenter := false
	for _, uuid := range nodeUUIDs {
		if uuid == centerNodeUUID {
			containsCenter = true
			break
		}
	}

	var resultUUIDs []string
	var resultScores []float64

	if containsCenter {
		resultUUIDs = append(resultUUIDs, centerNodeUUID)
		resultScores = append(resultScores, 0.1) // Small positive score for center
	}

	// Add other nodes with inverted distance scores
	for _, item := range sortedNodes {
		score := 1.0 / (1.0 + item.distance) // Invert distance to get score
		if score >= minScore {
			resultUUIDs = append(resultUUIDs, item.uuid)
			resultScores = append(resultScores, score)
		}
	}

	return resultUUIDs, resultScores, nil
}

// EpisodeMentionsReranker reranks nodes based on how many episodes mention them
func EpisodeMentionsReranker(ctx context.Context, driver driver.GraphDriver, nodeUUIDs [][]string, minScore float64) ([]string, []float64, error) {
	// Use RRF as preliminary ranking
	sortedUUIDs, _ := RRF(nodeUUIDs, DefaultRankConstant, 0)

	scores := make(map[string]float64)

	// For now, assign default episode mention counts
	// In a full implementation, this would query the database for actual episode mentions
	for i, uuid := range sortedUUIDs {
		// Assign decreasing scores based on position (simulating episode mentions)
		mentionCount := float64(len(sortedUUIDs) - i)
		scores[uuid] = mentionCount
	}

	// Sort by mention count (descending)
	type uuidMentions struct {
		uuid     string
		mentions float64
	}

	var sortedNodes []uuidMentions
	for _, uuid := range sortedUUIDs {
		mentions := scores[uuid]
		if mentions >= minScore {
			sortedNodes = append(sortedNodes, uuidMentions{uuid: uuid, mentions: mentions})
		}
	}

	sort.Slice(sortedNodes, func(i, j int) bool {
		return sortedNodes[i].mentions > sortedNodes[j].mentions
	})

	// Extract results
	resultUUIDs := make([]string, len(sortedNodes))
	resultScores := make([]float64, len(sortedNodes))
	for i, item := range sortedNodes {
		resultUUIDs[i] = item.uuid
		resultScores[i] = item.mentions
	}

	return resultUUIDs, resultScores, nil
}

// MaximalMarginalRelevance (MMR) reranks results to balance relevance and diversity
func MaximalMarginalRelevance(queryVector []float32, candidates map[string][]float32, mmrLambda float64, minScore float64) ([]string, []float64) {
	if mmrLambda == 0 {
		mmrLambda = DefaultMMRLambda
	}

	if len(candidates) == 0 {
		return []string{}, []float64{}
	}

	// Convert to internal format for calculations
	candidateVectors := make(map[string][]float32)
	uuids := make([]string, 0, len(candidates))

	for uuid, embedding := range candidates {
		normalized := utils.Normalize(embedding)
		if normalized == nil {
			normalized = embedding
		}
		candidateVectors[uuid] = normalized
		uuids = append(uuids, uuid)
	}

	normalizedQuery := utils.Normalize(queryVector)
	if normalizedQuery == nil {
		normalizedQuery = queryVector
	}

	// Calculate similarity matrix between candidates
	similarityMatrix := make(map[string]map[string]float64)
	for _, uuid1 := range uuids {
		similarityMatrix[uuid1] = make(map[string]float64)
		for _, uuid2 := range uuids {
			if uuid1 == uuid2 {
				similarityMatrix[uuid1][uuid2] = 1.0
			} else {
				sim := utils.CosineSimilarity(candidateVectors[uuid1], candidateVectors[uuid2])
				similarityMatrix[uuid1][uuid2] = sim
			}
		}
	}

	// Calculate MMR scores
	mmrScores := make(map[string]float64)
	for _, uuid := range uuids {
		// Query-document similarity
		queryDocSim := utils.CosineSimilarity(normalizedQuery, candidateVectors[uuid])

		// Find maximum similarity to any other document
		maxSim := 0.0
		for _, otherUUID := range uuids {
			if uuid != otherUUID {
				sim := similarityMatrix[uuid][otherUUID]
				if sim > maxSim {
					maxSim = sim
				}
			}
		}

		// MMR formula: λ * query_sim - (1-λ) * max_doc_sim
		mmr := mmrLambda*queryDocSim - (1-mmrLambda)*maxSim
		mmrScores[uuid] = mmr
	}

	// Sort by MMR score (descending)
	type uuidMMR struct {
		uuid string
		mmr  float64
	}

	var sortedUUIDs []uuidMMR
	for _, uuid := range uuids {
		mmr := mmrScores[uuid]
		if mmr >= minScore {
			sortedUUIDs = append(sortedUUIDs, uuidMMR{uuid: uuid, mmr: mmr})
		}
	}

	sort.Slice(sortedUUIDs, func(i, j int) bool {
		return sortedUUIDs[i].mmr > sortedUUIDs[j].mmr
	})

	// Extract results
	resultUUIDs := make([]string, len(sortedUUIDs))
	resultScores := make([]float64, len(sortedUUIDs))
	for i, item := range sortedUUIDs {
		resultUUIDs[i] = item.uuid
		resultScores[i] = item.mmr
	}

	return resultUUIDs, resultScores
}

// CrossEncoderReranker uses an LLM to rerank results (placeholder implementation)
func CrossEncoderReranker(ctx context.Context, query string, candidates []*types.Node, minScore float64, limit int) ([]*types.Node, []float64, error) {
	// This would use an LLM to score each candidate against the query
	// For now, return the candidates as-is with default scores

	if len(candidates) > limit {
		candidates = candidates[:limit]
	}

	scores := make([]float64, len(candidates))
	for i := range scores {
		// Default score - in real implementation this would use LLM scoring
		scores[i] = 1.0
	}

	return candidates, scores, nil
}

// GetEmbeddingsForNodes retrieves embeddings for the given nodes from the database
func GetEmbeddingsForNodes(ctx context.Context, driver driver.GraphDriver, nodes []*types.Node) (map[string][]float32, error) {
	embeddings := make(map[string][]float32)

	for _, node := range nodes {
		// Extract embeddings from node metadata if available
		if node.Metadata != nil {
			if embeddingData, exists := node.Metadata["name_embedding"]; exists {
				if embedding := toFloat32Slice(embeddingData); embedding != nil {
					embeddings[node.Uuid] = embedding
				}
			}
		}
	}

	return embeddings, nil
}

// GetEmbeddingsForEdges retrieves embeddings for the given edges from the database
func GetEmbeddingsForEdges(ctx context.Context, driver driver.GraphDriver, edges []*types.Edge) (map[string][]float32, error) {
	embeddings := make(map[string][]float32)

	for _, edge := range edges {
		// Extract embeddings from edge metadata if available
		if edge.Metadata != nil {
			if embeddingData, exists := edge.Metadata["name_embedding"]; exists {
				if embedding := toFloat32Slice(embeddingData); embedding != nil {
					embeddings[edge.Uuid] = embedding
				}
			}
		}
	}

	return embeddings, nil
}

// GetEmbeddingsForCommunities retrieves embeddings for community nodes
func GetEmbeddingsForCommunities(ctx context.Context, driver driver.GraphDriver, communities []*types.Node) (map[string][]float32, error) {
	embeddings := make(map[string][]float32)

	for _, community := range communities {
		// Extract embeddings from community metadata if available
		if community.Metadata != nil {
			if embeddingData, exists := community.Metadata["name_embedding"]; exists {
				if embedding := toFloat32Slice(embeddingData); embedding != nil {
					embeddings[community.Uuid] = embedding
				}
			}
		}
	}

	return embeddings, nil
}
