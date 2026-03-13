package router

import (
	"context"
	"math"
	"slices"
	"sort"
	"strings"

	"github.com/soundprediction/predicato/pkg/crossencoder"
	"github.com/soundprediction/predicato/pkg/embedder"
)

// TopicMatch represents a match between a query and a client's topics.
type TopicMatch struct {
	// ClientName is the name of the matched client.
	ClientName string `json:"client_name"`

	// MatchedTopics are the specific topics that matched.
	MatchedTopics []string `json:"matched_topics"`

	// Confidence is the confidence score (0.0 to 1.0).
	Confidence float64 `json:"confidence"`
}

// TopicClassifier classifies queries to determine which clients should handle them.
type TopicClassifier interface {
	// Classify returns a list of topic matches for the given query, sorted by confidence descending.
	Classify(ctx context.Context, query string) ([]TopicMatch, error)
}

// KeywordTopicClassifier uses simple keyword matching for topic classification.
type KeywordTopicClassifier struct {
	clientTopics map[string][]string
	topicClients map[string][]string
}

// NewKeywordTopicClassifier creates a new keyword-based topic classifier.
func NewKeywordTopicClassifier(configs []ClientConfig) *KeywordTopicClassifier {
	tc := &KeywordTopicClassifier{
		clientTopics: make(map[string][]string),
		topicClients: make(map[string][]string),
	}

	for _, cfg := range configs {
		if len(cfg.Topics) > 0 {
			tc.clientTopics[cfg.Name] = cfg.Topics
			for _, topic := range cfg.Topics {
				normalizedTopic := strings.ToLower(topic)
				tc.topicClients[normalizedTopic] = append(tc.topicClients[normalizedTopic], cfg.Name)
			}
		}
	}

	return tc
}

// Classify implements TopicClassifier using keyword matching.
func (tc *KeywordTopicClassifier) Classify(_ context.Context, query string) ([]TopicMatch, error) {
	queryWords := tokenizeQuery(query)
	if len(queryWords) == 0 {
		return nil, nil
	}

	clientMatches := make(map[string][]string)

	for _, word := range queryWords {
		if clients, ok := tc.topicClients[word]; ok {
			for _, clientName := range clients {
				for _, topic := range tc.clientTopics[clientName] {
					if strings.ToLower(topic) == word {
						clientMatches[clientName] = appendUnique(clientMatches[clientName], topic)
					}
				}
			}
		}
	}

	matches := make([]TopicMatch, 0, len(clientMatches))
	for clientName, matchedTopics := range clientMatches {
		totalTopics := len(tc.clientTopics[clientName])
		if totalTopics == 0 {
			continue
		}

		topicCoverage := float64(len(matchedTopics)) / float64(totalTopics)
		queryRelevance := float64(len(matchedTopics)) / float64(len(queryWords))
		confidence := 0.7*topicCoverage + 0.3*queryRelevance

		matches = append(matches, TopicMatch{
			ClientName:    clientName,
			MatchedTopics: matchedTopics,
			Confidence:    confidence,
		})
	}

	sort.Slice(matches, func(i, j int) bool {
		return matches[i].Confidence > matches[j].Confidence
	})

	return matches, nil
}

// EmbeddingTopicClassifier uses semantic embeddings for topic classification.
type EmbeddingTopicClassifier struct {
	embedder           embedder.Client
	clientTopics       map[string][]string
	topicEmbeddings    map[string][]float32
	fallbackClassifier *KeywordTopicClassifier
}

// NewEmbeddingTopicClassifier creates a new embedding-based topic classifier.
// It pre-computes embeddings for all client topics for efficient classification.
func NewEmbeddingTopicClassifier(
	configs []ClientConfig,
	embedderClient embedder.Client,
) (*EmbeddingTopicClassifier, error) {
	tc := &EmbeddingTopicClassifier{
		embedder:           embedderClient,
		clientTopics:       make(map[string][]string),
		topicEmbeddings:    make(map[string][]float32),
		fallbackClassifier: NewKeywordTopicClassifier(configs),
	}

	ctx := context.Background()

	for _, cfg := range configs {
		if len(cfg.Topics) > 0 {
			tc.clientTopics[cfg.Name] = cfg.Topics

			topicText := strings.Join(cfg.Topics, " ")
			embedding, err := embedderClient.EmbedSingle(ctx, topicText)
			if err != nil {
				continue
			}
			tc.topicEmbeddings[cfg.Name] = embedding
		}
	}

	return tc, nil
}

// Classify implements TopicClassifier using semantic similarity.
func (tc *EmbeddingTopicClassifier) Classify(ctx context.Context, query string) ([]TopicMatch, error) {
	queryEmbedding, err := tc.embedder.EmbedSingle(ctx, query)
	if err != nil {
		return tc.fallbackClassifier.Classify(ctx, query)
	}

	matches := make([]TopicMatch, 0, len(tc.topicEmbeddings))

	for clientName, topicEmbedding := range tc.topicEmbeddings {
		similarity := cosineSimilarity(queryEmbedding, topicEmbedding)

		if similarity > 0.0 {
			var matchedTopics []string
			keywordMatches, _ := tc.fallbackClassifier.Classify(ctx, query)
			for _, km := range keywordMatches {
				if km.ClientName == clientName {
					matchedTopics = km.MatchedTopics
					break
				}
			}

			matches = append(matches, TopicMatch{
				ClientName:    clientName,
				MatchedTopics: matchedTopics,
				Confidence:    similarity,
			})
		}
	}

	sort.Slice(matches, func(i, j int) bool {
		return matches[i].Confidence > matches[j].Confidence
	})

	return matches, nil
}

// CrossEncoderTopicClassifier uses a cross-encoder reranker model to score
// a query against each topic's routing descriptor text.
type CrossEncoderTopicClassifier struct {
	reranker           crossencoder.Client
	clientDescriptors  map[string]string
	clientTopics       map[string][]string
	fallbackClassifier *KeywordTopicClassifier
	clientOrder        []string
}

// NewCrossEncoderTopicClassifier creates a new cross-encoder based topic classifier.
func NewCrossEncoderTopicClassifier(
	configs []ClientConfig,
	rerankerClient crossencoder.Client,
	routingTexts map[string]string,
) *CrossEncoderTopicClassifier {
	tc := &CrossEncoderTopicClassifier{
		reranker:           rerankerClient,
		clientDescriptors:  make(map[string]string),
		clientTopics:       make(map[string][]string),
		fallbackClassifier: NewKeywordTopicClassifier(configs),
	}

	for _, cfg := range configs {
		if len(cfg.Topics) > 0 {
			tc.clientTopics[cfg.Name] = cfg.Topics
		}

		if text, ok := routingTexts[cfg.Name]; ok && text != "" {
			tc.clientDescriptors[cfg.Name] = text
			tc.clientOrder = append(tc.clientOrder, cfg.Name)
		} else if len(cfg.Topics) > 0 {
			tc.clientDescriptors[cfg.Name] = strings.Join(cfg.Topics, " ")
			tc.clientOrder = append(tc.clientOrder, cfg.Name)
		}
	}

	return tc
}

// Classify implements TopicClassifier using cross-encoder scoring.
func (tc *CrossEncoderTopicClassifier) Classify(ctx context.Context, query string) ([]TopicMatch, error) {
	if len(tc.clientOrder) == 0 {
		return nil, nil
	}

	passages := make([]string, len(tc.clientOrder))
	for i, name := range tc.clientOrder {
		passages[i] = tc.clientDescriptors[name]
	}

	ranked, err := tc.reranker.Rank(ctx, query, passages)
	if err != nil {
		return tc.fallbackClassifier.Classify(ctx, query)
	}

	passageToClient := make(map[string]string, len(tc.clientOrder))
	for i, name := range tc.clientOrder {
		passageToClient[passages[i]] = name
	}

	matches := make([]TopicMatch, 0, len(ranked))
	for _, rp := range ranked {
		clientName, ok := passageToClient[rp.Passage]
		if !ok {
			continue
		}

		matches = append(matches, TopicMatch{
			ClientName:    clientName,
			MatchedTopics: tc.clientTopics[clientName],
			Confidence:    rp.Score,
		})
	}

	sort.Slice(matches, func(i, j int) bool {
		return matches[i].Confidence > matches[j].Confidence
	})

	return matches, nil
}

// NewTopicClassifier creates a topic classifier based on the router strategy.
func NewTopicClassifier(
	configs []ClientConfig,
	routerConfig RouterConfig,
	embedderClient embedder.Client,
	rerankerClient crossencoder.Client,
	routingTexts map[string]string,
) (TopicClassifier, error) {
	strategy := routerConfig.GetStrategyOrDefault()

	switch strategy {
	case "topic_based":
		if embedderClient != nil {
			return NewEmbeddingTopicClassifier(configs, embedderClient)
		}
		return NewKeywordTopicClassifier(configs), nil

	case "keyword":
		return NewKeywordTopicClassifier(configs), nil

	case "embedding":
		if embedderClient == nil {
			return NewKeywordTopicClassifier(configs), nil
		}
		return NewEmbeddingTopicClassifier(configs, embedderClient)

	case "cross_encoder":
		if rerankerClient != nil {
			return NewCrossEncoderTopicClassifier(configs, rerankerClient, routingTexts), nil
		}
		return NewKeywordTopicClassifier(configs), nil

	default:
		return NewKeywordTopicClassifier(configs), nil
	}
}

// Helper functions

func tokenizeQuery(query string) []string {
	query = strings.ToLower(query)

	query = strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == ' ' {
			return r
		}
		return ' '
	}, query)

	words := strings.Fields(query)

	stopWords := map[string]bool{
		"a": true, "an": true, "the": true, "is": true, "are": true,
		"was": true, "were": true, "be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
		"may": true, "might": true, "must": true, "shall": true,
		"i": true, "you": true, "he": true, "she": true, "it": true,
		"we": true, "they": true, "what": true, "which": true, "who": true,
		"when": true, "where": true, "why": true, "how": true,
		"in": true, "on": true, "at": true, "by": true, "for": true,
		"with": true, "about": true, "against": true, "between": true,
		"into": true, "through": true, "during": true, "before": true,
		"after": true, "above": true, "below": true, "to": true, "from": true,
		"up": true, "down": true, "out": true, "off": true, "over": true,
		"under": true, "again": true, "further": true, "then": true, "once": true,
		"and": true, "but": true, "or": true, "nor": true, "so": true, "yet": true,
		"of": true, "as": true, "if": true, "than": true,
	}

	filtered := make([]string, 0, len(words))
	for _, word := range words {
		if len(word) > 2 && !stopWords[word] {
			filtered = append(filtered, word)
		}
	}

	return filtered
}

func appendUnique(slice []string, item string) []string {
	if slices.Contains(slice, item) {
		return slice
	}
	return append(slice, item)
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
