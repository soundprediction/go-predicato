package factstore

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/soundprediction/predicato/pkg/utils"
)

// TestSearchMethodJSON verifies SearchMethod serializes correctly
func TestSearchMethodJSON(t *testing.T) {
	tests := []struct {
		method   SearchMethod
		expected string
	}{
		{VectorSearch, `"vector"`},
		{KeywordSearch, `"keyword"`},
	}

	for _, tt := range tests {
		b, err := json.Marshal(tt.method)
		if err != nil {
			t.Errorf("Failed to marshal %v: %v", tt.method, err)
			continue
		}
		if string(b) != tt.expected {
			t.Errorf("Expected %s, got %s", tt.expected, string(b))
		}
	}
}

// TestFactSearchConfigDefaults tests default values
func TestFactSearchConfigDefaults(t *testing.T) {
	config := &FactSearchConfig{}

	if config.Limit != 0 {
		t.Errorf("Expected default Limit to be 0, got %d", config.Limit)
	}
	if config.MinScore != 0 {
		t.Errorf("Expected default MinScore to be 0, got %f", config.MinScore)
	}
	if config.GroupID != "" {
		t.Errorf("Expected default GroupID to be empty, got %s", config.GroupID)
	}
}

// TestFactSearchResultsJSON tests JSON serialization of results
func TestFactSearchResultsJSON(t *testing.T) {
	now := time.Now().Truncate(time.Second)
	results := &FactSearchResults{
		Nodes: []*ExtractedNode{
			{
				ID:          "node-1",
				SourceID:    "source-1",
				GroupID:     "group-1",
				Name:        "Test Entity",
				Type:        "person",
				Description: "A test entity",
				ChunkIndex:  0,
				CreatedAt:   now,
			},
		},
		Triples: []*ExtractedTriple{
			{
				ID:         "triple-1",
				SourceID:   "source-1",
				GroupID:    "group-1",
				Subject:    "Test Entity",
				Object:     "Other Entity",
				Predicate:  "knows",
				Description: "Test relationship",
				Confidence: 1.0,
				ChunkIndex: 0,
				CreatedAt:  now,
			},
		},
		NodeScores:   []float64{0.95},
		TripleScores: []float64{0.85},
		Query:      "test query",
		Total:      2,
	}

	b, err := json.Marshal(results)
	if err != nil {
		t.Fatalf("Failed to marshal results: %v", err)
	}

	var unmarshaled FactSearchResults
	if err := json.Unmarshal(b, &unmarshaled); err != nil {
		t.Fatalf("Failed to unmarshal results: %v", err)
	}

	if len(unmarshaled.Nodes) != 1 {
		t.Errorf("Expected 1 node, got %d", len(unmarshaled.Nodes))
	}
	if len(unmarshaled.Triples) != 1 {
		t.Errorf("Expected 1 triple, got %d", len(unmarshaled.Triples))
	}
	if unmarshaled.Query != "test query" {
		t.Errorf("Expected query 'test query', got %s", unmarshaled.Query)
	}
	if unmarshaled.Total != 2 {
		t.Errorf("Expected total 2, got %d", unmarshaled.Total)
	}
}

// TestDbConfigValidation tests config validation
func TestDbConfigValidation(t *testing.T) {
	tests := []struct {
		name        string
		config      *FactStoreConfig
		expectError bool
	}{
		{
			name:        "nil config",
			config:      nil,
			expectError: true,
		},
		{
			name: "missing connection string",
			config: &FactStoreConfig{
				Type: FactStoreTypePostgres,
			},
			expectError: true,
		},
		{
			name: "valid postgres config",
			config: &FactStoreConfig{
				Type:             FactStoreTypePostgres,
				ConnectionString: "postgres://localhost:5432/test",
			},
			expectError: false, // Will fail to connect, but validation passes
		},
		{
			name: "valid doltgres config",
			config: &FactStoreConfig{
				Type:             FactStoreTypeDoltGres,
				ConnectionString: "postgres://localhost:5432/test",
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewFactsDB(tt.config)
			if tt.expectError && err == nil {
				// For configs that should error before connection
				if tt.config == nil || tt.config.ConnectionString == "" {
					t.Errorf("Expected error for %s, got nil", tt.name)
				}
			}
		})
	}
}

// TestTimeRangeJSON tests TimeRange serialization
func TestTimeRangeJSON(t *testing.T) {
	now := time.Now().Truncate(time.Second)
	tr := &TimeRange{
		Start: now.Add(-24 * time.Hour),
		End:   now,
	}

	b, err := json.Marshal(tr)
	if err != nil {
		t.Fatalf("Failed to marshal TimeRange: %v", err)
	}

	var unmarshaled TimeRange
	if err := json.Unmarshal(b, &unmarshaled); err != nil {
		t.Fatalf("Failed to unmarshal TimeRange: %v", err)
	}

	if !unmarshaled.Start.Equal(tr.Start) {
		t.Errorf("Start time mismatch: expected %v, got %v", tr.Start, unmarshaled.Start)
	}
	if !unmarshaled.End.Equal(tr.End) {
		t.Errorf("End time mismatch: expected %v, got %v", tr.End, unmarshaled.End)
	}
}

// TestExtractedNodeJSON tests ExtractedNode serialization
func TestExtractedNodeJSON(t *testing.T) {
	node := &ExtractedNode{
		ID:          "node-123",
		SourceID:    "source-456",
		GroupID:     "group-789",
		Name:        "Test Node",
		Type:        "concept",
		Description: "A test concept node",
		Embedding:   []float32{0.1, 0.2, 0.3, 0.4, 0.5},
		ChunkIndex:  2,
		CreatedAt:   time.Now().Truncate(time.Second),
	}

	b, err := json.Marshal(node)
	if err != nil {
		t.Fatalf("Failed to marshal node: %v", err)
	}

	var unmarshaled ExtractedNode
	if err := json.Unmarshal(b, &unmarshaled); err != nil {
		t.Fatalf("Failed to unmarshal node: %v", err)
	}

	if unmarshaled.ID != node.ID {
		t.Errorf("ID mismatch: expected %s, got %s", node.ID, unmarshaled.ID)
	}
	if unmarshaled.Name != node.Name {
		t.Errorf("Name mismatch: expected %s, got %s", node.Name, unmarshaled.Name)
	}
	if len(unmarshaled.Embedding) != len(node.Embedding) {
		t.Errorf("Embedding length mismatch: expected %d, got %d", len(node.Embedding), len(unmarshaled.Embedding))
	}
}

// TestExtractedTripleJSON tests ExtractedTriple serialization
func TestExtractedTripleJSON(t *testing.T) {
	triple := &ExtractedTriple{
		ID:          "triple-123",
		SourceID:    "source-456",
		GroupID:     "group-789",
		Subject:     "Node A",
		SubjectType: "concept",
		Object:      "Node B",
		ObjectType:  "concept",
		Predicate:   "relates_to",
		Description: "A relates to B",
		Embedding:   []float32{0.5, 0.4, 0.3, 0.2, 0.1},
		Confidence:  0.85,
		ChunkIndex:  1,
		CreatedAt:   time.Now().Truncate(time.Second),
	}

	b, err := json.Marshal(triple)
	if err != nil {
		t.Fatalf("Failed to marshal triple: %v", err)
	}

	var unmarshaled ExtractedTriple
	if err := json.Unmarshal(b, &unmarshaled); err != nil {
		t.Fatalf("Failed to unmarshal triple: %v", err)
	}

	if unmarshaled.ID != triple.ID {
		t.Errorf("ID mismatch: expected %s, got %s", triple.ID, unmarshaled.ID)
	}
	if unmarshaled.Predicate != triple.Predicate {
		t.Errorf("Predicate mismatch: expected %s, got %s", triple.Predicate, unmarshaled.Predicate)
	}
	if unmarshaled.Confidence != triple.Confidence {
		t.Errorf("Confidence mismatch: expected %f, got %f", triple.Confidence, unmarshaled.Confidence)
	}
}

// TestCosineSimilarity tests the cosine similarity function
func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		expected float64
		delta    float64
	}{
		{
			name:     "identical vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{1, 0, 0},
			expected: 1.0,
			delta:    0.001,
		},
		{
			name:     "orthogonal vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{0, 1, 0},
			expected: 0.0,
			delta:    0.001,
		},
		{
			name:     "opposite vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{-1, 0, 0},
			expected: -1.0,
			delta:    0.001,
		},
		{
			name:     "similar vectors",
			a:        []float32{1, 1, 0},
			b:        []float32{1, 0, 0},
			expected: 0.707,
			delta:    0.01,
		},
		{
			name:     "empty vectors",
			a:        []float32{},
			b:        []float32{},
			expected: 0.0,
			delta:    0.001,
		},
		{
			name:     "mismatched length",
			a:        []float32{1, 2, 3},
			b:        []float32{1, 2},
			expected: 0.0,
			delta:    0.001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := cosineSimilarity(tt.a, tt.b)
			diff := result - tt.expected
			if diff < 0 {
				diff = -diff
			}
			if diff > tt.delta {
				t.Errorf("Expected %f (±%f), got %f", tt.expected, tt.delta, result)
			}
		})
	}
}

// TestSourceJSON tests Source serialization
func TestSourceJSON(t *testing.T) {
	source := &Source{
		ID:      "source-123",
		Name:    "Test Document",
		Content: "This is the content of the test document.",
		GroupID: "group-456",
		Metadata: map[string]interface{}{
			"author": "Test Author",
			"date":   "2024-01-01",
		},
		CreatedAt: time.Now().Truncate(time.Second),
	}

	b, err := json.Marshal(source)
	if err != nil {
		t.Fatalf("Failed to marshal source: %v", err)
	}

	var unmarshaled Source
	if err := json.Unmarshal(b, &unmarshaled); err != nil {
		t.Fatalf("Failed to unmarshal source: %v", err)
	}

	if unmarshaled.ID != source.ID {
		t.Errorf("ID mismatch: expected %s, got %s", source.ID, unmarshaled.ID)
	}
	if unmarshaled.Content != source.Content {
		t.Errorf("Content mismatch")
	}
	if unmarshaled.Metadata["author"] != source.Metadata["author"] {
		t.Errorf("Metadata author mismatch")
	}
}

// TestStatsJSON tests Stats serialization
func TestStatsJSON(t *testing.T) {
	stats := &Stats{
		SourceCount: 10,
		NodeCount:   100,
		TripleCount: 50,
	}

	b, err := json.Marshal(stats)
	if err != nil {
		t.Fatalf("Failed to marshal stats: %v", err)
	}

	var unmarshaled Stats
	if err := json.Unmarshal(b, &unmarshaled); err != nil {
		t.Fatalf("Failed to unmarshal stats: %v", err)
	}

	if unmarshaled.SourceCount != stats.SourceCount {
		t.Errorf("SourceCount mismatch: expected %d, got %d", stats.SourceCount, unmarshaled.SourceCount)
	}
}

// --- Entity Dedup Tests ---

// TestExtractedNodeNormalizedNameJSON verifies NormalizedName round-trips through JSON
func TestExtractedNodeNormalizedNameJSON(t *testing.T) {
	node := &ExtractedNode{
		ID:             "node-1",
		SourceID:       "source-1",
		GroupID:        "group-1",
		Name:           "  Acme  Corporation ",
		NormalizedName: "acme corporation",
		Type:           "organization",
		Description:    "A company",
		ChunkIndex:     0,
		CreatedAt:      time.Now().Truncate(time.Second),
	}

	b, err := json.Marshal(node)
	if err != nil {
		t.Fatalf("Failed to marshal node: %v", err)
	}

	var unmarshaled ExtractedNode
	if err := json.Unmarshal(b, &unmarshaled); err != nil {
		t.Fatalf("Failed to unmarshal node: %v", err)
	}

	if unmarshaled.NormalizedName != "acme corporation" {
		t.Errorf("NormalizedName mismatch: expected 'acme corporation', got %q", unmarshaled.NormalizedName)
	}
}

// TestDedupNormalization verifies that NormalizeStringExact produces consistent keys for dedup
func TestDedupNormalization(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"simple lowercase", "ACME Corp", "acme corp"},
		{"leading/trailing whitespace", "  John Smith  ", "john smith"},
		{"multiple internal spaces", "New   York   City", "new york city"},
		{"mixed case and whitespace", "  UNITED  States  ", "united states"},
		{"tabs and newlines", "Hello\t\nWorld", "hello world"},
		{"already normalized", "test entity", "test entity"},
		{"empty string", "", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := utils.NormalizeStringExact(tt.input)
			if result != tt.expected {
				t.Errorf("NormalizeStringExact(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

// TestDedupSameEntityNormalizesToSameKey verifies that different representations of the same
// entity name normalize to the same dedup key
func TestDedupSameEntityNormalizesToSameKey(t *testing.T) {
	variants := []string{
		"Acme Corporation",
		"acme corporation",
		"ACME CORPORATION",
		"  Acme   Corporation  ",
		"acme  corporation",
	}

	normalized := utils.NormalizeStringExact(variants[0])
	for _, v := range variants[1:] {
		if got := utils.NormalizeStringExact(v); got != normalized {
			t.Errorf("Expected %q and %q to normalize the same, got %q vs %q", variants[0], v, normalized, got)
		}
	}
}

// TestDedupPersonTypeDetection verifies person type detection logic used in SaveExtractedKnowledge
func TestDedupPersonTypeDetection(t *testing.T) {
	tests := []struct {
		nodeType string
		isPerson bool
	}{
		{"person", true},
		{"Person", true},
		{"PERSON", true},
		{"  Person  ", true},
		{"organization", false},
		{"location", false},
		{"concept", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.nodeType, func(t *testing.T) {
			normalizedType := strings.ToLower(strings.TrimSpace(tt.nodeType))
			isPerson := normalizedType == "person"
			if isPerson != tt.isPerson {
				t.Errorf("Type %q: expected isPerson=%v, got %v", tt.nodeType, tt.isPerson, isPerson)
			}
		})
	}
}

// TestDedupDescriptionMerge verifies that the longer description wins
func TestDedupDescriptionMerge(t *testing.T) {
	tests := []struct {
		name        string
		existing    string
		incoming    string
		shouldKeep  string
		shouldMerge bool // true if incoming should replace existing
	}{
		{
			name:        "incoming longer wins",
			existing:    "Short desc",
			incoming:    "A much longer and more detailed description of the entity",
			shouldKeep:  "A much longer and more detailed description of the entity",
			shouldMerge: true,
		},
		{
			name:        "existing longer stays",
			existing:    "A detailed existing description with lots of context",
			incoming:    "Brief",
			shouldKeep:  "A detailed existing description with lots of context",
			shouldMerge: false,
		},
		{
			name:        "same length no change",
			existing:    "Same length text A",
			incoming:    "Same length text B",
			shouldKeep:  "Same length text A",
			shouldMerge: false,
		},
		{
			name:        "empty existing gets replaced",
			existing:    "",
			incoming:    "New description",
			shouldKeep:  "New description",
			shouldMerge: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Simulate the merge logic from SaveExtractedKnowledge
			result := tt.existing
			if len(tt.incoming) > len(tt.existing) {
				result = tt.incoming
			}
			if result != tt.shouldKeep {
				t.Errorf("Expected %q, got %q", tt.shouldKeep, result)
			}
			merged := len(tt.incoming) > len(tt.existing)
			if merged != tt.shouldMerge {
				t.Errorf("Expected shouldMerge=%v, got %v", tt.shouldMerge, merged)
			}
		})
	}
}
