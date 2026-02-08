package dto

import "time"

// SearchQuery represents a search query request
type SearchQuery struct {
	Query    string   `json:"query" binding:"required"`
	GroupIDs []string `json:"group_ids,omitempty"`
	MaxFacts int      `json:"max_facts,omitempty"`
}

// SearchResults represents search results
type SearchResults struct {
	Facts []FactResult `json:"facts"`
	Total int          `json:"total"`
}

// GetMemoryRequest represents a request to get memory
type GetMemoryRequest struct {
	Messages []Message `json:"messages" binding:"required"`
	GroupIDs []string  `json:"group_ids,omitempty"`
	MaxFacts int       `json:"max_facts,omitempty"`
}

// GetMemoryResponse represents a memory response
type GetMemoryResponse struct {
	Facts []FactResult `json:"facts"`
	Total int          `json:"total"`
}

// GetEpisodesRequest represents a request to get episodes
type GetEpisodesRequest struct {
	GroupID string `json:"group_id" binding:"required"`
	LastN   int    `json:"last_n,omitempty"`
}

// Episode represents an episode in the knowledge graph
type Episode struct {
	CreatedAt time.Time `json:"created_at"`
	UUID      string    `json:"uuid"`
	GroupID   string    `json:"group_id"`
	Content   string    `json:"content"`
	Source    string    `json:"source,omitempty"`
}

// GetEpisodesResponse represents episodes response
type GetEpisodesResponse struct {
	Episodes []Episode `json:"episodes"`
	Total    int       `json:"total"`
}

// SearchFactsRequest represents a request to search the fact store
type SearchFactsRequest struct {
	Query         string   `json:"query" binding:"required"`
	GroupID       string   `json:"group_id,omitempty"`
	NodeTypes     []string `json:"node_types,omitempty"`
	SearchMethods []string `json:"search_methods,omitempty"` // "vector", "keyword", or both
	Limit         int      `json:"limit,omitempty"`
	MinScore      float64  `json:"min_score,omitempty"`
}

// SearchFactsResponse represents the response from a fact store search
// Uses ExtractedNodeDTO and ExtractedTripleDTO from ingest.go
type SearchFactsResponse struct {
	Query        string               `json:"query"`
	Nodes        []ExtractedNodeDTO   `json:"nodes"`
	Triples      []ExtractedTripleDTO `json:"triples,omitempty"`
	NodeScores   []float64            `json:"node_scores"`
	TripleScores []float64            `json:"triple_scores,omitempty"`
	Total        int                  `json:"total"`
	Success      bool                 `json:"success"`
}
