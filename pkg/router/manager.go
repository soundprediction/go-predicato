//go:build cgo

package router

import (
	"context"
	"fmt"
	"log/slog"
	"maps"
	"sync"
	"time"

	predicato "github.com/soundprediction/predicato"
	"github.com/soundprediction/predicato/pkg/crossencoder"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/soundprediction/predicato/pkg/embedder"
	"github.com/soundprediction/predicato/pkg/nlp"
	"github.com/soundprediction/predicato/pkg/types"
)

const defaultMaxOpenClients = 10

// Manager manages multiple predicato clients for different knowledge domains.
// It provides lazy initialization, thread-safe access, LRU eviction, and unified lifecycle management.
type Manager struct {
	sharedEmbedder embedder.Client
	clients        map[string]*predicato.Client
	drivers        map[string]driver.GraphDriver

	// sharedNLPClient is the NLP client shared across all predicato clients.
	// Can be nil for search-only use cases.
	sharedNLPClient nlp.Client

	// crossEncoder is set on every new client for reranking support.
	crossEncoder crossencoder.Client

	// LRU tracking: ordered list of client names, most recently used at the end.
	lruOrder       []string
	maxOpenClients int

	logger             *slog.Logger
	defaultClientName  string
	fallbackClientName string
	configs            []ClientConfig
	mu                 sync.Mutex
}

// NewManager creates a new Manager with the given configuration.
// nlpClient can be nil if you only need search (no ingestion).
func NewManager(
	configs []ClientConfig,
	nlpClient nlp.Client,
	sharedEmbedder embedder.Client,
	logger *slog.Logger,
) *Manager {
	return NewManagerWithOptions(configs, nlpClient, sharedEmbedder, logger, defaultMaxOpenClients)
}

// NewManagerWithOptions creates a new Manager with the given configuration and max open clients limit.
// maxOpenClients controls how many graph databases can be open simultaneously.
// Set to 0 for unlimited (not recommended for ladybug backends with many DBs).
func NewManagerWithOptions(
	configs []ClientConfig,
	nlpClient nlp.Client,
	sharedEmbedder embedder.Client,
	logger *slog.Logger,
	maxOpenClients int,
) *Manager {
	if logger == nil {
		logger = slog.Default()
	}

	pm := &Manager{
		configs:         configs,
		clients:         make(map[string]*predicato.Client),
		drivers:         make(map[string]driver.GraphDriver),
		sharedNLPClient: nlpClient,
		sharedEmbedder:  sharedEmbedder,
		logger:          logger,
		maxOpenClients:  maxOpenClients,
	}

	for _, cfg := range configs {
		if cfg.Default {
			pm.defaultClientName = cfg.Name
		}
		if cfg.Fallback {
			pm.fallbackClientName = cfg.Name
		}
	}

	if pm.defaultClientName == "" && len(configs) > 0 {
		pm.defaultClientName = configs[0].Name
	}

	logger.Info("Manager initialized",
		"num_clients", len(configs),
		"default_client", pm.defaultClientName,
		"fallback_client", pm.fallbackClientName,
		"max_open_clients", maxOpenClients,
	)

	return pm
}

// GetClient returns the predicato client with the given name.
// If the client hasn't been initialized yet, it will be lazily initialized.
// If the max open clients limit is reached, the least recently used client is evicted.
func (pm *Manager) GetClient(name string) (*predicato.Client, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if client, exists := pm.clients[name]; exists {
		pm.touchLRU(name)
		return client, nil
	}

	var clientConfig *ClientConfig
	for i := range pm.configs {
		if pm.configs[i].Name == name {
			clientConfig = &pm.configs[i]
			break
		}
	}

	if clientConfig == nil {
		return nil, fmt.Errorf("client %q not found in configuration", name)
	}

	// Evict LRU client if at capacity
	if pm.maxOpenClients > 0 && len(pm.clients) >= pm.maxOpenClients {
		pm.evictLRU()
	}

	client, err := pm.initializeClient(clientConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize client %q: %w", name, err)
	}

	pm.clients[name] = client
	pm.lruOrder = append(pm.lruOrder, name)
	pm.logger.Info("Predicato client initialized", "name", name, "group_id", clientConfig.GroupID,
		"open_clients", len(pm.clients))

	return client, nil
}

// touchLRU moves a name to the end of the LRU list (most recently used).
func (pm *Manager) touchLRU(name string) {
	for i, n := range pm.lruOrder {
		if n == name {
			pm.lruOrder = append(pm.lruOrder[:i], pm.lruOrder[i+1:]...)
			pm.lruOrder = append(pm.lruOrder, name)
			return
		}
	}
	pm.lruOrder = append(pm.lruOrder, name)
}

// evictLRU closes and removes the least recently used client.
func (pm *Manager) evictLRU() {
	if len(pm.lruOrder) == 0 {
		return
	}

	evictName := pm.lruOrder[0]
	pm.lruOrder = pm.lruOrder[1:]

	pm.logger.Info("Evicting LRU client", "name", evictName, "open_clients", len(pm.clients))

	ctx := context.Background()
	if client, ok := pm.clients[evictName]; ok {
		if err := client.Close(ctx); err != nil {
			pm.logger.Warn("Error closing evicted client", "name", evictName, "error", err)
		}
		delete(pm.clients, evictName)
	}
	if d, ok := pm.drivers[evictName]; ok {
		if err := d.Close(); err != nil {
			pm.logger.Warn("Error closing evicted driver", "name", evictName, "error", err)
		}
		delete(pm.drivers, evictName)
	}
}

// initializeClient creates a new predicato client from the given config.
func (pm *Manager) initializeClient(cfg *ClientConfig) (*predicato.Client, error) {
	ctx := context.Background()

	graphDriver, err := cfg.CreateDriver(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create graph driver: %w", err)
	}

	pm.drivers[cfg.Name] = graphDriver

	if !cfg.ReadOnly {
		if err := graphDriver.CreateIndices(ctx); err != nil {
			_ = graphDriver.Close()
			delete(pm.drivers, cfg.Name)
			return nil, fmt.Errorf("failed to create indices: %w", err)
		}
	}

	groupID := cfg.GroupID
	if groupID == "" {
		groupID = "default"
	}

	predicatoConfig := &predicato.Config{
		GroupID:  groupID,
		TimeZone: time.UTC,
		SearchConfig: &types.SearchConfig{
			Limit:              10,
			CenterNodeDistance: 2,
			MinScore:           0.0,
			IncludeEdges:       true,
			Rerank:             false,
			NodeConfig:         &types.NodeSearchConfig{},
			EdgeConfig:         &types.EdgeSearchConfig{},
		},
	}

	client, err := predicato.NewClient(graphDriver, pm.sharedNLPClient, pm.sharedEmbedder, predicatoConfig, pm.logger)
	if err != nil {
		_ = graphDriver.Close()
		delete(pm.drivers, cfg.Name)
		return nil, fmt.Errorf("failed to create predicato client: %w", err)
	}

	if pm.crossEncoder != nil {
		client.SetCrossEncoder(pm.crossEncoder)
	}

	return client, nil
}

// SetCrossEncoder sets the cross-encoder reranker to be applied to all clients.
func (pm *Manager) SetCrossEncoder(ce crossencoder.Client) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.crossEncoder = ce
	// Apply to already-initialized clients
	for _, client := range pm.clients {
		client.SetCrossEncoder(ce)
	}
}

// GetDefaultClient returns the default predicato client.
func (pm *Manager) GetDefaultClient() (*predicato.Client, error) {
	if pm.defaultClientName == "" {
		return nil, fmt.Errorf("no default client configured")
	}
	return pm.GetClient(pm.defaultClientName)
}

// GetAllClients returns all initialized clients.
func (pm *Manager) GetAllClients() map[string]*predicato.Client {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	result := make(map[string]*predicato.Client, len(pm.clients))
	maps.Copy(result, pm.clients)
	return result
}

// GetClientConfigs returns all client configurations.
func (pm *Manager) GetClientConfigs() []ClientConfig {
	return pm.configs
}

// InitializeAllClients eagerly initializes all configured clients.
// Errors are logged but do not stop initialization of remaining clients.
// Returns the number of successfully initialized clients.
func (pm *Manager) InitializeAllClients() (int, int) {
	ok, fail := 0, 0
	for _, cfg := range pm.configs {
		if _, err := pm.GetClient(cfg.Name); err != nil {
			pm.logger.Warn("Failed to initialize client", "name", cfg.Name, "error", err)
			fail++
		} else {
			ok++
		}
	}
	return ok, fail
}

// Close closes all initialized clients and their drivers.
func (pm *Manager) Close(ctx context.Context) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	var firstErr error

	for name, client := range pm.clients {
		pm.logger.Info("Closing predicato client", "name", name)
		if err := client.Close(ctx); err != nil {
			pm.logger.Error("Error closing predicato client", "name", name, "error", err)
			if firstErr == nil {
				firstErr = err
			}
		}
	}

	for name, d := range pm.drivers {
		pm.logger.Info("Closing graph driver", "name", name)
		if err := d.Close(); err != nil {
			pm.logger.Error("Error closing graph driver", "name", name, "error", err)
			if firstErr == nil {
				firstErr = err
			}
		}
	}

	pm.clients = make(map[string]*predicato.Client)
	pm.drivers = make(map[string]driver.GraphDriver)
	pm.lruOrder = nil

	pm.logger.Info("Manager closed")
	return firstErr
}

// ClientStatus represents the status of a predicato client.
type ClientStatus struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	GroupID     string   `json:"group_id"`
	GraphDbType string   `json:"graph_db_type"`
	Topics      []string `json:"topics"`
	Initialized bool     `json:"initialized"`
	IsDefault   bool     `json:"is_default"`
	IsFallback  bool     `json:"is_fallback"`
}

// GetClientStatuses returns the status of all configured clients.
func (pm *Manager) GetClientStatuses() []ClientStatus {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	statuses := make([]ClientStatus, 0, len(pm.configs))
	for _, cfg := range pm.configs {
		_, initialized := pm.clients[cfg.Name]

		graphDbType := "unknown"
		if dbType, ok := cfg.GraphDb["type"].(string); ok {
			graphDbType = dbType
		}

		statuses = append(statuses, ClientStatus{
			Name:        cfg.Name,
			Description: cfg.Description,
			GroupID:     cfg.GroupID,
			Initialized: initialized,
			IsDefault:   cfg.Name == pm.defaultClientName,
			IsFallback:  cfg.Name == pm.fallbackClientName,
			Topics:      cfg.Topics,
			GraphDbType: graphDbType,
		})
	}

	return statuses
}
