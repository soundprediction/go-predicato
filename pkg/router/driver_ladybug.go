//go:build cgo && system_ladybug

package router

import (
	"github.com/soundprediction/predicato/pkg/driver"
)

func init() {
	RegisterDriverFactory(GraphDbTypeLadybug, func(c *ClientConfig, dbPath string, embDim int) (driver.GraphDriver, error) {
		cfg := driver.DefaultLadybugDriverConfig()
		cfg.DBPath = dbPath
		cfg.ReadOnly = c.ReadOnly
		if c.ReadOnly {
			cfg.BufferPoolSize = bufferPoolForDBSize(dbPath)
		}
		return driver.NewLadybugDriverWithConfig(cfg)
	})
}
