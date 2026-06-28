//go:build !cgo

package predicato

import (
	"fmt"

	"github.com/soundprediction/predicato/pkg/config"
	"github.com/spf13/cobra"
)

// runServeGRPCRouterMode is unavailable without CGO, since the multi-graph
// router engine (pkg/router) and graph drivers require CGO.
func runServeGRPCRouterMode(_ *cobra.Command, _ *config.Config, _, _ string) error {
	return fmt.Errorf("router mode (--source-dir) requires a CGO build of predicato")
}
