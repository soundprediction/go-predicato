//go:build !cgo

package predicato

import (
	"fmt"

	core "github.com/soundprediction/predicato"
	"github.com/soundprediction/predicato/pkg/config"
	"github.com/spf13/cobra"
)

func initializeRouterPredicato(cmd *cobra.Command, cfg *config.Config) (core.Predicato, error) {
	return nil, fmt.Errorf("pooled router mode requires cgo")
}
