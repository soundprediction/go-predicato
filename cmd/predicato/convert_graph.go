package predicato

import (
	"context"
	"fmt"

	"github.com/soundprediction/predicato/pkg/conversion"
	"github.com/soundprediction/predicato/pkg/driver"
	"github.com/spf13/cobra"
)

var convertCmd = &cobra.Command{
	Use:   "convert-graph",
	Short: "Convert graph data between drivers",
	Long: `Migrate graph data from one driver to another.

Copies all nodes and edges from the source driver to the destination driver,
preserving all properties, embeddings, and relationships. The operation is
idempotent (safe to retry).

Examples:
  predicato convert-graph --source-driver ladybug --source-uri ./ladybug_db --dest-driver cozo --dest-uri ./cozo_db
  predicato convert-graph --source-driver cozo --source-uri ./cozo_db --dest-driver duckpgq --dest-uri ./duck_graph.db`,
	RunE: runConvertGraph,
}

func init() {
	rootCmd.AddCommand(convertCmd)

	convertCmd.Flags().String("source-driver", "", "source graph driver (ladybug, cozo, duckpgq, neo4j, memgraph)")
	convertCmd.Flags().String("source-uri", "", "source database URI or path")
	convertCmd.Flags().String("dest-driver", "", "destination graph driver (ladybug, cozo, duckpgq, neo4j, memgraph)")
	convertCmd.Flags().String("dest-uri", "", "destination database URI or path")
	convertCmd.Flags().Int("embedding-dim", 1024, "embedding dimensions for new drivers")

	_ = convertCmd.MarkFlagRequired("source-driver")
	_ = convertCmd.MarkFlagRequired("source-uri")
	_ = convertCmd.MarkFlagRequired("dest-driver")
	_ = convertCmd.MarkFlagRequired("dest-uri")
}

func runConvertGraph(cmd *cobra.Command, args []string) error {
	srcDriverName, _ := cmd.Flags().GetString("source-driver")
	srcURI, _ := cmd.Flags().GetString("source-uri")
	destDriverName, _ := cmd.Flags().GetString("dest-driver")
	destURI, _ := cmd.Flags().GetString("dest-uri")
	embDim, _ := cmd.Flags().GetInt("embedding-dim")

	fmt.Printf("Converting graph: %s(%s) -> %s(%s)\n", srcDriverName, srcURI, destDriverName, destURI)

	srcDriver, err := newGraphDriver(srcDriverName, srcURI, embDim)
	if err != nil {
		return fmt.Errorf("failed to create source driver: %w", err)
	}
	defer srcDriver.Close()

	destDriver, err := newGraphDriver(destDriverName, destURI, embDim)
	if err != nil {
		return fmt.Errorf("failed to create destination driver: %w", err)
	}
	defer destDriver.Close()

	// Create indices on destination
	ctx := context.Background()
	if err := destDriver.CreateIndices(ctx); err != nil {
		fmt.Printf("Warning: failed to create indices on destination: %v\n", err)
	}

	return conversion.CopyGraph(ctx, srcDriver, destDriver)
}

func newGraphDriver(driverName, uri string, embDim int) (driver.GraphDriver, error) {
	switch driverName {
	case "ladybug":
		return driver.NewLadybugDriver(uri, 16)
	case "cozo":
		return driver.NewCozoDriver(uri, embDim)
	case "duckpgq":
		return driver.NewDuckPGQDriver(uri, embDim)
	default:
		return nil, fmt.Errorf("unsupported driver: %s (supported: ladybug, cozo, duckpgq)", driverName)
	}
}
