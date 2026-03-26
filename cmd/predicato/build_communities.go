package predicato

import (
	"context"
	"fmt"
	"time"

	"github.com/spf13/cobra"
)

var buildCommunitiesCmd = &cobra.Command{
	Use:   "build-communities",
	Short: "Run community detection on an existing graph database",
	Long: `Run label propagation community detection on an existing graph database.

Examples:
  predicato build-communities --driver duckpgq --uri ./graph.duckdb --group-id wikidata
  predicato build-communities --driver duckpgq --uri ./graph.duckdb --group-id statpearls`,
	RunE: runBuildCommunities,
}

func init() {
	rootCmd.AddCommand(buildCommunitiesCmd)

	buildCommunitiesCmd.Flags().String("driver", "duckpgq", "graph driver (duckpgq, ladybug, cozo)")
	buildCommunitiesCmd.Flags().String("uri", "", "database URI or path")
	buildCommunitiesCmd.Flags().StringSlice("group-id", nil, "group IDs to build communities for (default: all)")
	buildCommunitiesCmd.Flags().Int("embedding-dim", 1024, "embedding dimensions")

	_ = buildCommunitiesCmd.MarkFlagRequired("uri")
}

func runBuildCommunities(cmd *cobra.Command, args []string) error {
	driverName, _ := cmd.Flags().GetString("driver")
	uri, _ := cmd.Flags().GetString("uri")
	groupIDs, _ := cmd.Flags().GetStringSlice("group-id")
	embDim, _ := cmd.Flags().GetInt("embedding-dim")

	drv, err := newGraphDriver(driverName, uri, embDim)
	if err != nil {
		return fmt.Errorf("failed to create driver: %w", err)
	}
	defer drv.Close()

	ctx := context.Background()

	// If no group IDs specified, discover them
	if len(groupIDs) == 0 {
		groupIDs, err = drv.GetAllGroupIDs(ctx)
		if err != nil {
			return fmt.Errorf("failed to get group IDs: %w", err)
		}
	}

	for _, gid := range groupIDs {
		fmt.Printf("Building communities for group %q...\n", gid)
		start := time.Now()
		if err := drv.BuildCommunities(ctx, gid); err != nil {
			fmt.Printf("  Warning: community detection failed for %q: %v\n", gid, err)
			continue
		}
		fmt.Printf("  Done in %s\n", time.Since(start).Round(time.Second))
	}

	return nil
}
