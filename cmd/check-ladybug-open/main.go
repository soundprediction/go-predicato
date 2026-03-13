//go:build cgo

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/soundprediction/predicato/pkg/driver"
)

func main() {
	dir := "/data/topic-graphs/ladybug"
	entries, _ := os.ReadDir(dir)

	ok, fail := 0, 0
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".ladybug") {
			continue
		}
		name := strings.TrimSuffix(e.Name(), ".ladybug")
		path := filepath.Join(dir, e.Name())

		d, err := driver.NewLadybugDriver(path, 1)
		if err != nil {
			fmt.Printf("FAIL  %-35s  %v\n", name, err)
			fail++
			continue
		}
		d.Close()
		fmt.Printf("OK    %-35s\n", name)
		ok++
	}
	fmt.Printf("\n%d OK, %d FAIL\n", ok, fail)
}
