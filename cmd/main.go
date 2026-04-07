package main

//go:generate bash update-vendor-libs.sh

/*
#cgo LDFLAGS: -L${SRCDIR}/lib-cozo
#include <stdlib.h>

// fast_exit bypasses normal cleanup (atexit handlers, C++ destructors)
// to prevent segfaults from the Ladybug library's cleanup routines
static void fast_exit(int status) {
    _Exit(status);
}
*/
import "C"

import (
	"github.com/soundprediction/predicato/cmd/predicato"
)

func main() {
	if err := predicato.Execute(); err != nil {
		// Use fast_exit(1) to avoid segfault from Ladybug library cleanup
		C.fast_exit(1)
	}
	// Use fast_exit(0) to bypass Go runtime and C++ destructor cleanup
	// which can cause segfaults with the Ladybug library
	C.fast_exit(0)
}
