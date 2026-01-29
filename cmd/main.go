package main

//go:generate sh -c "curl -sL https://raw.githubusercontent.com/LadybugDB/go-ladybug/refs/heads/master/download_lbug.sh | bash -s -- -out lib-ladybug"

/*
#cgo darwin LDFLAGS: -L${SRCDIR}/lib-ladybug -llbug -Wl,-rpath,${SRCDIR}/lib-ladybug
#cgo linux LDFLAGS: -L${SRCDIR}/lib-ladybug -llbug -Wl,-rpath,${SRCDIR}/lib-ladybug
#cgo windows LDFLAGS: -L${SRCDIR}/lib-ladybug -llbug_shared
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
