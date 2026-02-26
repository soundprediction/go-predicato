//go:build cgo && system_ladybug

package main

/*
#cgo darwin LDFLAGS: -L${SRCDIR}/lib-ladybug -llbug -Wl,-rpath,${SRCDIR}/lib-ladybug
#cgo linux LDFLAGS: -L${SRCDIR}/lib-ladybug -llbug -Wl,-rpath,${SRCDIR}/lib-ladybug
#cgo windows LDFLAGS: -L${SRCDIR}/lib-ladybug -llbug_shared
*/
import "C"
