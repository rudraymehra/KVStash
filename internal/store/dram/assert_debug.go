//go:build kvbdebug

package dram

import "fmt"

// assertf panics on violated invariants in debug builds (-tags kvbdebug); the
// release build compiles it away to a no-op (offsetalloc "no panic paths"
// contract, mirrored on transport's poison_debug.go convention).
func assertf(cond bool, format string, args ...any) {
	if !cond {
		panic(fmt.Sprintf(format, args...))
	}
}
