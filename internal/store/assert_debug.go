//go:build kvbdebug

package store

import "fmt"

// assertf panics on violated invariants in debug builds (-tags kvbdebug);
// the release build compiles it away to a no-op — the same convention as
// dram's assert_debug.go and transport's poison_debug.go.
func assertf(cond bool, format string, args ...any) {
	if !cond {
		panic(fmt.Sprintf(format, args...))
	}
}
