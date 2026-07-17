//go:build !kvbdebug

package dram

// assertf is a no-op in release builds; see assert_debug.go.
func assertf(bool, string, ...any) {}

// debugAssertsEnabled reports whether kvbdebug asserts are compiled in.
func debugAssertsEnabled() bool { return false }
