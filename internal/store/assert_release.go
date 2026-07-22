//go:build !kvbdebug

package store

// assertf is a no-op in release builds; see assert_debug.go.
func assertf(bool, string, ...any) {}
