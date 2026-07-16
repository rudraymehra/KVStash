//go:build !kvbdebug

package transport

// Poison is a no-op in release builds; see poison_debug.go for the contract.
// A memset per 0.4–2.5 MB buffer is measurable on the hot path, so the check
// exists only under -tags kvbdebug (CI runs a leg with it).
func Poison([]byte) {}

// conserveCheck is a no-op in release builds; see poison_debug.go.
func conserveCheck(*CreditWindow) {}
