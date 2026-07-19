//go:build !kvbdebug

package tenant

// quotaUnderflow is a no-op in release builds — Refund heals the balance
// back to zero and serving continues (an accounting bug, not corruption).
func quotaUnderflow(uint32, Tier, int64) {}
