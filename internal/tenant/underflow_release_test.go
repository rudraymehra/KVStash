//go:build !kvbdebug

package tenant

import "testing"

// Release build: an unmatched Refund (accounting bug) heals to zero and
// serving continues — the balance may neither stay negative (silently
// widening the quota) nor poison later charges.
func TestRefundClampsAtZero(t *testing.T) {
	q := quotasWith(t, 100)
	q.Refund(1, TierDRAM, 50)
	if got := q.Usage(1, TierDRAM); got != 0 {
		t.Fatalf("negative balance survived: %d", got)
	}
	if err := q.Charge(1, TierDRAM, 100); err != nil {
		t.Fatalf("quota widened or narrowed by the underflow heal: %v", err)
	}
}
