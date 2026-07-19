//go:build kvbdebug

package tenant

import "testing"

// Debug build: the same unmatched Refund is a hard stop — accounting bugs
// must not hide behind the release heal.
func TestRefundUnderflowPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("kvbdebug refund underflow did not panic")
		}
	}()
	q := quotasWith(t, 100)
	q.Refund(1, TierDRAM, 50)
}
