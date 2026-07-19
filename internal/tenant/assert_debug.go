//go:build kvbdebug

package tenant

import "fmt"

// quotaUnderflow: a negative tier balance means some exit path refunded
// bytes it never charged (or refunded twice). In the debug build that is a
// hard stop — accounting bugs must not hide.
func quotaUnderflow(ns uint32, tier Tier, now int64) {
	panic(fmt.Sprintf("tenant: quota underflow ns=%d tier=%s balance=%d", ns, tier, now))
}
