//go:build kvbdebug

package transport

import "strconv"

// assertConserved panics if the credit ledger is unbalanced at connection
// close (consumed != granted after every in-flight frame has resolved). Every
// Consume/Account MUST be matched by exactly one Grant on every read-loop path
// — skip, violation, and truncated-frame paths included — so an imbalance here
// is a transport bug that would stall a real client ~payload_len bytes at a
// time. Debug-build only, wired through conserveCheck at teardown.
func (w *CreditWindow) assertConserved() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.consumed != w.granted {
		panic("transport: credit ledger not conserved at close: consumed=" +
			strconv.FormatUint(w.consumed, 10) + " granted=" + strconv.FormatUint(w.granted, 10))
	}
}

// Poison overwrites b with 0xDE in debug builds (-tags kvbdebug), so any code
// that kept a view of a Returned buffer reads garbage immediately instead of
// silently reading recycled bytes. BufferSource implementations call it first
// thing in Return.
func Poison(b []byte) {
	for i := range b {
		b[i] = 0xDE
	}
}

// conserveCheck asserts the credit ledger balanced at connection close
// (debug builds only). A panic here means a Consume/Account without a matching
// Grant on some read-loop path — the bug class that stalls a real client.
func conserveCheck(w *CreditWindow) { w.assertConserved() }
