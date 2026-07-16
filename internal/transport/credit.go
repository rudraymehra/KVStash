package transport

import (
	"math"
	"sync"
	"time"
)

// Violation is CreditWindow.Consume's verdict on one request frame.
type Violation int

const (
	// ViolationNone: the frame is within the window; process it.
	ViolationNone Violation = iota
	// ViolationBusy: first breach — skip the frame, answer ERR_BUSY, re-grant
	// the bytes (PROTOCOL.md §8 rule 5, first strike).
	ViolationBusy
	// ViolationFatal: repeat breach — F_FATAL close (§8 rule 5, escalation).
	ViolationFatal
)

// CreditWindow is the server-side ledger for the §8 byte-granular
// backpressure window on client→server payload bytes.
//
// The design center is CONSERVATION (amended §8 rule 4): every Consume(n) is
// matched by exactly one Grant(n) — from the handler once it has drained the
// bytes, or from the transport for frames it skipped (oversize, credit
// violation, NOP payloads, tombstone discards). One missed Grant path stalls
// a client permanently, ~payload_len bytes at a time; the Totals accessor
// exists so tests (and the kvbdebug close-time assert) can prove the books
// balance.
type CreditWindow struct {
	mu          sync.Mutex
	window      uint64 // negotiated initial_credit (pre-HELLO: the pre-negotiation cap)
	maxFrame    uint64 // negotiated max_frame_len: the sanctioned overshoot (§8 rule 3)
	outstanding uint64 // debited-not-yet-granted bytes
	pending     uint64 // granted-not-yet-sent bytes; the writer harvests these
	strikes     int    // ERR_BUSY → F_FATAL escalation state

	consumed uint64 // lifetime totals, for the conservation invariant
	granted  uint64
}

// SetWindow installs the (re)negotiated window and frame cap. Called once at
// connection start with the pre-negotiation values and once post-HELLO with
// the negotiated ones.
func (w *CreditWindow) SetWindow(window, maxFrame uint32) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.window = uint64(window)
	w.maxFrame = uint64(maxFrame)
}

// Consume debits one request frame's payload bytes and rules on the window.
// The enforcement threshold is window + maxFrame: §8 rule 3 sanctions sending
// one final frame from a small-positive window, so overshoot by at most one
// frame IS the legal limit, and crossing it is rule 5's violation.
func (w *CreditWindow) Consume(n uint32) Violation {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.outstanding += uint64(n)
	w.consumed += uint64(n)
	if w.outstanding <= w.window+w.maxFrame {
		return ViolationNone
	}
	w.strikes++
	if w.strikes == 1 {
		return ViolationBusy
	}
	return ViolationFatal
}

// Grant returns n consumed bytes to the client's window: they become pending
// wire grants the writer piggybacks on the next response header (or ships in
// an unsolicited NOP/CREDIT frame when the writer is otherwise idle).
func (w *CreditWindow) Grant(n uint32) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if uint64(n) > w.outstanding {
		// More granted than consumed = a transport bug; clamp so the ledger
		// never underflows, and let the conservation check flag the imbalance.
		n = uint32(w.outstanding) //nolint:gosec // G115: outstanding < n <= MaxUint32 in this branch
	}
	w.outstanding -= uint64(n)
	w.granted += uint64(n)
	w.pending += uint64(n)
}

// TakeGrant harvests up to MaxUint32 pending grant bytes for stamping into a
// response header's credit field; the remainder stays pending.
func (w *CreditWindow) TakeGrant() uint32 {
	w.mu.Lock()
	defer w.mu.Unlock()
	g := w.pending
	if g > math.MaxUint32 {
		g = math.MaxUint32
	}
	w.pending -= g
	return uint32(g) //nolint:gosec // G115: g clamped to MaxUint32 above
}

// PendingGrant reports whether grant bytes are waiting for a ride (drives the
// writer's unsolicited NOP/CREDIT timer).
func (w *CreditWindow) PendingGrant() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.pending > 0
}

// Totals returns the lifetime (consumed, granted) byte counts. Conservation:
// at connection close with no frames in flight, consumed == granted.
func (w *CreditWindow) Totals() (consumed, granted uint64) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.consumed, w.granted
}

// StallTimeout is the §8 rule-5 zero-drain connection deadline: 2× the
// PUT_STREAM inactivity timeout. One source of truth for the constant; the
// write loop's per-flush deadline is the enforcement mechanism.
func StallTimeout(streamTimeoutMS uint32) time.Duration {
	return 2 * time.Duration(streamTimeoutMS) * time.Millisecond
}
