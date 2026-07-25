package transport

// BufferSource lends payload buffers to the read loop; the frame handler's
// owner implements it (the server lends a recycled per-connection extent) so
// steady-state PUT traffic reuses one buffer instead of allocating per frame.
//
// Ownership protocol (load-bearing): the read loop Lends a buffer and hands
// it to the FrameHandler with the frame → the handler owns it → whoever
// finishes with the bytes Returns it, either directly or via the release
// callback of Conn.WriteFrames after the kernel has accepted the response.
// The Conn never auto-Returns a body it dispatched. After Return, the caller
// MUST NOT retain any view of the buffer — debug builds (-tags kvbdebug)
// poison it with 0xDE so a stale alias fails loudly, not subtly.
type BufferSource interface {
	// Lend returns a buffer with len == n (cap may exceed it). Ownership
	// transfers to the caller until Return.
	Lend(n int) []byte
	// Return releases b back to the source. Implementations call Poison(b)
	// first thing, so the debug-build contract holds for every source.
	Return(b []byte)
}

// HeapSource is the trivial BufferSource: plain heap allocations. It exists
// for tests and standalone transport use; the server's session lends a
// recycled buffer instead (see internal/server), so per-frame allocation
// never sits on the hot PUT path.
type HeapSource struct{}

// Lend allocates a fresh buffer.
func (HeapSource) Lend(n int) []byte { return make([]byte, n) }

// Return poisons in debug builds and lets the GC collect otherwise.
func (HeapSource) Return(b []byte) { Poison(b) }
