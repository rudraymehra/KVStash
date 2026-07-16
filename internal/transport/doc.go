// Package transport implements the batched multi-stream TCP transport: framed
// connection loops (64-byte KVB1 headers, PROTOCOL.md), buffer lending from
// the store, credit-based backpressure (§8), and coalesced writev responses.
//
// Topology: one reader + one writer goroutine per connection, joined by a
// bounded write queue. Handlers run on the reader goroutine. The writer
// coalesces backlogged responses into ≥16 MiB writev flushes but never delays
// a lone frame. Responses carry credit grants piggybacked in the header; an
// idle writer ships pending grants via unsolicited NOP/CREDIT frames.
//
// The transport answers exactly three things itself — the §9 protocol-fatal
// report, ERR_TOO_LARGE for over-cap frames (skipped via their authenticated
// payload_len), and ERR_BUSY for credit violations; NOP keepalives are
// swallowed. Everything else is dispatched to the server's FrameHandler.
package transport
