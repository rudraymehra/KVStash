// Package protocol implements the kvblockd wire protocol codec ("KVB1").
//
// The normative specification is docs/PROTOCOL.md. This package owns the
// 64-byte frame header, the opcode/flag/status tables, and (as they land) the
// body codecs. It is pure — no I/O, no goroutines, no allocation on the frame
// hot path — so every entry point is fuzzable in isolation. (One compiler
// caveat: buffers passed in escape via the stdlib CRC assembly routine, so
// callers must use heap/pooled buffers, never stack arrays — see Header.)
//
// The load-bearing safety property: a receiver validates everything it is
// about to trust *before* trusting it — magic, then version, then the header
// CRC32C (which protects payload_len), then the payload_len cap — and only
// then may a caller allocate or read payload bytes. An error can never
// desynchronize the stream, because payload_len always tells the receiver how
// many bytes to skip.
package protocol
