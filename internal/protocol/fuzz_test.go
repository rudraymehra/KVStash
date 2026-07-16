package protocol

import (
	"errors"
	"math"
	"testing"
)

// FuzzParseHeader drives hostile bytes through the full parse/marshal cycle.
// Invariants (any violation is a wire-safety bug):
//  1. never panic, whatever the bytes;
//  2. on success, re-marshalling the parsed header reproduces the input's
//     first 64 bytes exactly (decode∘encode identity);
//  3. ErrPayloadTooLarge is purely the cap's fault: the same bytes parse
//     cleanly under the maximum cap, to an identical header;
//  4. every fatal error is classified fatal, every recoverable one is not.
func FuzzParseHeader(f *testing.F) {
	_, valid := sampleHeader()
	f.Add(valid, uint32(DefaultMaxFrameLen))
	f.Add(valid, uint32(16))                                // over-cap on valid bytes
	f.Add(valid[:HeaderSize-1], uint32(DefaultMaxFrameLen)) // truncated
	f.Add([]byte{}, uint32(0))
	flipped := append([]byte(nil), valid...)
	flipped[keyOffset] ^= 0x01
	f.Add(flipped, uint32(DefaultMaxFrameLen)) // CRC-broken
	zeros := make([]byte, HeaderSize)
	f.Add(zeros, uint32(DefaultMaxFrameLen)) // bad magic

	f.Fuzz(func(t *testing.T, data []byte, maxPayload uint32) {
		h, err := ParseHeader(data, maxPayload) // invariant 1: must not panic

		switch {
		case err == nil:
			out := make([]byte, HeaderSize)
			h.MarshalTo(out)
			if string(out) != string(data[:HeaderSize]) {
				t.Fatalf("decode∘encode not identity:\n in  %x\n out %x", data[:HeaderSize], out)
			}
		case errors.Is(err, ErrPayloadTooLarge):
			if errors.Is(err, ErrFatalFrame) {
				t.Fatal("ErrPayloadTooLarge classified fatal")
			}
			h2, err2 := ParseHeader(data, math.MaxUint32)
			if err2 != nil {
				t.Fatalf("over-cap bytes failed under max cap: %v", err2)
			}
			if h != h2 {
				t.Fatalf("over-cap header differs from max-cap parse:\n %+v\n %+v", h, h2)
			}
		default:
			if !errors.Is(err, ErrFatalFrame) {
				t.Fatalf("unclassified error %v: neither recoverable nor fatal", err)
			}
		}
	})
}
