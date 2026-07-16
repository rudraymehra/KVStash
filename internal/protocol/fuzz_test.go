package protocol

import (
	"errors"
	"math"
	"testing"
)

// FuzzParseBatch drives hostile bytes through every body codec. The first
// input byte selects the codec; the rest is the body. Invariants:
//  1. never panic, whatever the bytes;
//  2. every error classifies as ErrBadBody and maps to a §9 status other
//     than ERR_INTERNAL (internal = a codec bug, not wire input);
//  3. decode∘encode∘decode is a fixpoint on accepted inputs (byte identity is
//     deliberately NOT asserted: §0 pad contents are ignored on receive, so
//     re-encoding canonicalizes nonzero pads to 0x00).
func FuzzParseBatch(f *testing.F) {
	keys := testKeys(3)
	f.Add([]byte{0}, uint32(DefaultMaxBatchKeys))
	f.Add(append([]byte{0}, AppendKeyList(nil, 5000, keys)...), uint32(DefaultMaxBatchKeys))
	f.Add(append([]byte{0}, AppendKeyList(nil, 0, keys)...), uint32(1)) // over-cap
	f.Add(append([]byte{1}, AppendExistsResp(nil, 3, 2, []Status{0, 0, 0x10})...), uint32(512))
	f.Add(append([]byte{2}, AppendExistsResp(nil, 3, 2, nil)...), uint32(512))
	f.Add(append([]byte{3}, AppendKeyStatusResp(nil, []Status{0})...), uint32(512))
	f.Add(append([]byte{4}, AppendGetRespHeader(nil, StatusOK, 0, 1, []Desc{{Len: 42}})...), uint32(512))
	f.Add(append([]byte{5}, AppendHelloReq(nil, HelloReq{ProtoMin: 1, ProtoMax: 1, Token: []byte("t"), Namespace: "ns"})...), uint32(512))
	f.Add(append([]byte{6}, AppendHelloResp(nil, HelloResp{Proto: 1, ServerName: "s"})...), uint32(512))
	f.Add(append([]byte{7}, AppendPutBegin(nil, PutBeginBody{TotalLen: 1})...), uint32(512))
	f.Add(append([]byte{8}, AppendPutCommit(nil, 9)...), uint32(512))
	f.Add(append([]byte{9}, AppendStatsReq(nil, 0)...), uint32(512))

	f.Fuzz(func(t *testing.T, data []byte, maxKeys uint32) {
		if len(data) == 0 {
			return
		}
		body := data[1:]
		var err error
		switch data[0] % 10 {
		case 0:
			var aux uint32
			var keys [][32]byte
			aux, keys, err = DecodeKeyList(body, maxKeys, nil)
			if err == nil {
				re := AppendKeyList(nil, aux, keys)
				aux2, keys2, err2 := DecodeKeyList(re, maxKeys, nil)
				if err2 != nil || aux2 != aux || len(keys2) != len(keys) {
					t.Fatalf("keylist fixpoint broken: %v", err2)
				}
			}
		case 1, 2:
			expectBitmap := data[0]%10 == 1
			var r ExistsResp
			r, err = DecodeExistsResp(body, expectBitmap)
			if err == nil && r.Status == StatusOK {
				perKey := make([]Status, len(r.PerKey))
				for i, b := range r.PerKey {
					perKey[i] = Status(b)
				}
				if !expectBitmap {
					perKey = nil
				}
				re := AppendExistsResp(nil, r.Count, r.NConsecutive, perKey)
				r2, err2 := DecodeExistsResp(re, expectBitmap)
				if err2 != nil || r2.NConsecutive != r.NConsecutive || r2.Count != r.Count {
					t.Fatalf("exists fixpoint broken: %v", err2)
				}
			}
		case 3:
			var p Preamble
			var st []byte
			p, st, err = DecodeKeyStatusResp(body)
			if err == nil && p.Status == StatusOK {
				perKey := make([]Status, len(st))
				for i, b := range st {
					perKey[i] = Status(b)
				}
				if _, _, err2 := DecodeKeyStatusResp(AppendKeyStatusResp(nil, perKey)); err2 != nil {
					t.Fatalf("keystatus fixpoint broken: %v", err2)
				}
			}
		case 4:
			var g GetRespHeader
			g, err = DecodeGetRespHeader(body)
			if err == nil && g.Status == StatusOK {
				descs := make([]Desc, g.Count)
				for i := range descs {
					descs[i] = g.Desc(i)
				}
				re := AppendGetRespHeader(nil, g.Status, g.FirstIndex, g.TotalKeys, descs)
				g2, err2 := DecodeGetRespHeader(re)
				if err2 != nil || g2.Count != g.Count {
					t.Fatalf("getresp fixpoint broken: %v", err2)
				}
			}
		case 5:
			var r HelloReq
			r, err = DecodeHelloReq(body)
			if err == nil {
				if _, err2 := DecodeHelloReq(AppendHelloReq(nil, r)); err2 != nil {
					t.Fatalf("helloreq fixpoint broken: %v", err2)
				}
			}
		case 6:
			var p Preamble
			var r HelloResp
			p, r, err = DecodeHelloResp(body)
			if err == nil && p.Status == StatusOK {
				if _, _, err2 := DecodeHelloResp(AppendHelloResp(nil, r)); err2 != nil {
					t.Fatalf("helloresp fixpoint broken: %v", err2)
				}
			}
		case 7:
			var b PutBeginBody
			b, err = DecodePutBegin(body)
			if err == nil {
				// Fixpoint, not byte identity: the ignored flags AND reserved
				// fields are canonicalized to their spec'd values on re-encode.
				b2, err2 := DecodePutBegin(AppendPutBegin(nil, b))
				if err2 != nil || b2.TotalLen != b.TotalLen || b2.XXH3Hint != b.XXH3Hint || b2.TTLms != b.TTLms {
					t.Fatalf("putbegin fixpoint broken: %+v vs %+v (%v)", b2, b, err2)
				}
			}
		case 8:
			_, err = DecodePutCommit(body)
		case 9:
			_, err = DecodeStatsReq(body)
		}
		if err != nil {
			if !errors.Is(err, ErrBadBody) {
				t.Fatalf("unclassified body error: %v", err)
			}
			if s := ErrorStatus(err); s == StatusErrInternal {
				t.Fatalf("wire input mapped to ERR_INTERNAL: %v", err)
			}
		}
	})
}

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
