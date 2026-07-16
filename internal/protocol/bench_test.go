package protocol

import (
	"testing"
)

// Codec microbenchmarks: the header pair is per-frame hot (every frame both
// directions), the key-list decode and GET-response-header append are per-op
// hot on the BATCH_GET path. All must stay zero-alloc on their steady state.

func BenchmarkMarshalHeader(b *testing.B) {
	h := Header{Opcode: OpBatchGet, Flags: FlagResp, NamespaceID: 7, RequestID: 42, PayloadLen: 1 << 20}
	dst := make([]byte, HeaderSize)
	b.ReportAllocs()
	b.SetBytes(HeaderSize)
	for i := 0; i < b.N; i++ {
		h.MarshalTo(dst)
	}
}

func BenchmarkParseHeader(b *testing.B) {
	h := Header{Opcode: OpBatchGet, Flags: FlagResp, NamespaceID: 7, RequestID: 42, PayloadLen: 1 << 20}
	buf := make([]byte, HeaderSize)
	h.MarshalTo(buf)
	b.ReportAllocs()
	b.SetBytes(HeaderSize)
	for i := 0; i < b.N; i++ {
		if _, err := ParseHeader(buf, DefaultMaxFrameLen); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeKeyList_32(b *testing.B) {
	keys := make([][32]byte, 32)
	for i := range keys {
		keys[i][0] = byte(i)
	}
	body := AppendKeyList(nil, 0, keys)
	scratch := make([][32]byte, 0, 32)
	b.ReportAllocs()
	b.SetBytes(int64(len(body)))
	for i := 0; i < b.N; i++ {
		_, out, err := DecodeKeyList(body, 512, scratch[:0])
		if err != nil {
			b.Fatal(err)
		}
		scratch = out[:0] // keep the grown backing array, as the server does
	}
}

func BenchmarkAppendGetRespHeader_32(b *testing.B) {
	descs := make([]Desc, 32)
	for i := range descs {
		descs[i] = Desc{Status: StatusOK, Len: 1 << 20, XXH3: uint64(i) * 0x9E3779B97F4A7C15}
	}
	dst := make([]byte, 0, GetRespHeaderSize(len(descs)))
	b.ReportAllocs()
	b.SetBytes(int64(GetRespHeaderSize(len(descs))))
	for i := 0; i < b.N; i++ {
		dst = AppendGetRespHeader(dst[:0], StatusOK, 0, 32, descs)
	}
}

func BenchmarkDecodePutBegin(b *testing.B) {
	body := AppendPutBegin(nil, PutBeginBody{TotalLen: 1 << 20, XXH3Hint: 0xABCD})
	b.ReportAllocs()
	b.SetBytes(int64(len(body)))
	for i := 0; i < b.N; i++ {
		if _, err := DecodePutBegin(body); err != nil {
			b.Fatal(err)
		}
	}
}
