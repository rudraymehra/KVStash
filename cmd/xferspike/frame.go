package main

import (
	"encoding/binary"
	"errors"
)

// The xferspike wire frame: a fixed 16-byte header followed by a payload of
// header.length bytes. This is a deliberately minimal ancestor of the real
// 64-byte kvblockd protocol header (see docs/PROTOCOL.md, Week 2) — enough to
// measure raw transport throughput and nothing more.
//
// Byte order (locked here, ahead of PROTOCOL.md): the magic field is written
// big-endian so the four wire bytes read literally as "KVB1" in a hexdump — the
// universal convention for protocol magics. The numeric fields (seq, length)
// are little-endian to match host order on our amd64/arm64 targets.
const (
	// magic is the ASCII "KVB1", carried big-endian so the wire bytes are
	// 0x4B 0x56 0x42 0x31. Every frame starts with it so a desynchronized
	// stream is detected immediately instead of silently mis-parsed.
	magic uint32 = 0x4B564231

	headerSize = 16
)

type frameHeader struct {
	magic  uint32
	seq    uint64
	length uint32
}

var errBadMagic = errors.New("bad magic: stream out of sync")

// putHeader writes h into the first headerSize bytes of buf without allocating.
// It panics if buf is too short — callers own a correctly sized buffer.
func putHeader(buf []byte, h frameHeader) {
	binary.BigEndian.PutUint32(buf[0:4], h.magic)
	binary.LittleEndian.PutUint64(buf[4:12], h.seq)
	binary.LittleEndian.PutUint32(buf[12:16], h.length)
}

// encodeHeader allocates a fresh headerSize buffer and writes h into it.
func encodeHeader(h frameHeader) []byte {
	buf := make([]byte, headerSize)
	putHeader(buf, h)
	return buf
}

// decodeHeader parses a headerSize buffer, returning errBadMagic if the frame
// does not begin with the expected magic value.
func decodeHeader(buf []byte) (frameHeader, error) {
	var h frameHeader
	h.magic = binary.BigEndian.Uint32(buf[0:4])
	h.seq = binary.LittleEndian.Uint64(buf[4:12])
	h.length = binary.LittleEndian.Uint32(buf[12:16])
	if h.magic != magic {
		return h, errBadMagic
	}
	return h, nil
}
