package main

import (
	"encoding/binary"
	"errors"
	"io"
	"net"
	"testing"
)

const magic uint32 = 0x4B564231
const headerSize = 16

type frameHeader struct {
	magic  uint32
	seq    uint64
	length uint32
}

func encodeHeader(h frameHeader) []byte {
	buf := make([]byte, headerSize)
	binary.LittleEndian.PutUint32(buf[0:4], h.magic)
	binary.LittleEndian.PutUint64(buf[4:12], h.seq)
	binary.LittleEndian.PutUint32(buf[12:16], h.length)
	return buf
}

var errBadMagic = errors.New("bad magic: stream out of sync")

func decodeHeader(buf []byte) (frameHeader, error) {
	var h frameHeader
	h.magic = binary.LittleEndian.Uint32(buf[0:4])
	h.seq = binary.LittleEndian.Uint64(buf[4:12])
	h.length = binary.LittleEndian.Uint32(buf[12:16])
	if h.magic != magic {
		return h, errBadMagic
	}
	return h, nil
}

func startEchoServer(tb testing.TB) net.Listener {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		tb.Fatal(err)
	}
	go func() {
		for {
			conn, err := ln.Accept()
			if err != nil {
				return
			}
			go echoConn(conn)
		}
	}()
	return ln
}

func echoConn(conn net.Conn) {
	defer conn.Close()
	hdr := make([]byte, headerSize)
	payload := make([]byte, 1<<20)
	for {
		if _, err := io.ReadFull(conn, hdr); err != nil {
			return
		}
		h, err := decodeHeader(hdr)
		if err != nil {
			return
		}
		body := payload[:h.length]
		if _, err := io.ReadFull(conn, body); err != nil {
			return
		}
		if _, err := conn.Write(hdr); err != nil {
			return
		}
		if _, err := conn.Write(body); err != nil {
			return
		}
	}
}

func runEcho(b *testing.B, useWritev bool) {
	ln := startEchoServer(b)
	defer ln.Close()

	conn, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		b.Fatal(err)
	}
	defer conn.Close()

	payload := make([]byte, 1<<20)
	frame := make([]byte, headerSize+len(payload))
	respHdr := make([]byte, headerSize)
	respBody := make([]byte, 1<<20)

	b.SetBytes(1 << 20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hdr := encodeHeader(frameHeader{
			magic:  magic,
			seq:    uint64(i),
			length: uint32(len(payload)),
		})
		if useWritev {
			bufs := net.Buffers{hdr, payload}
			if _, err := bufs.WriteTo(conn); err != nil {
				b.Fatal(err)
			}
		} else {
			copy(frame[0:headerSize], hdr)
			copy(frame[headerSize:], payload)
			if _, err := conn.Write(frame); err != nil {
				b.Fatal(err)
			}
		}
		if _, err := io.ReadFull(conn, respHdr); err != nil {
			b.Fatal(err)
		}
		if _, err := io.ReadFull(conn, respBody); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEchoSingleWrite(b *testing.B) { runEcho(b, false) }

func BenchmarkEchoWritev(b *testing.B) { runEcho(b, true) }
