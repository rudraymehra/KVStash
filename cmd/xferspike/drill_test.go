package main

import (
	"io"
	"net"
	"testing"
)

// The frame format (magic, headerSize, frameHeader, encodeHeader, decodeHeader)
// lives in frame.go — this drill was its ancestor and now shares that code.

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
