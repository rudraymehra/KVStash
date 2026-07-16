package client

import (
	"context"
	"errors"
	"io"
	"net"
	"strings"
	"testing"
	"time"

	"github.com/kvstash/kvblockd/internal/protocol"
)

// fakeServer accepts connections, answers HELLO with OK, then hands every
// subsequent request frame to handle. It lets the client tests script exact
// server behavior (error replies, garbage) without a real store.
type fakeServer struct {
	ln net.Listener
	t  *testing.T
}

func newFakeServer(t *testing.T, handle func(conn net.Conn, h protocol.Header, body []byte) bool) *fakeServer {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	fs := &fakeServer{ln: ln, t: t}
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go fs.serve(c, handle)
		}
	}()
	return fs
}

func (fs *fakeServer) serve(c net.Conn, handle func(net.Conn, protocol.Header, []byte) bool) {
	defer c.Close()
	for {
		hb := make([]byte, protocol.HeaderSize)
		if _, err := io.ReadFull(c, hb); err != nil {
			return
		}
		h, err := protocol.ParseHeader(hb, protocol.DefaultMaxFrameLen)
		if err != nil {
			return
		}
		body := make([]byte, h.PayloadLen)
		if _, err := io.ReadFull(c, body); err != nil {
			return
		}
		if h.Opcode == protocol.OpHello {
			resp := protocol.AppendHelloResp(nil, protocol.HelloResp{
				Proto:       protocol.Version1,
				Limits:      protocol.DefaultLimits(),
				NamespaceID: 1,
				ServerName:  "fake",
			})
			replyFrame(c, h, resp)
			continue
		}
		if !handle(c, h, body) {
			return
		}
	}
}

// replyFrame writes a response frame echoing opcode/request_id with F_RESP.
func replyFrame(c net.Conn, req protocol.Header, body []byte) {
	h := protocol.Header{
		Opcode:      req.Opcode,
		Flags:       protocol.FlagResp,
		NamespaceID: req.NamespaceID,
		RequestID:   req.RequestID,
		PayloadLen:  uint32(len(body)), //nolint:gosec // G115: test body
	}
	hb := make([]byte, protocol.HeaderSize)
	h.MarshalTo(hb)
	bufs := net.Buffers{hb, body}
	_, _ = bufs.WriteTo(c)
}

func (fs *fakeServer) addr() string { return fs.ln.Addr().String() }
func (fs *fakeServer) close()       { _ = fs.ln.Close() }

// TestGetNonOKReturnsPromptly: a preamble-only error reply to BATCH_GET must
// surface as a *StatusError without reading ahead — the read-ahead deadlock
// the review found (the client used to block waiting for descriptors that an
// error response never carries).
func TestGetNonOKReturnsPromptly(t *testing.T) {
	fs := newFakeServer(t, func(c net.Conn, h protocol.Header, _ []byte) bool {
		replyFrame(c, h, protocol.AppendPreamble(nil, protocol.StatusErrBusy, 0))
		return true
	})
	defer fs.close()

	c, err := Dial(context.Background(), fs.addr(), Options{Streams: 1, Namespace: "t"})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()

	done := make(chan error, 1)
	go func() {
		keys := [][32]byte{{1}}
		into := make([][]byte, 1)
		_, err := c.BatchGet(context.Background(), keys, into)
		done <- err
	}()
	select {
	case err := <-done:
		var se *StatusError
		if !errors.As(err, &se) || se.Status != protocol.StatusErrBusy {
			t.Fatalf("want StatusError{ERR_BUSY}, got %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("BatchGet hung on a non-OK response (read-ahead deadlock)")
	}
}

// TestVerifyCatchesCorruption: a GET whose payload does not match its
// descriptor xxh3 fails with a corruption error by default, and is passed
// through when Options.SkipVerify is set (the caller owns re-verification).
func TestVerifyCatchesCorruption(t *testing.T) {
	respond := func(c net.Conn, h protocol.Header, _ []byte) bool {
		payload := []byte{1, 2, 3, 4, 5, 6, 7, 8}
		descs := []protocol.Desc{{Status: protocol.StatusOK, Len: 8, XXH3: 0xBAD}} // wrong sum
		body := protocol.AppendGetRespHeader(nil, protocol.StatusOK, 0, 1, descs)
		body = append(body, payload...)
		replyFrame(c, h, body)
		return true
	}

	for _, skip := range []bool{false, true} {
		fs := newFakeServer(t, respond)
		c, err := Dial(context.Background(), fs.addr(), Options{Streams: 1, Namespace: "t", SkipVerify: skip})
		if err != nil {
			fs.close()
			t.Fatal(err)
		}
		keys := [][32]byte{{9}}
		into := make([][]byte, 1)
		_, err = c.BatchGet(context.Background(), keys, into)
		if skip && err != nil {
			t.Fatalf("SkipVerify: unexpected error %v", err)
		}
		if !skip && (err == nil || !strings.Contains(err.Error(), "checksum")) {
			t.Fatalf("verify on: want checksum mismatch, got %v", err)
		}
		c.Close()
		fs.close()
	}
}

// TestPoolHealsAfterDesync: a framing error must evict the poisoned connection
// and redial, so the NEXT call on the same pool works — the pool-desync bug
// the review found (three reviewers, convergent).
func TestPoolHealsAfterDesync(t *testing.T) {
	poisoned := false
	fs := newFakeServer(t, func(c net.Conn, h protocol.Header, _ []byte) bool {
		if !poisoned {
			poisoned = true
			// Garbage: an unparseable response header desyncs the stream.
			_, _ = c.Write(make([]byte, protocol.HeaderSize))
			return false // close this conn after the garbage
		}
		replyFrame(c, h, protocol.AppendPreamble(nil, protocol.StatusOK, 0))
		return true
	})
	defer fs.close()

	c, err := Dial(context.Background(), fs.addr(), Options{Streams: 1, Namespace: "t"})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	ctx := context.Background()

	if _, err := c.Stats(ctx); err == nil {
		t.Fatal("expected an error from the garbage response")
	}
	// The pool must have replaced the poisoned connection: this call gets the
	// scripted OK (an un-healed pool would hand back the desynced conn or hang).
	ctx2, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	if _, err := c.Stats(ctx2); err != nil {
		t.Fatalf("pool did not heal after desync: %v", err)
	}
}
