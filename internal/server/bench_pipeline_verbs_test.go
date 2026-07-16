package server_test

import (
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"testing"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/protocol"
)

// BenchmarkExistsPipelined measures probe THROUGHPUT with depth in-flight
// EXISTS requests per connection (single-probe latency is already at the
// interrupt-driven kernel floor; pipelining is the published lever — Redis
// P=16 measured 8.5–10x on macOS loopback). ns/op ÷ depth ≈ per-probe cost.
func BenchmarkExistsPipelined(b *testing.B) {
	addr, _, cleanup := benchServer(b, 1)
	defer cleanup()

	keys := make([][32]byte, 32)
	for i := range keys {
		binary.LittleEndian.PutUint32(keys[i][:], uint32(i)+1) //nolint:gosec // G115: bench index
	}
	// benchServer's client seeded nothing here: EXISTS on missing keys is the
	// same wire shape and cost (metadata-only), so no seeding is needed.
	reqBody := protocol.AppendKeyList(nil, 0, keys)

	for _, depth := range []int{1, 4, 16} {
		b.Run(fmt.Sprintf("depth=%d", depth), func(b *testing.B) {
			nc, err := net.Dial("tcp", addr)
			if err != nil {
				b.Fatal(err)
			}
			defer nc.Close()
			_ = nc.(*net.TCPConn).SetNoDelay(true)
			helloB2(b, nc)

			b.ResetTimer()
			done := make(chan error, 1)
			tokens := make(chan struct{}, depth) // window: at most depth in flight
			// Reader: drain b.N responses, releasing one window slot each.
			go func() {
				hb := make([]byte, protocol.HeaderSize)
				body := make([]byte, 256)
				for i := 0; i < b.N; i++ {
					if _, err := io.ReadFull(nc, hb); err != nil {
						done <- err
						return
					}
					h, err := protocol.ParseHeader(hb, protocol.DefaultMaxFrameLen)
					if err != nil {
						done <- err
						return
					}
					if int(h.PayloadLen) > cap(body) {
						body = make([]byte, h.PayloadLen)
					}
					if _, err := io.ReadFull(nc, body[:h.PayloadLen]); err != nil {
						done <- err
						return
					}
					<-tokens
				}
				done <- nil
			}()
			// Writer: blocks once depth requests are unanswered.
			hb := make([]byte, protocol.HeaderSize)
			for i := 0; i < b.N; i++ {
				tokens <- struct{}{}
				h := protocol.Header{
					Opcode:     protocol.OpBatchExists,
					RequestID:  uint64(i) + 2,        //nolint:gosec // G115: bench counter
					PayloadLen: uint32(len(reqBody)), //nolint:gosec // G115: bench body
				}
				h.MarshalTo(hb)
				bufs := net.Buffers{hb, reqBody}
				if _, err := bufs.WriteTo(nc); err != nil {
					b.Fatal(err)
				}
			}
			if err := <-done; err != nil {
				b.Fatal(err)
			}
		})
	}
}

// BenchmarkPutPipelined_1MB measures the 1-RTT optimistic PUT shape: BEGIN,
// CHUNK, and COMMIT are written back-to-back WITHOUT waiting for the BEGIN
// ack, then both acks are read. Protocol-legal per §5: a rejected/exists BEGIN
// tombstones the id, arriving chunks are discarded, and the COMMIT answers
// ERR_STALE_STREAM — which the client maps from the BEGIN status. Compares
// against the two-RTT product client (BenchmarkPut_1MB).
func BenchmarkPutPipelined_1MB(b *testing.B) {
	addr, _, cleanup := benchServer(b, 1)
	defer cleanup()

	const sz = 1 << 20
	blob := make([]byte, sz)
	for j := range blob {
		blob[j] = byte(j)
	}
	sum := xxh3.Hash(blob)

	nc, err := net.Dial("tcp", addr)
	if err != nil {
		b.Fatal(err)
	}
	defer nc.Close()
	_ = nc.(*net.TCPConn).SetNoDelay(true)
	helloB2(b, nc)

	readAck := func(hb []byte) (protocol.Status, error) {
		for {
			if _, err := io.ReadFull(nc, hb); err != nil {
				return 0, err
			}
			h, err := protocol.ParseHeader(hb, protocol.DefaultMaxFrameLen)
			if err != nil {
				return 0, err
			}
			body := make([]byte, h.PayloadLen)
			if _, err := io.ReadFull(nc, body); err != nil {
				return 0, err
			}
			// Skip unsolicited NOP/CREDIT keepalives (§8 rule 4), as the real
			// client's nextHeader does.
			if h.Opcode == protocol.OpNop && h.Flags&protocol.FlagFatal == 0 {
				continue
			}
			return protocol.Status(body[0]), nil
		}
	}

	b.SetBytes(sz)
	b.ReportAllocs()
	b.ResetTimer()
	hb := make([]byte, protocol.HeaderSize)
	hb2 := make([]byte, protocol.HeaderSize)
	hb3 := make([]byte, protocol.HeaderSize)
	rhb := make([]byte, protocol.HeaderSize)
	for i := 0; i < b.N; i++ {
		var k [32]byte
		binary.LittleEndian.PutUint64(k[:], uint64(i)+0xFEED) //nolint:gosec // G115: bench counter
		id := uint64(i) + 10                                  //nolint:gosec // G115: bench counter

		begin := protocol.AppendPutBegin(nil, protocol.PutBeginBody{TotalLen: sz, XXH3Hint: sum})
		commit := protocol.AppendPutCommit(nil, sum)
		// One burst: BEGIN + CHUNK + COMMIT, no intermediate round trips.
		bh := protocol.Header{Opcode: protocol.OpPutStream, Flags: protocol.WithSubOp(0, protocol.PutBegin), Key: k, RequestID: id, PayloadLen: uint32(len(begin))} //nolint:gosec // G115: bench body
		ch := protocol.Header{Opcode: protocol.OpPutStream, Flags: protocol.WithSubOp(0, protocol.PutChunk), Key: k, RequestID: id, PayloadLen: sz}
		mh := protocol.Header{Opcode: protocol.OpPutStream, Flags: protocol.WithSubOp(0, protocol.PutCommit), Key: k, RequestID: id, PayloadLen: uint32(len(commit))} //nolint:gosec // G115: bench body
		bh.MarshalTo(hb)
		ch.MarshalTo(hb2)
		mh.MarshalTo(hb3)
		bufs := net.Buffers{hb, begin, hb2, blob, hb3, commit}
		if _, err := bufs.WriteTo(nc); err != nil {
			b.Fatal(err)
		}
		bst, err := readAck(rhb) // BEGIN ack
		if err != nil {
			b.Fatal(err)
		}
		cst, err := readAck(rhb) // COMMIT ack
		if err != nil {
			b.Fatal(err)
		}
		if bst != protocol.StatusOK || cst != protocol.StatusOK {
			b.Fatalf("begin=%s commit=%s", bst, cst)
		}
		// Delete to keep the store bounded (payload-free metadata op).
		del := protocol.AppendKeyList(nil, 0, [][32]byte{k})
		dh := protocol.Header{Opcode: protocol.OpDelete, RequestID: id + 1, PayloadLen: uint32(len(del))} //nolint:gosec // G115: bench body
		dh.MarshalTo(hb)
		dbufs := net.Buffers{hb, del}
		if _, err := dbufs.WriteTo(nc); err != nil {
			b.Fatal(err)
		}
		if _, err := readAck(rhb); err != nil {
			b.Fatal(err)
		}
	}
}

// helloB2 runs a HELLO on a raw conn (shared by the verb pipeline benches).
func helloB2(b *testing.B, nc net.Conn) {
	b.Helper()
	body := protocol.AppendHelloReq(nil, protocol.HelloReq{
		ProtoMin: 1, ProtoMax: 1, Token: []byte(testToken), Namespace: "t",
	})
	h := protocol.Header{Opcode: protocol.OpHello, RequestID: 1, PayloadLen: uint32(len(body))} //nolint:gosec // G115: bench body
	hb := make([]byte, protocol.HeaderSize)
	h.MarshalTo(hb)
	bufs := net.Buffers{hb, body}
	if _, err := bufs.WriteTo(nc); err != nil {
		b.Fatal(err)
	}
	rh := make([]byte, protocol.HeaderSize)
	if _, err := io.ReadFull(nc, rh); err != nil {
		b.Fatal(err)
	}
	ph, err := protocol.ParseHeader(rh, protocol.DefaultMaxFrameLen)
	if err != nil {
		b.Fatal(err)
	}
	rb := make([]byte, ph.PayloadLen)
	if _, err := io.ReadFull(nc, rb); err != nil {
		b.Fatal(err)
	}
	if protocol.Status(rb[0]) != protocol.StatusOK {
		b.Fatalf("hello: %s", protocol.Status(rb[0]))
	}
}
