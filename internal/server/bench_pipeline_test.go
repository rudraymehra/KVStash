package server_test

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"testing"
	"time"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/config"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/server"
	"github.com/kvstash/kvblockd/internal/store/ramstub"
	"github.com/kvstash/kvblockd/pkg/client"
)

// BenchmarkBatchGetPipelined measures IN-ORDER request pipelining: `depth`
// BATCH_GETs are kept in flight per connection, so the server is already
// writing response N+1 while the client finishes reading response N — the
// request-turnaround bubble the synchronous client pays once per op
// disappears. The server processes one connection's requests sequentially, so
// responses arrive in request order and no FEAT_OOO demux is needed. This is
// a bench-only raw client; the pooled client stays synchronous per connection
// until the OOO demux lands (recorded client-roadmap item).
//
// Verification parity with the gate bench: every payload byte is read into a
// per-op destination set and xxh3-verified on a sidecar goroutine.
func BenchmarkBatchGetPipelined(b *testing.B) {
	cfg := config.Default()
	cfg.ListenAddr = "127.0.0.1:0"
	cfg.SockSndBuf = envBytes("KVB_BENCH_SOCKBUF", 0)
	cfg.SockRcvBuf = cfg.SockSndBuf
	cfg.WriteChunkBytes = envBytes("KVB_BENCH_CHUNK", 1<<20)
	srv := server.New(cfg, ramstub.New(), server.NewNamespaces("t", 7, testToken))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	addr, err := srv.Start(ctx)
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		dctx, dc := context.WithTimeout(context.Background(), 5*time.Second)
		defer dc()
		srv.Drain(dctx)
	}()

	const n = 32
	const sz = 1 << 20
	keys := make([][32]byte, n)
	{
		seed, err := client.Dial(ctx, addr, client.Options{Streams: 1, Namespace: "t", Token: testToken})
		if err != nil {
			b.Fatal(err)
		}
		for i := range keys {
			binary.LittleEndian.PutUint32(keys[i][:], uint32(i)+1) //nolint:gosec // G115: bench index
			blob := make([]byte, sz)
			for j := range blob {
				blob[j] = byte(i)
			}
			if err := seed.Put(ctx, keys[i], blob); err != nil {
				seed.Close()
				b.Fatal(err)
			}
		}
		seed.Close()
	}
	reqBody := protocol.AppendKeyList(nil, 0, keys)

	for _, streams := range []int{1, 2, 4} {
		for _, depth := range []int{1, 2, 4} {
			b.Run(fmt.Sprintf("streams=%d/depth=%d", streams, depth), func(b *testing.B) {
				conns := make([]net.Conn, streams)
				for i := range conns {
					nc, err := net.Dial("tcp", addr)
					if err != nil {
						b.Fatal(err)
					}
					_ = nc.(*net.TCPConn).SetNoDelay(true)
					helloB(b, nc)
					conns[i] = nc
					defer nc.Close() //nolint:gocritic // bench teardown
				}

				b.SetBytes(int64(n) * sz)
				b.ResetTimer()

				// Split b.N across connections; each conn runs an independent
				// sliding window of `depth` in-flight requests.
				per := b.N / streams
				errs := make(chan error, streams)
				for ci := 0; ci < streams; ci++ {
					ops := per
					if ci == 0 {
						ops += b.N - per*streams
					}
					go func(nc net.Conn, ops int) {
						errs <- pipelineConn(nc, reqBody, ops, depth, n, sz)
					}(conns[ci], ops)
				}
				for i := 0; i < streams; i++ {
					if err := <-errs; err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// helloB runs a HELLO on a raw conn for benchmarks.
func helloB(b *testing.B, nc net.Conn) {
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

// pipelineConn drives ops GET round trips with `depth` requests in flight.
// The sender runs ahead of the reader by at most depth requests; the reader
// consumes responses in order, verifying every block on a sidecar goroutine.
func pipelineConn(nc net.Conn, reqBody []byte, ops, depth, nKeys, blockSz int) error {
	sendErr := make(chan error, 1)
	tokens := make(chan struct{}, depth)
	go func() {
		hb := make([]byte, protocol.HeaderSize)
		for i := 0; i < ops; i++ {
			tokens <- struct{}{} // cap in-flight at depth
			h := protocol.Header{
				Opcode:     protocol.OpBatchGet,
				RequestID:  uint64(i) + 2,        //nolint:gosec // G115: bench counter
				PayloadLen: uint32(len(reqBody)), //nolint:gosec // G115: bench body
			}
			h.MarshalTo(hb)
			bufs := net.Buffers{hb, reqBody}
			if _, err := bufs.WriteTo(nc); err != nil {
				sendErr <- err
				return
			}
		}
		sendErr <- nil
	}()

	into := make([][]byte, nKeys)
	for i := range into {
		into[i] = make([]byte, blockSz)
	}
	type vjob struct {
		slot int
		sum  uint64
	}
	hb := make([]byte, protocol.HeaderSize)
	small := make([]byte, 16)
	descs := make([]byte, nKeys*protocol.DescSize)

	for op := 0; op < ops; op++ {
		verifyCh := make(chan vjob, nKeys)
		verifyErr := make(chan error, 1)
		go func() {
			var err error
			for j := range verifyCh {
				if err == nil && xxh3.Hash(into[j.slot]) != j.sum {
					err = fmt.Errorf("checksum mismatch slot %d", j.slot)
				}
			}
			verifyErr <- err
		}()
		got := 0
		for {
			if _, err := io.ReadFull(nc, hb); err != nil {
				close(verifyCh)
				return err
			}
			h, err := protocol.ParseHeader(hb, protocol.DefaultMaxFrameLen)
			if err != nil {
				close(verifyCh)
				return err
			}
			if h.Opcode == protocol.OpNop && h.Flags&protocol.FlagFatal == 0 {
				if h.PayloadLen > 0 {
					if _, err := io.CopyN(io.Discard, nc, int64(h.PayloadLen)); err != nil {
						close(verifyCh)
						return err
					}
				}
				continue
			}
			if _, err := io.ReadFull(nc, small[:16]); err != nil { // preamble + idx
				close(verifyCh)
				return err
			}
			if st := protocol.Status(small[0]); st != protocol.StatusOK {
				close(verifyCh)
				return fmt.Errorf("GET status %s", st)
			}
			count := int(binary.LittleEndian.Uint32(small[4:8]))
			first := int(binary.LittleEndian.Uint32(small[8:12]))
			if _, err := io.ReadFull(nc, descs[:count*protocol.DescSize]); err != nil {
				close(verifyCh)
				return err
			}
			for i := 0; i < count; i++ {
				d := protocol.GetDesc(descs[i*protocol.DescSize:])
				slot := first + i
				if _, err := io.ReadFull(nc, into[slot][:d.Len]); err != nil {
					close(verifyCh)
					return err
				}
				verifyCh <- vjob{slot: slot, sum: d.XXH3}
			}
			got += count
			if h.Flags&protocol.FlagMore == 0 {
				break
			}
		}
		close(verifyCh)
		if err := <-verifyErr; err != nil {
			return err
		}
		if got != nKeys {
			return fmt.Errorf("got %d of %d keys", got, nKeys)
		}
		<-tokens // one response fully consumed → release a window slot
	}
	return <-sendErr
}
