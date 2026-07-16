// Package client is the reference Go client for kvblockd: dial + HELLO, a small
// connection pool, and the batch verbs. This week's subset is deliberately
// synchronous per connection (one in-flight request each) — pipelined
// out-of-order demux and rendezvous hashing across nodes arrive in later
// weeks. Concurrency comes from the pool: N connections serve N callers.
package client

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	"github.com/kvstash/kvblockd/internal/protocol"
)

// Options configures a Dial.
type Options struct {
	Streams      int    // pooled connections (default 8)
	Namespace    string // tenant namespace
	Token        string // bearer token
	MaxBatchKeys uint32 // client proposal (0 = accept server default)
	MaxFrameLen  uint32 // client proposal (0 = accept server default)
	DialTimeout  time.Duration
	// SockSndBuf / SockRcvBuf request per-connection socket buffer sizes in
	// bytes (0 = OS default). Batch responses are tens of MiB; an OS-default
	// receive buffer (a few hundred KiB) caps the sender's in-flight window and
	// fragments reads into small chunks — the server side already defaults to
	// 16 MiB, and a matched client buffer is what lets a GET response stream at
	// wire speed. The kernel may clamp the request (Linux: net.core.rmem_max).
	SockSndBuf int
	SockRcvBuf int

	// SkipVerify disables the client-side xxh3 check of every GET payload
	// against its descriptor checksum. Default false: verify. Only set this
	// when the consumer re-verifies downstream anyway — TCP's checksum alone
	// does not protect against server-side corruption.
	SkipVerify bool
}

// Client is a pool of authenticated connections to one kvblockd endpoint.
type Client struct {
	addr   string
	opts   Options
	ns     uint32
	limits protocol.Limits
	feats  uint64

	mu     sync.Mutex
	closed bool
	pool   chan *conn

	closeOnce sync.Once
}

// Dial opens Options.Streams connections, runs HELLO on each, and returns a
// ready client. All connections share the negotiated limits from the first
// handshake.
func Dial(ctx context.Context, addr string, o Options) (*Client, error) {
	if o.Streams <= 0 {
		o.Streams = 8
	}
	if o.DialTimeout <= 0 {
		o.DialTimeout = 5 * time.Second
	}
	c := &Client{addr: addr, opts: o, pool: make(chan *conn, o.Streams)}
	for i := 0; i < o.Streams; i++ {
		cn, err := dialConn(ctx, addr, o)
		if err != nil {
			c.Close()
			return nil, err
		}
		if i == 0 {
			c.ns, c.limits, c.feats = cn.ns, cn.limits, cn.feats
		}
		c.pool <- cn
	}
	return c, nil
}

// Close shuts the pool. Connections currently borrowed are closed when their
// caller releases them (release honors the closed flag).
func (c *Client) Close() {
	c.closeOnce.Do(func() {
		c.mu.Lock()
		c.closed = true
		c.mu.Unlock()
		for {
			select {
			case cn := <-c.pool:
				_ = cn.nc.Close()
			default:
				return
			}
		}
	})
}

// Limits are the negotiated per-connection caps.
func (c *Client) Limits() protocol.Limits { return c.limits }

// get borrows a connection; a borrowed connection is used by exactly one caller
// at a time (synchronous request/response for now).
func (c *Client) get(ctx context.Context) (*conn, error) {
	select {
	case cn := <-c.pool:
		c.mu.Lock()
		closed := c.closed
		c.mu.Unlock()
		if closed {
			_ = cn.nc.Close()
			return nil, ErrClosed
		}
		return cn, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// release returns a connection after a request. A clean result (nil error, or a
// protocol-level *StatusError which leaves the stream in sync) re-pools the
// connection; any framing/I/O error means the connection is desynchronized, so
// it is closed and a fresh one dialed to replace it — a poisoned connection is
// never handed to the next caller (the pool-desync bug the review found).
func (c *Client) release(cn *conn, err error) {
	clean := err == nil
	if !clean {
		var se *StatusError
		clean = errors.As(err, &se)
	}
	c.mu.Lock()
	closed := c.closed
	c.mu.Unlock()
	if closed {
		_ = cn.nc.Close()
		return
	}
	if clean {
		c.pool <- cn
		return
	}
	_ = cn.nc.Close()
	// Redial a replacement so the pool keeps its size; on failure the pool
	// shrinks by one (still functional) rather than re-pooling a dead conn.
	ctx, cancel := context.WithTimeout(context.Background(), c.opts.DialTimeout)
	defer cancel()
	if repl, derr := dialConn(ctx, c.addr, c.opts); derr == nil {
		c.pool <- repl
	}
}

// conn is one pooled connection with its own request_id counter and scratch.
type conn struct {
	nc     net.Conn
	ns     uint32
	limits protocol.Limits
	feats  uint64
	nextID uint64
	hdr    [protocol.HeaderSize]byte // reusable header scratch
	rbuf   []byte                    // reusable readN scratch (single-caller conn)
	wbuf   []byte                    // reusable request-body scratch (WriteTo is synchronous)
}

// reqBuf hands out the request-body scratch (len 0, grown capacity kept).
func (cn *conn) reqBuf() []byte { return cn.wbuf[:0] }

// keepReq remembers a (possibly grown) request body's backing array.
func (cn *conn) keepReq(b []byte) { cn.wbuf = b[:0] }

func dialConn(ctx context.Context, addr string, o Options) (*conn, error) {
	d := net.Dialer{Timeout: o.DialTimeout}
	nc, err := d.DialContext(ctx, "tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("client: dial %s: %w", addr, err)
	}
	if tc, ok := nc.(*net.TCPConn); ok {
		_ = tc.SetNoDelay(true)
		if o.SockSndBuf > 0 {
			_ = tc.SetWriteBuffer(o.SockSndBuf)
		}
		if o.SockRcvBuf > 0 {
			_ = tc.SetReadBuffer(o.SockRcvBuf)
		}
	}
	cn := &conn{nc: nc, nextID: 1}
	if err := cn.hello(o); err != nil {
		_ = nc.Close()
		return nil, err
	}
	return cn, nil
}

func (cn *conn) hello(o Options) error {
	req := protocol.HelloReq{
		ProtoMin: protocol.Version1,
		ProtoMax: protocol.Version1,
		// Advertise only what this client implements: it parses the EXISTS
		// bitmap but is synchronous per connection, so it does NOT support OOO.
		Features:     protocol.FeatExistsBitmap,
		MaxBatchKeys: o.MaxBatchKeys,
		MaxFrameLen:  o.MaxFrameLen,
		Token:        []byte(o.Token),
		Namespace:    o.Namespace,
		ClientName:   "kvblockd-go",
	}
	body := protocol.AppendHelloReq(nil, req)
	// request_id 0 is reserved; HELLO uses a nonzero id like any request.
	if err := cn.writeFrame(protocol.OpHello, 0, [32]byte{}, cn.id(), body); err != nil {
		return err
	}
	respBody, err := cn.readFrame()
	if err != nil {
		return err
	}
	p, resp, err := protocol.DecodeHelloResp(respBody)
	if err != nil {
		return fmt.Errorf("client: bad HELLO response: %w", err)
	}
	if p.Status != protocol.StatusOK {
		return fmt.Errorf("client: HELLO rejected: %s", p.Status)
	}
	cn.ns, cn.limits, cn.feats = resp.NamespaceID, resp.Limits, resp.Features
	return nil
}

func (cn *conn) id() uint64 {
	id := cn.nextID
	cn.nextID++
	return id
}

// writeFrame marshals a header and writes header+body in one writev.
func (cn *conn) writeFrame(op protocol.Opcode, flags uint16, key [32]byte, reqID uint64, body []byte) error {
	h := protocol.Header{
		Opcode:     op,
		Flags:      flags,
		RequestID:  reqID,
		Key:        key,
		PayloadLen: uint32(len(body)), //nolint:gosec // G115: bodies are bounded by the negotiated max_frame_len
	}
	h.MarshalTo(cn.hdr[:])
	bufs := net.Buffers{cn.hdr[:]}
	if len(body) > 0 {
		bufs = append(bufs, body)
	}
	_, err := bufs.WriteTo(cn.nc)
	return err
}

// nextHeader reads the next response frame header, transparently skipping any
// unsolicited NOP/CREDIT frames the server interleaves (§8 rule 4). This week
// the client does not track its own send window (its requests are tiny and the
// server window is large), so a credit grant is simply consumed and ignored —
// but the frame on the wire MUST be skipped or every subsequent read desyncs.
func (cn *conn) nextHeader() (protocol.Header, error) {
	for {
		var hb [protocol.HeaderSize]byte
		if _, err := io.ReadFull(cn.nc, hb[:]); err != nil {
			return protocol.Header{}, err
		}
		h, err := protocol.ParseHeader(hb[:], protocol.DefaultMaxFrameLen)
		if err != nil {
			return protocol.Header{}, fmt.Errorf("client: parse response header: %w", err)
		}
		// Skip pure NOP/CREDIT keepalives — but NOT a protocol-fatal report,
		// which also rides opcode 0 with F_FATAL and carries the status the
		// caller must see (§9). Return the fatal frame so callers decode it.
		if h.Opcode == protocol.OpNop && h.Flags&protocol.FlagFatal == 0 {
			if h.PayloadLen > 0 {
				if _, err := io.CopyN(io.Discard, cn.nc, int64(h.PayloadLen)); err != nil {
					return protocol.Header{}, err
				}
			}
			continue
		}
		return h, nil
	}
}

// readFrame reads one full response frame's payload, skipping NOPs. The
// returned slice aliases the conn's readN scratch — valid until the next
// read on this conn; callers copy anything that outlives their verb.
func (cn *conn) readFrame() ([]byte, error) {
	h, err := cn.nextHeader()
	if err != nil {
		return nil, err
	}
	return cn.readN(int(h.PayloadLen))
}

// maxReadReuse bounds the scratch readN retains across calls. Metadata
// responses (preamble, descriptors, EXISTS bitmap, HELLO, Stats) are all well
// under this; GET block payloads bypass readN entirely (read straight into the
// caller's into[slot]). A larger response — e.g. a malicious server inflating a
// metadata frame's payload_len up to the negotiated cap — is read into a
// transient buffer that GC reclaims, so it can never pin the scratch at up to
// max_frame_len for the pooled connection's whole life.
const maxReadReuse = 1 << 20

// readN reads exactly n bytes. For n <= maxReadReuse it uses the conn's
// reusable scratch (valid only until the next readN — callers copy out what
// outlives the call); larger reads use a transient buffer. A conn has exactly
// one caller at a time, so the scratch needs no locking.
func (cn *conn) readN(n int) ([]byte, error) {
	if n > maxReadReuse {
		b := make([]byte, n)
		_, err := io.ReadFull(cn.nc, b)
		return b, err
	}
	if cap(cn.rbuf) < n {
		cn.rbuf = make([]byte, maxReadReuse)
	}
	b := cn.rbuf[:n]
	_, err := io.ReadFull(cn.nc, b)
	return b, err
}

// StatusError reports a non-OK batch-level response status from the server.
type StatusError struct {
	Op     protocol.Opcode
	Status protocol.Status
}

func (e *StatusError) Error() string {
	return fmt.Sprintf("client: op %#x returned %s", uint8(e.Op), e.Status)
}

var (
	errShortResponse = errors.New("client: truncated response")
	// ErrClosed is returned by a verb when the client has been closed.
	ErrClosed = errors.New("client: closed")
)
