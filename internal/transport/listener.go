package transport

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"sync/atomic"
	"syscall"
	"time"

	"golang.org/x/sys/unix"

	"github.com/kvstash/kvblockd/internal/protocol"
)

// Config tunes a Listener and every connection it accepts.
type Config struct {
	// Addr is the TCP listen address ("host:port").
	Addr string

	// SndBuf/RcvBuf are per-connection socket buffer REQUESTS in bytes
	// (0 = OS default). 16 MiB saturated 50 GbE on the benchmark rig; on an
	// untuned host the kernel silently clamps to net.core.*mem_max — the
	// listener logs the effective sizes once so a mystery throughput
	// regression points at host sysctls, not at this code. Host-level tuning
	// (BBR, MTU, rmem_max) is deployment scope: bench/rigs/sysctl-esnet.conf.
	SndBuf, RcvBuf int
	// NoDelay disables Nagle (default true via DefaultConfig).
	NoDelay bool

	// PreNegCap is the ParseHeader payload cap before HELLO negotiation. It
	// MUST be >= protocol.MaxHelloBody or a legal maximum-size HELLO is
	// rejected pre-auth; DefaultConfig derives it, and startConn floors it.
	PreNegCap uint32

	// CoalesceBytes is the writev flush target when the write queue is
	// backlogged (16 MiB — the coalescing knee; never delays a lone frame).
	CoalesceBytes int

	// WriteStallTimeout is the per-flush write deadline: 2×stream_timeout
	// (§8 rule 5's zero-drain closure). Use StallTimeout(). A zero or too-small
	// value would let a dead peer wedge the reader/writer cycle, so startConn
	// floors it at minStallTimeout.
	WriteStallTimeout time.Duration

	// BodyReadTimeout bounds how long a single frame body (or a skipped
	// payload) may take to arrive once its header has been read — the
	// slow-loris guard that keeps a 2-byte dribble from pinning a full
	// max_frame_len buffer for the whole IdleReadTimeout.
	BodyReadTimeout time.Duration

	// IdleReadTimeout closes connections with no inbound frame in progress
	// (clients are advised to NOP-keepalive when idle >10 s; default generous).
	IdleReadTimeout time.Duration

	// GrantTick is how often an otherwise-idle writer ships pending credit
	// grants in an unsolicited NOP/CREDIT frame (§8 rule 4). Zero → 100 ms.
	GrantTick time.Duration
}

// minStallTimeout floors WriteStallTimeout so a misassembled Config can never
// disable the deadline that breaks a dead-peer reader/writer wedge.
const minStallTimeout = 5 * time.Second

// DefaultConfig returns production defaults for the given address and
// negotiated stream timeout.
func DefaultConfig(addr string, streamTimeoutMS uint32) Config {
	preNeg := uint32(256 << 10)
	if preNeg < protocol.MaxHelloBody {
		preNeg = protocol.MaxHelloBody
	}
	return Config{
		Addr:              addr,
		SndBuf:            16 << 20,
		RcvBuf:            16 << 20,
		NoDelay:           true,
		PreNegCap:         preNeg,
		CoalesceBytes:     16 << 20,
		WriteStallTimeout: StallTimeout(streamTimeoutMS),
		BodyReadTimeout:   time.Duration(streamTimeoutMS) * time.Millisecond,
		IdleReadTimeout:   5 * time.Minute,
		GrantTick:         100 * time.Millisecond,
	}
}

func (cfg Config) grantTickInterval() time.Duration {
	if cfg.GrantTick > 0 {
		return cfg.GrantTick
	}
	return 100 * time.Millisecond
}

// Listener accepts kvblockd connections.
type Listener struct {
	ln     net.Listener
	cfg    Config
	logged atomic.Bool // guards the one-shot effective-buffer log across concurrent Accept
}

// Listen opens the data-plane listener. The only listening-socket option is
// SO_REUSEADDR — buffer sizes are per-connection concerns (inheritance across
// accept is not reliable across kernels), applied in Accept. WriteStallTimeout
// is floored so a stalled peer can always be reclaimed.
func Listen(ctx context.Context, cfg Config) (*Listener, error) {
	if cfg.WriteStallTimeout < minStallTimeout {
		cfg.WriteStallTimeout = minStallTimeout
	}
	if cfg.PreNegCap < protocol.MaxHelloBody {
		cfg.PreNegCap = protocol.MaxHelloBody
	}
	lc := net.ListenConfig{
		Control: func(_, _ string, c syscall.RawConn) error {
			var serr error
			err := c.Control(func(fd uintptr) {
				serr = unix.SetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_REUSEADDR, 1)
			})
			if err != nil {
				return err
			}
			return serr
		},
	}
	ln, err := lc.Listen(ctx, "tcp", cfg.Addr)
	if err != nil {
		return nil, fmt.Errorf("transport: listen %s: %w", cfg.Addr, err)
	}
	return &Listener{ln: ln, cfg: cfg}, nil
}

// Addr returns the bound address (useful with ":0" in tests).
func (l *Listener) Addr() net.Addr { return l.ln.Addr() }

// Accept blocks for the next connection, tunes it, and starts its loops with
// the given buffer source and handler. Safe to call from multiple goroutines.
func (l *Listener) Accept(bs BufferSource, h FrameHandler) (*Conn, error) {
	nc, err := l.ln.Accept()
	if err != nil {
		return nil, err
	}
	l.tune(nc)
	return startConn(nc, bs, h, l.cfg), nil
}

// Close stops accepting. Existing connections are unaffected.
func (l *Listener) Close() error { return l.ln.Close() }

// tune applies the per-connection socket options (the promoted pattern from
// the benchmark rig) and, once per listener, logs the effective buffer sizes
// so silent kernel clamping is visible.
func (l *Listener) tune(nc net.Conn) {
	tc, ok := nc.(*net.TCPConn)
	if !ok {
		return
	}
	if l.cfg.NoDelay {
		_ = tc.SetNoDelay(true)
	}
	if l.cfg.SndBuf > 0 {
		_ = tc.SetWriteBuffer(l.cfg.SndBuf)
	}
	if l.cfg.RcvBuf > 0 {
		_ = tc.SetReadBuffer(l.cfg.RcvBuf)
	}
	_ = tc.SetKeepAlive(true)
	_ = tc.SetKeepAlivePeriod(60 * time.Second)

	if (l.cfg.SndBuf > 0 || l.cfg.RcvBuf > 0) && l.logged.CompareAndSwap(false, true) {
		if snd, rcv, ok := effectiveBufSizes(tc); ok {
			// The kernel reports 2× the requested value when it honors a
			// setsockopt (bookkeeping overhead) and the sysctl cap otherwise.
			if snd < l.cfg.SndBuf || rcv < l.cfg.RcvBuf {
				slog.Warn("transport: kernel clamped socket buffers — apply host sysctls (see bench/rigs/sysctl-esnet.conf)",
					"requested_snd", l.cfg.SndBuf, "effective_snd", snd,
					"requested_rcv", l.cfg.RcvBuf, "effective_rcv", rcv)
			} else {
				slog.Info("transport: socket buffers", "effective_snd", snd, "effective_rcv", rcv)
			}
		}
	}
}
