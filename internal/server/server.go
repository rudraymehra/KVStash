// Package server wires the transport to a store: it accepts connections,
// enforces HELLO-first authentication, negotiates limits, and dispatches the
// eight KVB1 verbs (PROTOCOL.md §3) — including the two-phase PUT_STREAM state
// machine (§5) — against a Store. The Store is the temporary in-heap
// ramstub; the interface is what the DRAM/NVMe/S3 tiers implement.
package server

import (
	"context"
	"log/slog"
	"sync"

	"github.com/kvstash/kvblockd/internal/config"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/transport"
)

// Store is the block-store surface the server dispatches to. ramstub
// implements it now; the real DRAM/NVMe/S3 tiers implement it later.
// Put takes OWNERSHIP of data — the caller must not touch the slice after the
// call (the PUT commit path hands over its staging extent instead of copying).
type Store interface {
	ExistsPrefix(ns uint32, keys [][32]byte, withBitmap bool) (nConsecutive uint32, perKey []protocol.Status)
	Get(ns uint32, key [32]byte) (data []byte, xxh3 uint64, ok bool)
	Put(ns uint32, key [32]byte, data []byte, xxh3 uint64) protocol.Status
	Contains(ns uint32, key [32]byte) bool
	Delete(ns uint32, key [32]byte, force bool) protocol.Status
	Stats() []byte
}

// Server accepts and serves KVB1 connections.
type Server struct {
	cfg    config.Config
	store  Store
	ns     *Namespaces
	lcfg   transport.Config
	ln     *transport.Listener
	logger *slog.Logger

	mu       sync.Mutex
	draining bool
	conns    map[*transport.Conn]struct{}
}

// New builds a server. cfg supplies the negotiated limits and timeouts; ns is
// the token→namespace table; store is the block store.
func New(cfg config.Config, store Store, ns *Namespaces) *Server {
	lcfg := transport.DefaultConfig(cfg.ListenAddr, cfg.StreamTimeoutMS)
	lcfg.SndBuf = cfg.SockSndBuf
	lcfg.RcvBuf = cfg.SockRcvBuf
	lcfg.WriteChunkBytes = cfg.WriteChunkBytes
	return &Server{
		cfg:    cfg,
		store:  store,
		ns:     ns,
		lcfg:   lcfg,
		logger: slog.Default(),
		conns:  make(map[*transport.Conn]struct{}),
	}
}

// Start binds the listener (so the address is available immediately, useful
// with ":0") and runs the accept loop in the background. It returns the bound
// address. Cancelling ctx, or calling Drain, stops accepting.
func (s *Server) Start(ctx context.Context) (string, error) {
	ln, err := transport.Listen(ctx, s.lcfg)
	if err != nil {
		return "", err
	}
	s.ln = ln
	s.logger.Info("kvblockd listening", "addr", ln.Addr().String())

	go func() {
		<-ctx.Done()
		_ = ln.Close() // unblock Accept on cancel
	}()
	go s.acceptLoop(ln)
	return ln.Addr().String(), nil
}

// acceptLoop spawns a session per accepted connection until the listener closes.
func (s *Server) acceptLoop(ln *transport.Listener) {
	for {
		sess := newSession(s)
		conn, err := ln.Accept(sess, sess)
		if err != nil {
			return // listener closed (ctx cancel or Drain)
		}
		// track reports false if Drain has already started; then this conn is
		// not in the drain set, so close it here rather than leak it.
		if !s.track(conn) {
			_ = conn.Close()
			return
		}
		sess.bind(conn)
		go s.reapOnClose(conn)
	}
}

// Addr returns the bound listen address (":0" resolves after Serve starts).
func (s *Server) Addr() string {
	if s.ln == nil {
		return s.cfg.ListenAddr
	}
	return s.ln.Addr().String()
}

// track adds c to the live set unless Drain has begun (checked in the same
// critical section, so no conn can slip past Drain's snapshot). Returns false
// if draining.
func (s *Server) track(c *transport.Conn) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.draining {
		return false
	}
	s.conns[c] = struct{}{}
	return true
}

func (s *Server) untrack(c *transport.Conn) {
	s.mu.Lock()
	delete(s.conns, c)
	s.mu.Unlock()
}

// reapOnClose removes a connection from the tracking set once it is fully done.
func (s *Server) reapOnClose(c *transport.Conn) {
	<-c.Done()
	s.untrack(c)
}

// Drain stops accepting and closes every live connection, waiting up to
// timeout for them to finish. It closes the listener first so no new
// connections arrive.
func (s *Server) Drain(ctx context.Context) {
	if s.ln != nil {
		_ = s.ln.Close()
	}
	s.mu.Lock()
	s.draining = true // acceptLoop's track() now refuses new conns → none escape the snapshot
	conns := make([]*transport.Conn, 0, len(s.conns))
	for c := range s.conns {
		conns = append(conns, c)
	}
	s.mu.Unlock()

	for _, c := range conns {
		_ = c.Close()
	}
	for _, c := range conns {
		select {
		case <-c.Done():
		case <-ctx.Done():
			return
		}
	}
}
