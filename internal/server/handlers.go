package server

import (
	"encoding/json"
	"net"
	"strconv"

	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/transport"
)

// netBuffers wraps a single body slice as net.Buffers (nil → empty).
func netBuffers(body []byte) net.Buffers {
	if body == nil {
		return nil
	}
	return net.Buffers{body}
}

// fatalHello answers a rejected HELLO with a HELLO response (opcode echoed,
// request_id echoed, F_FATAL) then closes — so a client correlating by
// request_id or opcode sees the specific status (§3.1), rather than the generic
// opcode-0 report c.Abort emits for header-level violations.
func (s *session) fatalHello(c *transport.Conn, h protocol.Header, status protocol.Status) {
	body := protocol.AppendPreamble(make([]byte, 0, protocol.PreambleSize), status, 0)
	resp := protocol.Header{
		Opcode:      protocol.OpHello,
		Flags:       protocol.FlagResp | protocol.FlagFatal,
		NamespaceID: h.NamespaceID,
		RequestID:   h.RequestID,
	}
	_ = c.WriteFrames(resp, netBuffers(body), nil)
	_ = c.Close()
}

// handleHello authenticates, negotiates limits/features, and installs them on
// the connection (§3.1). A non-OK outcome is a HELLO response with F_FATAL.
func (s *session) handleHello(c *transport.Conn, h protocol.Header, body []byte) {
	req, err := protocol.DecodeHelloReq(body)
	s.consume(c, h, body)
	if err != nil {
		// Valid header, unparseable body → ERR_MALFORMED (not FATAL_PROTOCOL,
		// which is a header CRC/magic/version violation, §9).
		s.fatalHello(c, h, protocol.StatusErrMalformed)
		return
	}
	id, ok := s.srv.ns.Authenticate(req.Namespace, req.Token)
	if !ok {
		// Bad token / unknown namespace collapse to ERR_AUTH_FAILED
		// deliberately (no namespace-enumeration oracle; see namespaces.go).
		s.fatalHello(c, h, protocol.StatusErrAuthFailed)
		return
	}

	s.ns = id
	s.feats = protocol.IntersectFeatures(req.Features, protocol.ServerFeatures)
	s.limits = protocol.NegotiateLimits(s.srv.cfg.WireLimits(), req.MaxBatchKeys, req.MaxFrameLen)
	c.SetLimits(s.limits)
	s.authed = true

	resp := protocol.HelloResp{
		Proto:           protocol.Version1,
		Features:        s.feats,
		Limits:          s.limits,
		NamespaceID:     id,
		LeaseDefaultMS:  s.srv.cfg.LeaseDefaultMS,
		LeaseMaxMS:      s.srv.cfg.LeaseMaxMS,
		StreamTimeoutMS: s.srv.cfg.StreamTimeoutMS,
		ServerName:      "kvblockd",
	}
	out := protocol.AppendHelloResp(make([]byte, 0, 128), resp)
	s.writeResp(c, h, out)
}

// decodeKeys decodes the shared key-list body into the reusable scratch,
// returning the keys (aliasing scratch) and the aux u32 (ttl_ms for
// TOUCH_LEASE; reserved-zero elsewhere). On a body error it consumes the
// frame, answers the mapped status, and returns ok=false.
func (s *session) decodeKeys(c *transport.Conn, h protocol.Header, body []byte) (aux uint32, keys [][32]byte, ok bool) {
	aux, keys, err := protocol.DecodeKeyList(body, s.limits.MaxBatchKeys, s.keyScratch[:0])
	s.keyScratch = keys[:0] // keep the grown backing array for the next batch
	s.consume(c, h, body)
	if err != nil {
		s.respondStatus(c, h, protocol.ErrorStatus(err))
		return 0, nil, false
	}
	return aux, keys, true
}

// handleBatchExists serves the consecutive-prefix probe from the store index
// (§3.2). The bitmap is included only when FEAT_EXISTS_BITMAP was negotiated.
func (s *session) handleBatchExists(c *transport.Conn, h protocol.Header, body []byte) {
	_, keys, ok := s.decodeKeys(c, h, body)
	if !ok {
		return
	}
	withBitmap := s.feats&protocol.FeatExistsBitmap != 0
	nConsec, perKey := s.srv.store.ExistsPrefix(s.ns, keys, withBitmap)
	out := protocol.AppendExistsResp(make([]byte, 0, protocol.PreambleSize+8+len(perKey)+8),
		uint32(len(keys)), nConsec, perKey) //nolint:gosec // G115: len(keys) capped by max_batch_keys
	s.writeResp(c, h, out)
}

// handleBatchGet returns descriptors + concatenated payloads (§3.3, §12). Each
// response frame is ONE writev — iov[0] = the header region, iov[1..] = store
// block bytes referenced directly — and the response is split into frames no
// larger than the negotiated max_frame_len, all but the last carrying F_MORE.
// A single block always fits a frame (max_blob_len ≤ max_frame_len, clamped at
// negotiation), so greedy packing never wedges.
func (s *session) handleBatchGet(c *transport.Conn, h protocol.Header, body []byte) {
	_, keys, ok := s.decodeKeys(c, h, body)
	if !ok {
		return
	}
	// Zero-copy stores hand out per-block release callbacks that must fire
	// after the kernel accepts THIS frame's writev (the §12 hold-until-written
	// rule); heap stores (ramstub) don't implement refGetter and release nil.
	// Tiered stores additionally report the serving tier + a per-key status
	// (NVMe saturation → ERR_BUSY descriptor).
	tg, hasTier := s.srv.store.(tierRefGetter)
	rg, hasRefs := s.srv.store.(refGetter)

	maxFrame := int(s.limits.MaxFrameLen)
	total := len(keys)
	if total == 0 {
		// §3.2 note: an empty batch still gets a well-formed count=0
		// response — silence would wedge the request id forever.
		region := protocol.AppendGetRespHeader(make([]byte, 0, protocol.GetRespHeaderSize(0)),
			protocol.StatusOK, 0, 0, nil)
		s.writeRespFlags(c, h, protocol.FlagResp, net.Buffers{region}, nil)
		return
	}
	var nMisses, nBusy int
	var dramHits, nvmeHits, dramBytes, nvmeBytes int
	for i := 0; i < total; {
		// descScratch is reused across frames and requests: AppendGetRespHeader
		// (below) serializes it into the region bytes synchronously, so it is
		// dead before bufs/region are handed to the async writer — safe to
		// recycle, unlike region/bufs themselves.
		descs := s.descScratch[:0]
		bufs := net.Buffers{nil} // iov[0] placeholder for the header region
		// releases is FRESH per frame (never scratch): the combined closure is
		// captured by the async writer and fires after the flush.
		var releases []func()
		payloadBytes := 0
		j := i
		for j < total {
			var data []byte
			var sum uint64
			var hit, busy bool
			var rel func()
			tier := "dram"
			switch {
			case hasTier:
				var st protocol.Status
				data, sum, rel, tier, st = tg.GetRefTier(s.ns, keys[j])
				hit = st == protocol.StatusOK
				busy = st == protocol.StatusErrBusy
			case hasRefs:
				data, sum, rel, hit = rg.GetRef(s.ns, keys[j])
			default:
				data, sum, hit = s.srv.store.Get(s.ns, keys[j])
			}
			blen := 0
			var d protocol.Desc
			switch {
			case hit:
				blen = len(data)
				d = protocol.Desc{Status: protocol.StatusOK, Len: uint32(blen), XXH3: sum} //nolint:gosec // G115: block sizes << MaxUint32
			case busy:
				// §3 descriptors carry per-key outcomes: a saturated device
				// reader answers ERR_BUSY for THIS key (retryable), payload-
				// free — the batch and the connection sail on.
				d = protocol.Desc{Status: protocol.StatusErrBusy}
			default:
				d = protocol.Desc{Status: protocol.StatusNotFound}
			}
			// Would adding this descriptor+payload overflow the frame? Always
			// keep at least one descriptor per frame so progress is guaranteed.
			projected := protocol.GetRespHeaderSize(len(descs)+1) + payloadBytes + blen
			if len(descs) > 0 && projected > maxFrame {
				if rel != nil {
					rel() // block not packed into this frame; re-fetched next frame
				}
				break
			}
			descs = append(descs, d)
			switch {
			case hit:
				if blen > 0 { // a zero-length block is descriptor-only (§3.4)
					bufs = append(bufs, data)
					payloadBytes += blen
				}
				if tier == "nvme" {
					nvmeHits++
					nvmeBytes += blen
				} else {
					dramHits++
					dramBytes += blen
				}
				if rel != nil {
					releases = append(releases, rel)
				}
			case busy:
				nBusy++
			default:
				nMisses++
				if rel != nil {
					rel() // defensive: a miss never hands out a release
				}
			}
			j++
		}
		region := protocol.AppendGetRespHeader(make([]byte, 0, protocol.GetRespHeaderSize(len(descs))),
			protocol.StatusOK, uint32(i), uint32(total), descs) //nolint:gosec // G115: indices capped by max_batch_keys
		s.descScratch = descs[:0] // keep the grown backing array for the next frame
		bufs[0] = region
		flags := protocol.FlagResp
		if j < total {
			flags |= protocol.FlagMore
		}
		// Fold this frame's per-block releases into ONE closure for the writer.
		var frameRelease func()
		if len(releases) > 0 {
			rels := releases
			frameRelease = func() {
				for _, r := range rels {
					r()
				}
			}
		}
		s.writeRespFlags(c, h, flags, bufs, frameRelease)
		i = j
	}
	if r := s.srv.rec; r != nil {
		r.GetResult(s.ns, "dram", dramHits, nMisses, dramBytes)
		if nvmeHits > 0 {
			r.GetResult(s.ns, "nvme", nvmeHits, 0, nvmeBytes)
		}
		if nBusy > 0 {
			r.GetBusy(s.ns, nBusy)
		}
	}
}

// writeRespFlags queues a response frame with explicit flags (the F_MORE
// split path); echoes opcode/namespace/request_id. release (may be nil) fires
// once after the kernel accepts the frame — the arena-refcount drop.
func (s *session) writeRespFlags(c *transport.Conn, h protocol.Header, flags uint16, bufs net.Buffers, release func()) {
	resp := protocol.Header{
		Opcode:      h.Opcode,
		Flags:       flags,
		NamespaceID: s.respNS(h),
		RequestID:   h.RequestID,
	}
	_ = c.WriteFrames(resp, bufs, release)
}

// statusBuf returns the per-session per-key-status scratch sized to n. The
// slice is consumed synchronously by AppendKeyStatusResp (copied into the
// response body) before the response is queued, so recycling it across the
// key-status verbs is safe.
func (s *session) statusBuf(n int) []protocol.Status {
	if cap(s.statusScratch) < n {
		s.statusScratch = make([]protocol.Status, n)
	}
	return s.statusScratch[:n]
}

// handleKeyStatusVerb serves TOUCH_LEASE and PIN (§3.5/§3.6). A lifecycle-
// aware store (the DRAM tier) gets the real semantics — sub-op dispatch,
// ttl_ms from the key-list aux, ERR_PIN_QUOTA; a store without the
// lifecycler extension (ramstub) answers the pre-tier metadata ack
// (present→OK, absent→NOT_FOUND).
func (s *session) handleKeyStatusVerb(c *transport.Conn, h protocol.Header, body []byte) {
	if protocol.SubOp(h.Flags) > protocol.Unpin { // TOUCH/LEASE/RELEASE and PIN/UNPIN sub-ops are 0..2
		s.consume(c, h, body)
		s.respondStatus(c, h, protocol.StatusErrUnsupported)
		return
	}
	ttlMS, keys, ok := s.decodeKeys(c, h, body)
	if !ok {
		return
	}
	lc, hasLifecycle := s.srv.store.(lifecycler)
	sub := protocol.SubOp(h.Flags)
	perKey := s.statusBuf(len(keys))
	for i, k := range keys {
		switch {
		case !hasLifecycle:
			// Pre-tier metadata ack: present→OK, absent→NOT_FOUND.
			if s.srv.store.Contains(s.ns, k) {
				perKey[i] = protocol.StatusOK
			} else {
				perKey[i] = protocol.StatusNotFound
			}
		case h.Opcode == protocol.OpTouchLease:
			perKey[i] = lc.TouchLease(s.ns, k, sub, ttlMS)
		default: // OpPin
			perKey[i] = lc.PinOp(s.ns, k, sub)
		}
	}
	out := protocol.AppendKeyStatusResp(make([]byte, 0, protocol.PreambleSize+len(perKey)+8), perKey)
	s.writeResp(c, h, out)
}

// handleDelete removes each key (§3.7). The store gates each delete
// (ERR_LEASED / ERR_PINNED on the DRAM tier; ramstub is ungated) and
// F_FORCE overrides leases and soft pins — never a hard pin.
func (s *session) handleDelete(c *transport.Conn, h protocol.Header, body []byte) {
	_, keys, ok := s.decodeKeys(c, h, body)
	if !ok {
		return
	}
	force := h.Flags&protocol.FlagForce != 0
	perKey := s.statusBuf(len(keys))
	for i, k := range keys {
		perKey[i] = s.srv.store.Delete(s.ns, k, force)
	}
	out := protocol.AppendKeyStatusResp(make([]byte, 0, protocol.PreambleSize+len(perKey)+8), perKey)
	s.writeResp(c, h, out)
}

// handleStats returns the store's JSON stats document (§3.8), scoped to the
// authenticated namespace where the document carries per-tenant detail.
func (s *session) handleStats(c *transport.Conn, h protocol.Header, body []byte) {
	if _, err := protocol.DecodeStatsReq(body); err != nil {
		s.consume(c, h, body)
		s.respondStatus(c, h, protocol.ErrorStatus(err))
		return
	}
	s.consume(c, h, body)
	doc := scopeStatsToNS(s.srv.store.Stats(), s.ns)
	out := protocol.AppendPreamble(make([]byte, 0, protocol.PreambleSize+len(doc)), protocol.StatusOK, uint32(len(doc))) //nolint:gosec // G115: stats doc is small
	out = append(out, doc...)
	s.writeResp(c, h, out)
}

// scopeStatsToNS strips other tenants' entries from the store's fleet-wide
// stats document before it crosses the wire: pinned_bytes is keyed by
// namespace id, and any bearer token reading every tenant's pin pressure is
// a cross-tenant information leak. The admin API and the scrape keep the
// fleet view — they sit behind the shell-trust boundary, not a token.
func scopeStatsToNS(doc []byte, ns uint32) []byte {
	var m map[string]any
	if err := json.Unmarshal(doc, &m); err != nil {
		return doc // malformed docs pass through — §3.8 is best-effort JSON
	}
	pinned, ok := m["pinned_bytes"].(map[string]any)
	if !ok {
		return doc // no per-tenant detail (ramstub, error doc) — nothing to scope
	}
	own := strconv.FormatUint(uint64(ns), 10)
	scoped := map[string]any{}
	if v, ok := pinned[own]; ok {
		scoped[own] = v
	}
	m["pinned_bytes"] = scoped
	out, err := json.Marshal(m)
	if err != nil {
		return doc
	}
	return out
}
