package server

import (
	"net"

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
// returning the keys (aliasing scratch) and the aux u32. On a body error it
// consumes the frame, answers the mapped status, and returns ok=false.
func (s *session) decodeKeys(c *transport.Conn, h protocol.Header, body []byte) (keys [][32]byte, ok bool) {
	// aux (the second u32) is ttl_ms for TOUCH_LEASE; the current metadata
	// no-op ignores it, so it is discarded here.
	_, keys, err := protocol.DecodeKeyList(body, s.limits.MaxBatchKeys, s.keyScratch[:0])
	s.keyScratch = keys[:0] // keep the grown backing array for the next batch
	s.consume(c, h, body)
	if err != nil {
		s.respondStatus(c, h, protocol.ErrorStatus(err))
		return nil, false
	}
	return keys, true
}

// handleBatchExists serves the consecutive-prefix probe from the store index
// (§3.2). The bitmap is included only when FEAT_EXISTS_BITMAP was negotiated.
func (s *session) handleBatchExists(c *transport.Conn, h protocol.Header, body []byte) {
	keys, ok := s.decodeKeys(c, h, body)
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
	keys, ok := s.decodeKeys(c, h, body)
	if !ok {
		return
	}
	maxFrame := int(s.limits.MaxFrameLen)
	total := len(keys)
	for i := 0; i < total; {
		descs := make([]protocol.Desc, 0, total-i)
		bufs := net.Buffers{nil} // iov[0] placeholder for the header region
		payloadBytes := 0
		j := i
		for j < total {
			data, sum, hit := s.srv.store.Get(s.ns, keys[j])
			blen := 0
			var d protocol.Desc
			if hit {
				blen = len(data)
				d = protocol.Desc{Status: protocol.StatusOK, Len: uint32(blen), XXH3: sum} //nolint:gosec // G115: block sizes << MaxUint32
			} else {
				d = protocol.Desc{Status: protocol.StatusNotFound}
			}
			// Would adding this descriptor+payload overflow the frame? Always
			// keep at least one descriptor per frame so progress is guaranteed.
			projected := protocol.GetRespHeaderSize(len(descs)+1) + payloadBytes + blen
			if len(descs) > 0 && projected > maxFrame {
				break
			}
			descs = append(descs, d)
			if hit {
				bufs = append(bufs, data)
				payloadBytes += blen
			}
			j++
		}
		region := protocol.AppendGetRespHeader(make([]byte, 0, protocol.GetRespHeaderSize(len(descs))),
			protocol.StatusOK, uint32(i), uint32(total), descs) //nolint:gosec // G115: indices capped by max_batch_keys
		bufs[0] = region
		flags := protocol.FlagResp
		if j < total {
			flags |= protocol.FlagMore
		}
		s.writeRespFlags(c, h, flags, bufs)
		i = j
	}
}

// writeRespFlags queues a response frame with explicit flags (the F_MORE split
// path); echoes opcode/namespace/request_id.
func (s *session) writeRespFlags(c *transport.Conn, h protocol.Header, flags uint16, bufs net.Buffers) {
	resp := protocol.Header{
		Opcode:      h.Opcode,
		Flags:       flags,
		NamespaceID: h.NamespaceID,
		RequestID:   h.RequestID,
	}
	_ = c.WriteFrames(resp, bufs, nil)
}

// handleKeyStatusVerb serves TOUCH_LEASE and PIN as metadata acks (§3.5/§3.6):
// ramstub has no lease/pin state yet, so every present key is OK and every
// absent key NOT_FOUND. Real semantics arrive with the tiers.
func (s *session) handleKeyStatusVerb(c *transport.Conn, h protocol.Header, body []byte) {
	if protocol.SubOp(h.Flags) > protocol.Unpin { // TOUCH/LEASE/RELEASE and PIN/UNPIN sub-ops are 0..2
		s.consume(c, h, body)
		s.respondStatus(c, h, protocol.StatusErrUnsupported)
		return
	}
	keys, ok := s.decodeKeys(c, h, body)
	if !ok {
		return
	}
	perKey := make([]protocol.Status, len(keys))
	for i, k := range keys {
		if s.srv.store.Contains(s.ns, k) {
			perKey[i] = protocol.StatusOK
		} else {
			perKey[i] = protocol.StatusNotFound
		}
	}
	out := protocol.AppendKeyStatusResp(make([]byte, 0, protocol.PreambleSize+len(perKey)+8), perKey)
	s.writeResp(c, h, out)
}

// handleDelete removes each key (§3.7). ramstub has no lease/pin protection, so
// F_FORCE is a no-op and every present key deletes.
func (s *session) handleDelete(c *transport.Conn, h protocol.Header, body []byte) {
	keys, ok := s.decodeKeys(c, h, body)
	if !ok {
		return
	}
	force := h.Flags&protocol.FlagForce != 0
	perKey := make([]protocol.Status, len(keys))
	for i, k := range keys {
		perKey[i] = s.srv.store.Delete(s.ns, k, force)
	}
	out := protocol.AppendKeyStatusResp(make([]byte, 0, protocol.PreambleSize+len(perKey)+8), perKey)
	s.writeResp(c, h, out)
}

// handleStats returns the store's JSON stats document (§3.8).
func (s *session) handleStats(c *transport.Conn, h protocol.Header, body []byte) {
	if _, err := protocol.DecodeStatsReq(body); err != nil {
		s.consume(c, h, body)
		s.respondStatus(c, h, protocol.ErrorStatus(err))
		return
	}
	s.consume(c, h, body)
	doc := s.srv.store.Stats()
	out := protocol.AppendPreamble(make([]byte, 0, protocol.PreambleSize+len(doc)), protocol.StatusOK, uint32(len(doc))) //nolint:gosec // G115: stats doc is small
	out = append(out, doc...)
	s.writeResp(c, h, out)
}
