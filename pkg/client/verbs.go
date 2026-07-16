package client

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/protocol"
)

// BatchExists probes a prefix chain and returns the number of consecutive hits
// from position 0 (the scheduler's load-vs-recompute number) and, when the
// server negotiated FEAT_EXISTS_BITMAP, the per-key statuses. Keys beyond the
// negotiated batch cap are rejected by the caller's responsibility to tile;
// this subset issues one batch.
func (c *Client) BatchExists(ctx context.Context, keys [][32]byte) (nConsecutive int, perKey []protocol.Status, err error) {
	if uint32(len(keys)) > c.limits.MaxBatchKeys { //nolint:gosec // G115: len compared to a u32 cap
		return 0, nil, fmt.Errorf("client: %d keys exceeds negotiated max_batch_keys %d", len(keys), c.limits.MaxBatchKeys)
	}
	cn, err := c.get(ctx)
	if err != nil {
		return 0, nil, err
	}
	defer func() { c.release(cn, err) }()

	body := protocol.AppendKeyList(cn.reqBuf(), 0, keys)
	cn.keepReq(body)
	if err = cn.writeFrame(protocol.OpBatchExists, 0, [32]byte{}, cn.id(), body); err != nil {
		return 0, nil, err
	}
	respBody, err := cn.readFrame()
	if err != nil {
		return 0, nil, err
	}
	r, err := protocol.DecodeExistsResp(respBody, c.feats&protocol.FeatExistsBitmap != 0)
	if err != nil {
		return 0, nil, err
	}
	if r.Status != protocol.StatusOK && r.Status != protocol.StatusOKExists {
		return 0, nil, &StatusError{Op: protocol.OpBatchExists, Status: r.Status}
	}
	if r.PerKey != nil {
		perKey = make([]protocol.Status, len(r.PerKey))
		for i, b := range r.PerKey {
			perKey[i] = protocol.Status(b)
		}
	}
	return int(r.NConsecutive), perKey, nil
}

// BatchGet fetches keys, reading each hit's bytes into the caller-provided
// buffer at the same index (into[i] must be sized to the block; a NOT_FOUND
// leaves into[i] untouched). It returns the per-key status slice. Payloads are
// streamed straight from the socket into into[i] — no intermediate copy of the
// block bytes.
func (c *Client) BatchGet(ctx context.Context, keys [][32]byte, into [][]byte) (statuses []protocol.Status, err error) {
	if len(keys) == 0 {
		return nil, nil // no keys → no request; a zero-key GET would hang on the response read
	}
	if len(into) != len(keys) {
		return nil, fmt.Errorf("client: into has %d slots, want %d", len(into), len(keys))
	}
	if uint32(len(keys)) > c.limits.MaxBatchKeys { //nolint:gosec // G115: len vs u32 cap
		return nil, fmt.Errorf("client: %d keys exceeds negotiated max_batch_keys %d", len(keys), c.limits.MaxBatchKeys)
	}
	cn, err := c.get(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { c.release(cn, err) }()

	body := protocol.AppendKeyList(cn.reqBuf(), 0, keys)
	cn.keepReq(body)
	if err = cn.writeFrame(protocol.OpBatchGet, 0, [32]byte{}, cn.id(), body); err != nil {
		return nil, err
	}
	statuses, err = cn.readGetInto(keys, into, !c.opts.SkipVerify)
	return statuses, err
}

// readGetInto streams one or more BATCH_GET response frames into into[i].
// A non-OK batch status is exactly the 8-byte preamble (§3), so the preamble
// MUST be read and inspected BEFORE reading any index/descriptor bytes — reading
// ahead would block forever on an ERR_BUSY/ERR_TOO_LARGE reply. F_MORE-split
// responses are reassembled: each frame covers [first_index, first_index+count)
// and all but the last carry F_MORE.
//
// A non-StatusError return means the connection is desynchronized (bytes left
// on the wire) and the caller MUST evict it; a *StatusError leaves the stream
// in sync.
func (cn *conn) readGetInto(keys [][32]byte, into [][]byte, verify bool) ([]protocol.Status, error) {
	statuses := make([]protocol.Status, len(keys))
	got := 0
	// Checksum verification runs on a sidecar goroutine so hashing block i
	// overlaps reading block i+1 — serialized read-then-hash on one goroutine
	// halves effective GET throughput (the socket sits idle during every hash).
	// The sidecar only reads into[slot] contents the reader has finished with
	// and never touches conn state; verifyWait joins it before return.
	var verifyCh chan verifyJob
	verifyErr := make(chan error, 1)
	if verify {
		verifyCh = make(chan verifyJob, 32)
		go func() {
			var err error
			for j := range verifyCh {
				if err == nil && xxh3.Hash(into[j.slot]) != j.sum {
					err = fmt.Errorf("client: key %d checksum mismatch (corruption)", j.slot)
				}
			}
			verifyErr <- err
		}()
	}
	verifyWait := func() error {
		if verifyCh == nil {
			return nil
		}
		close(verifyCh)
		return <-verifyErr
	}
	// fail joins the verifier (goroutine hygiene) before returning the read-path
	// error. A checksum mismatch outranks it: a *StatusError would re-pool the
	// connection as clean, and corruption must evict instead.
	fail := func(err error) ([]protocol.Status, error) {
		if verr := verifyWait(); verr != nil {
			return nil, verr
		}
		return nil, err
	}
	for {
		h, err := cn.nextHeader() // skips unsolicited NOP/CREDIT
		if err != nil {
			return fail(err)
		}
		// A non-OK batch response is exactly the 8-byte preamble (§3); the
		// header's payload_len says which shape arrived, so an OK frame's
		// preamble+index (16 B) is read in one syscall.
		if h.PayloadLen == protocol.PreambleSize {
			pre, err := cn.readN(protocol.PreambleSize)
			if err != nil {
				return fail(err)
			}
			return fail(&StatusError{Op: protocol.OpBatchGet, Status: protocol.Status(pre[0])})
		}
		pre, err := cn.readN(protocol.PreambleSize + 8)
		if err != nil {
			return fail(err)
		}
		status := protocol.Status(pre[0])
		count := binary.LittleEndian.Uint32(pre[4:])
		if status != protocol.StatusOK && status != protocol.StatusOKExists {
			// Defensive: a non-OK status must be preamble-only (§3); a larger
			// body is a protocol violation with unread bytes on the wire, so
			// return a non-StatusError to make release() evict the connection.
			return fail(fmt.Errorf("client: non-OK GET status %s with %d-byte body (protocol violation)", status, h.PayloadLen))
		}
		firstIndex := binary.LittleEndian.Uint32(pre[8:])
		totalKeys := binary.LittleEndian.Uint32(pre[12:])
		if int(totalKeys) != len(keys) || int(firstIndex)+int(count) > len(keys) {
			return fail(fmt.Errorf("client: GET frame [%d,+%d) inconsistent with %d requested keys", firstIndex, count, len(keys)))
		}
		descRegion, err := cn.readN(int(count) * protocol.DescSize)
		if err != nil {
			return fail(err)
		}
		for i := 0; i < int(count); i++ {
			slot := int(firstIndex) + i
			d := protocol.GetDesc(descRegion[i*protocol.DescSize:])
			statuses[slot] = d.Status
			if d.Status != protocol.StatusOK && d.Status != protocol.StatusOKExists {
				continue
			}
			if into[slot] == nil || uint32(len(into[slot])) != d.Len { //nolint:gosec // G115: len vs u32
				into[slot] = make([]byte, d.Len)
			}
			if _, err := io.ReadFull(cn.nc, into[slot]); err != nil {
				return fail(err)
			}
			if verifyCh != nil {
				verifyCh <- verifyJob{slot: slot, sum: d.XXH3}
			}
		}
		got += int(count)
		if h.Flags&protocol.FlagMore == 0 {
			break
		}
	}
	if err := verifyWait(); err != nil {
		return nil, err
	}
	if got != len(keys) {
		return nil, fmt.Errorf("client: GET returned %d of %d keys", got, len(keys))
	}
	return statuses, nil
}

// verifyJob asks the per-call verifier goroutine to xxh3-check into[slot]
// against the descriptor checksum the server sent.
type verifyJob struct {
	slot int
	sum  uint64
}

// Put streams a block via BEGIN → CHUNK(s) → COMMIT (§5). It computes the
// authoritative xxh3 while chunking. A write-once idempotent hit (OK_EXISTS at
// BEGIN) returns nil without sending the body.
func (c *Client) Put(ctx context.Context, key [32]byte, data []byte) (err error) {
	if uint32(len(data)) > c.limits.MaxBlobLen { //nolint:gosec // G115: len vs u32 cap
		return fmt.Errorf("client: blob %d exceeds negotiated max_blob_len %d", len(data), c.limits.MaxBlobLen)
	}
	cn, err := c.get(ctx)
	if err != nil {
		return err
	}
	defer func() { c.release(cn, err) }()
	reqID := cn.id()

	sum := xxh3.Hash(data)
	begin := protocol.AppendPutBegin(nil, protocol.PutBeginBody{
		TotalLen: uint32(len(data)), //nolint:gosec // G115: bounded by max_blob_len above
		XXH3Hint: sum,
	})
	if err = cn.writeFrame(protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutBegin), key, reqID, begin); err != nil {
		return err
	}
	beginResp, err := cn.readFrame()
	if err != nil {
		return err
	}
	bp, err := protocol.DecodePreamble(beginResp)
	if err != nil {
		return err
	}
	switch bp.Status {
	case protocol.StatusOKExists:
		return nil // already sealed; nothing to send
	case protocol.StatusOK:
		// proceed
	default:
		return &StatusError{Op: protocol.OpPutStream, Status: bp.Status}
	}

	// One CHUNK for the whole block for now (striping is a later refinement);
	// keep each chunk within the negotiated frame cap.
	chunk := c.limits.MaxFrameLen
	for off := 0; off < len(data); off += int(chunk) {
		end := off + int(chunk)
		if end > len(data) {
			end = len(data)
		}
		if err = cn.writeFrame(protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutChunk), key, reqID, data[off:end]); err != nil {
			return err
		}
	}

	commit := protocol.AppendPutCommit(nil, sum)
	if err = cn.writeFrame(protocol.OpPutStream, protocol.WithSubOp(0, protocol.PutCommit), key, reqID, commit); err != nil {
		return err
	}
	commitResp, err := cn.readFrame()
	if err != nil {
		return err
	}
	cp, err := protocol.DecodePreamble(commitResp)
	if err != nil {
		return err
	}
	if cp.Status != protocol.StatusOK && cp.Status != protocol.StatusOKExists {
		return &StatusError{Op: protocol.OpPutStream, Status: cp.Status}
	}
	return nil
}

// Delete removes keys (§3.7). force sets F_FORCE (evict even leased/pinned —
// lease/pin gating arrives with the tiers). Returns the per-key statuses.
func (c *Client) Delete(ctx context.Context, keys [][32]byte, force bool) (perKey []protocol.Status, err error) {
	if uint32(len(keys)) > c.limits.MaxBatchKeys { //nolint:gosec // G115: len vs u32 cap
		return nil, fmt.Errorf("client: %d keys exceeds negotiated max_batch_keys %d", len(keys), c.limits.MaxBatchKeys)
	}
	cn, err := c.get(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { c.release(cn, err) }()

	var flags uint16
	if force {
		flags |= protocol.FlagForce
	}
	delBody := protocol.AppendKeyList(cn.reqBuf(), 0, keys)
	cn.keepReq(delBody)
	if err = cn.writeFrame(protocol.OpDelete, flags, [32]byte{}, cn.id(), delBody); err != nil {
		return nil, err
	}
	body, err := cn.readFrame()
	if err != nil {
		return nil, err
	}
	p, raw, err := protocol.DecodeKeyStatusResp(body)
	if err != nil {
		return nil, err
	}
	if p.Status != protocol.StatusOK {
		return nil, &StatusError{Op: protocol.OpDelete, Status: p.Status}
	}
	if len(raw) != len(keys) {
		return nil, fmt.Errorf("client: DELETE returned %d statuses for %d keys", len(raw), len(keys))
	}
	perKey = make([]protocol.Status, len(raw))
	for i, s := range raw {
		perKey[i] = protocol.Status(s)
	}
	return perKey, nil
}

// Stats fetches the server's JSON stats document (§3.8).
func (c *Client) Stats(ctx context.Context) (doc []byte, err error) {
	cn, err := c.get(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { c.release(cn, err) }()
	if err = cn.writeFrame(protocol.OpStats, 0, [32]byte{}, cn.id(), protocol.AppendStatsReq(nil, 0)); err != nil {
		return nil, err
	}
	body, err := cn.readFrame()
	if err != nil {
		return nil, err
	}
	p, err := protocol.DecodePreamble(body)
	if err != nil {
		return nil, err
	}
	if p.Status != protocol.StatusOK {
		return nil, &StatusError{Op: protocol.OpStats, Status: p.Status}
	}
	if len(body) < protocol.PreambleSize+int(p.Count) {
		return nil, errShortResponse
	}
	// Copy out: body aliases the conn's scratch, but doc bytes go to the caller.
	doc = make([]byte, p.Count)
	copy(doc, body[protocol.PreambleSize:])
	return doc, nil
}
