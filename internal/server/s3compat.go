// s3compat.go is the S3-compatibility endpoint: a minimal S3 REST subset
// (PutObject, GetObject with Range, HeadObject) on its OWN HTTP listener
// (config s3compat_addr; "" = off, the default), so NIXL's `obj` storage
// plugin and vLLM's `obj` tier reach kvblockd with ZERO plugin code via
// endpoint_override (the week-11 / SPEC 5 zero-code path). It is a
// compatibility surface, not the performance path — the KVB1 data plane is;
// every GET here copies block bytes to the heap before they touch net/http,
// so no arena view ever escapes into an HTTP writer.
//
// Bucket = namespace. The bucket name is the namespace name exactly as
// HELLO's namespace field, and auth reuses the SAME tenant registry and
// constant-time token compare (Namespaces.Authenticate) as the wire plane.
// A wrong token and an unknown bucket deliberately COLLAPSE to one answer —
// 403 AccessDenied — mirroring the wire's ERR_AUTH_FAILED collapse (no
// bucket-enumeration oracle). Inside an authenticated bucket, a key that
// exists only in another namespace is plainly 404 NoSuchKey: cross-tenant
// isolation is NOT_FOUND semantics (disjoint keyspaces), never a FORBIDDEN
// that would leak that the key exists elsewhere.
//
// Why bearer tokens, not SigV4 (the C-11 verdict, a documented divergence):
//
//   - kvblockd tenant auth is a static per-namespace secret compared in
//     constant time; on the data plane it rides cleartext inside HELLO. An
//     `Authorization: Bearer <token>` header is the same secret under the
//     same channel-trust model (loopback / private network, or TLS
//     terminated in front). Verifying SigV4 would mean holding that same
//     shared secret server-side PLUS canonical-request reconstruction and
//     the HMAC derivation chain — strictly more code for zero added secrecy
//     at this trust boundary.
//   - What NIXL's obj backend needs (C-11): it is built on aws-sdk-cpp,
//     which ALWAYS SigV4-signs — no client mode emits a Bearer header. So
//     this endpoint ALSO accepts a SigV4-shaped Authorization header and
//     treats the access-key-id (the first segment of `Credential=`) as the
//     bearer token; the signature itself is NOT verified (same posture as
//     Bearer — the id IS the secret). Zero-code invocation:
//     AWS_ACCESS_KEY_ID=<namespace token>, any non-empty secret key,
//     endpoint_override=<this listener>, path-style addressing. The live
//     round-trip against a real NIXL build is the integration-rig step
//     recorded in docs/INTEGRATIONS.md — a unit wall cannot assert it.
//
// Object-key rule: the S3 object key IS the hex encoding of the 32-byte
// block key — exactly 64 hex characters (case-insensitive on decode;
// canonical form lowercase, i.e. hex.EncodeToString / Python bytes.hex(),
// the rendering the hash-parity oracle uses), decoded straight into the
// KVB1 key[32]. The key itself stays what PROTOCOL.md says it is: a BLAKE3
// prefix-chain hash COMPUTED BY THE ADAPTER (pkg/client.WireKey ≡
// python/kvblockd hashing.wire_key) — opaque here; this server never
// derives keys from content. Anything else is 400 InvalidArgument. So
// `PUT /tenant-a/<64 hex>` lands on the same (namespace, key) identity as
// the wire's PUT_STREAM, and the SAME bytes are readable on both planes.
//
// Store surface: PutObject commits through the SAME write-once entry the
// wire's PUT_STREAM COMMIT lands on — server.Store.Put(ns, key, payload,
// xxh3) — and GetObject/HeadObject read via the store's existing optional
// extensions (tierRefGetter → refGetter → Get). Write-once carries over:
// re-PUT of identical bytes is 200 (OK_EXISTS), of different bytes is 409
// (ERR_IMMUTABLE_CONFLICT — the corruption alarm). ETag is the block's
// xxh3_64 (16 hex chars, quoted) — NOT an MD5; clients that validate MD5
// ETags must turn that off.
//
// Deliberately NOT implemented, each answered with a well-formed S3 XML
// error: multipart (501 — NIXL obj writes whole objects below its CRT
// threshold), aws-chunked / STREAMING-* payloads (501 — configure the
// client for plain bodies rather than risk sealing chunk framing into a
// block), and ListBuckets/ListObjects/DeleteObject and every other verb
// (501).

package server

import (
	"context"
	"encoding/hex"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/zeebo/xxh3"

	"github.com/kvstash/kvblockd/internal/config"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/transport"
)

// S3Compat is the S3-subset HTTP handler. It holds no listener state of its
// own — Serve owns the lifecycle — so tests can drive ServeHTTP directly.
// readTimeout/writeTimeout are Serve's http.Server deadlines, derived from
// stream_timeout_ms in NewS3Compat (see there for the mapping).
type S3Compat struct {
	store        Store
	ns           *Namespaces
	maxBlobLen   uint32
	readTimeout  time.Duration
	writeTimeout time.Duration
	logger       *slog.Logger
}

// NewS3Compat builds the handler against the SAME store and tenant table the
// wire server dispatches to. cfg supplies max_blob_len — the identical
// ceiling the wire offers at HELLO; there is no per-connection negotiation
// on HTTP, so the config ceiling applies directly — and stream_timeout_ms,
// from which Serve's deadlines derive exactly as the transport's do: read =
// the stream timeout (BodyReadTimeout's slow-loris intent), write = 2× it
// (transport.StallTimeout, §8 rule 5's zero-drain closure). A zero stream
// timeout (a hand-rolled cfg that skipped Load/Validate) falls back to the
// protocol default rather than serving with deadlines DISABLED — the same
// never-run-unbounded posture as transport.Listen's stall floor.
func NewS3Compat(cfg config.Config, store Store, ns *Namespaces) *S3Compat {
	st := cfg.StreamTimeoutMS
	if st == 0 {
		st = protocol.DefaultStreamTimeoutMS
	}
	return &S3Compat{
		store:        store,
		ns:           ns,
		maxBlobLen:   cfg.MaxBlobLen,
		readTimeout:  time.Duration(st) * time.Millisecond,
		writeTimeout: transport.StallTimeout(st),
		logger:       slog.Default(),
	}
}

// Serve binds addr and serves until ctx is cancelled (mirrors metrics.Serve).
// wait blocks until the server has fully stopped and reports whether shutdown
// was CLEAN — every in-flight handler finished inside the grace period. The
// caller must not tear the store down after a false (the drain-before-Close
// rule: a straggling handler may be mid-Get).
func (h *S3Compat) Serve(ctx context.Context, addr string) (bound string, wait func() bool, err error) {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return "", nil, err
	}
	// The deadlines mirror the data plane's transport.Config intent
	// (transport/listener.go): ReadTimeout plays BodyReadTimeout — the
	// slow-loris guard that keeps a 2-byte dribble from pinning the
	// max_blob_len buffer putObject pre-allocates (HTTP/1 has no per-chunk
	// inactivity timer, so the whole-request deadline is the closest
	// primitive); WriteTimeout plays WriteStallTimeout — a peer that stops
	// reading its GET must not wedge the handler; IdleTimeout plays
	// IdleReadTimeout's 5 minutes for keep-alive connections with no request
	// in flight.
	srv := &http.Server{
		Handler:           h,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       h.readTimeout,
		WriteTimeout:      h.writeTimeout,
		IdleTimeout:       5 * time.Minute,
	}
	serveDone := make(chan struct{})
	go func() {
		defer close(serveDone)
		if serr := srv.Serve(ln); serr != nil && !errors.Is(serr, http.ErrServerClosed) {
			// The compat port dying must never take the data plane with it.
			h.logger.Error("s3compat serve failed", "addr", addr, "err", serr)
		}
	}()
	shutdownDone := make(chan struct{})
	var clean bool
	go func() { //nolint:gosec // G118: shutdown must outlive the cancelled ctx; the fresh timeout context is the point
		defer close(shutdownDone)
		<-ctx.Done()
		// The 3s grace bounds how long in-flight PUT/GET handlers may run.
		shCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		clean = srv.Shutdown(shCtx) == nil
	}()
	wait = func() bool {
		<-serveDone
		<-shutdownDone // orders the `clean` write before the read below
		return clean
	}
	return ln.Addr().String(), wait, nil
}

// ServeHTTP routes one request: capability guards → path split → tenant auth
// → key decode → verb dispatch. Auth runs before ANY per-object answer so an
// unauthenticated caller learns nothing beyond "this endpoint exists".
func (h *S3Compat) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	if q.Has("uploads") || q.Has("uploadId") || q.Has("partNumber") {
		h.writeError(w, r, http.StatusNotImplemented, "NotImplemented",
			"multipart upload is not supported: PUT whole objects (raise the client's multipart threshold above max_blob_len)")
		return
	}
	bucket, objectKey := splitObjectPath(r.URL.Path)
	if bucket == "" {
		// No bucket to authenticate against: a capability answer, not a data
		// answer (ListBuckets and friends live at the service root).
		h.writeError(w, r, http.StatusNotImplemented, "NotImplemented",
			"service-level operations are not supported; address objects as /<namespace>/<64-hex-key>")
		return
	}
	token, ok := tokenFromAuthorization(r.Header.Get("Authorization"))
	if !ok {
		h.writeError(w, r, http.StatusForbidden, "AccessDenied",
			"missing or unusable Authorization: send `Bearer <token>`, or SigV4 with the namespace token as the access-key-id")
		return
	}
	nsID, ok := h.ns.Authenticate(bucket, token)
	if !ok {
		// Wrong token and unknown bucket collapse to ONE answer — the wire's
		// ERR_AUTH_FAILED posture (no bucket-enumeration oracle).
		h.writeError(w, r, http.StatusForbidden, "AccessDenied", "access denied")
		return
	}
	if objectKey == "" {
		h.writeError(w, r, http.StatusNotImplemented, "NotImplemented",
			"bucket-level operations (list/create/delete bucket) are not supported")
		return
	}
	key, ok := decodeObjectKey(objectKey)
	if !ok {
		h.writeError(w, r, http.StatusBadRequest, "InvalidArgument",
			"object key must be exactly 64 hex characters — the hex encoding of the 32-byte block key")
		return
	}
	switch r.Method {
	case http.MethodPut:
		h.putObject(w, r, nsID, key)
	case http.MethodGet:
		h.getObject(w, r, nsID, key, false)
	case http.MethodHead:
		h.getObject(w, r, nsID, key, true)
	default:
		h.writeError(w, r, http.StatusNotImplemented, "NotImplemented",
			"only PutObject, GetObject, and HeadObject are supported")
	}
}

// putObject is whole-object PutObject → the write-once store commit. The
// payload buffer is freshly allocated per request and handed to Store.Put,
// which takes ownership (the same contract the PUT_STREAM commit path uses).
func (h *S3Compat) putObject(w http.ResponseWriter, r *http.Request, ns uint32, key [32]byte) {
	if strings.Contains(strings.ToLower(r.Header.Get("Content-Encoding")), "aws-chunked") ||
		strings.HasPrefix(r.Header.Get("X-Amz-Content-Sha256"), "STREAMING-") {
		// Refuse loudly rather than seal chunk-signature framing into a
		// block as if it were payload — silent corruption is the one
		// unforgivable failure for a content-addressed store.
		h.writeError(w, r, http.StatusNotImplemented, "NotImplemented",
			"streaming (aws-chunked) payloads are not supported; configure the client to send the body raw (disable streaming/trailer checksums)")
		return
	}
	switch {
	case r.ContentLength < 0:
		h.writeError(w, r, http.StatusLengthRequired, "MissingContentLength",
			"PutObject requires Content-Length (chunked transfer encoding is not supported)")
		return
	case r.ContentLength > int64(h.maxBlobLen):
		h.writeError(w, r, http.StatusBadRequest, "EntityTooLarge",
			fmt.Sprintf("object is %d bytes; max_blob_len is %d", r.ContentLength, h.maxBlobLen))
		return
	}
	// Backstop tripwire: the checks above are the real refusals (declared
	// oversize never allocates, undeclared length never reads). MaxBytesReader
	// guarantees that if a future path ever reaches ReadFull without a
	// trustworthy Content-Length (say, a chunked-transfer allowance), the read
	// still cannot exceed max_blob_len — and an overrun marks the connection
	// unreusable so a lying client cannot keep streaming into it.
	r.Body = http.MaxBytesReader(w, r.Body, int64(h.maxBlobLen))
	buf := make([]byte, r.ContentLength) // bounded by the max_blob_len check above
	if _, err := io.ReadFull(r.Body, buf); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			h.writeError(w, r, http.StatusBadRequest, "EntityTooLarge",
				fmt.Sprintf("object exceeds max_blob_len (%d)", h.maxBlobLen))
			return
		}
		h.writeError(w, r, http.StatusBadRequest, "IncompleteBody",
			"request body ended before Content-Length bytes arrived")
		return
	}
	sum := xxh3.Hash(buf)
	switch st := h.store.Put(ns, key, buf, sum); st { // Put takes ownership of buf
	case protocol.StatusOK, protocol.StatusOKExists:
		// OK_EXISTS is the write-once idempotent hit (same key, same digest):
		// a success on the wire (§3.4) and a success here.
		w.Header().Set("ETag", etagOf(sum))
		w.WriteHeader(http.StatusOK)
	case protocol.StatusErrImmutableConflict:
		h.writeError(w, r, http.StatusConflict, "ImmutableConflict",
			"key already stored with different content: blocks are write-once and keys are content-derived — this indicates client-side corruption")
	case protocol.StatusErrQuotaBytes:
		h.writeError(w, r, http.StatusInsufficientStorage, "QuotaExceeded",
			"namespace byte quota exhausted")
	case protocol.StatusErrBusy:
		// S3's retryable slow-down code; SDKs back off and retry on 503.
		h.writeError(w, r, http.StatusServiceUnavailable, "SlowDown", "transient backpressure; retry")
	default:
		h.writeError(w, r, http.StatusInternalServerError, "InternalError",
			"store refused the write: "+st.String())
	}
}

// getObject serves GetObject (head=false) and HeadObject (head=true) — the
// pair share lookup, headers, and Range arithmetic; HEAD just never writes
// the body (S3 HEAD honors Range in its headers, and so does this).
func (h *S3Compat) getObject(w http.ResponseWriter, r *http.Request, ns uint32, key [32]byte, head bool) {
	data, sum, st := h.getCopy(ns, key)
	switch st {
	case protocol.StatusOK:
	case protocol.StatusErrBusy:
		h.writeError(w, r, http.StatusServiceUnavailable, "SlowDown", "device readers saturated; retry")
		return
	default:
		// Absent here INCLUDES present-in-another-namespace: isolation is
		// NOT_FOUND semantics, never a FORBIDDEN that leaks existence.
		h.writeError(w, r, http.StatusNotFound, "NoSuchKey",
			"the specified key does not exist in this namespace")
		return
	}
	hdr := w.Header()
	hdr.Set("Accept-Ranges", "bytes")
	hdr.Set("ETag", etagOf(sum))
	hdr.Set("Content-Type", "application/octet-stream")
	body, status := data, http.StatusOK
	if rng := r.Header.Get("Range"); rng != "" {
		switch start, end, spec := parseRange(rng, int64(len(data))); spec {
		case rangeValid:
			hdr.Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, end, len(data)))
			body, status = data[start:end+1], http.StatusPartialContent
		case rangeUnsatisfiable:
			hdr.Set("Content-Range", fmt.Sprintf("bytes */%d", len(data)))
			h.writeError(w, r, http.StatusRequestedRangeNotSatisfiable, "InvalidRange",
				"the requested range is not satisfiable")
			return
		case rangeIgnore:
			// Malformed or multi-range: serve the whole object — RFC 9110's
			// MAY-ignore, and exactly what S3 does.
		}
	}
	hdr.Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(status)
	if !head {
		// G705: the body is opaque KV-block bytes served as
		// application/octet-stream to a machine S3 client (NIXL/vLLM obj) —
		// not HTML to a browser, so there is no XSS surface.
		_, _ = w.Write(body) //nolint:gosec // G705: binary object body, not markup
	}
}

// getCopy reads one block through the store's richest available extension —
// tierRefGetter (per-key ERR_BUSY honesty) → refGetter → plain Get — and
// ALWAYS returns a heap copy, releasing any arena reference before returning:
// an HTTP writer can stall for seconds, and holding a block reference across
// that would fight DELETE/eviction for the extent's lifetime.
func (h *S3Compat) getCopy(ns uint32, key [32]byte) (data []byte, sum uint64, st protocol.Status) {
	copyOut := func(view []byte, release func()) []byte {
		out := make([]byte, len(view))
		copy(out, view)
		if release != nil {
			release()
		}
		return out
	}
	switch s := h.store.(type) {
	case tierRefGetter:
		view, x, rel, _, tst := s.GetRefTier(ns, key)
		if tst != protocol.StatusOK {
			return nil, 0, tst
		}
		return copyOut(view, rel), x, protocol.StatusOK
	case refGetter:
		view, x, rel, ok := s.GetRef(ns, key)
		if !ok {
			return nil, 0, protocol.StatusNotFound
		}
		return copyOut(view, rel), x, protocol.StatusOK
	default:
		d, x, ok := h.store.Get(ns, key)
		if !ok {
			return nil, 0, protocol.StatusNotFound
		}
		return d, x, protocol.StatusOK
	}
}

// tokenFromAuthorization extracts the tenant token from an Authorization
// header, accepting BOTH shapes the header comment documents (C-11):
//
//	Bearer <token>                          — the native scheme
//	AWS4-HMAC-SHA256 Credential=<token>/... — aws-sdk clients (NIXL obj)
//	  that cannot emit Bearer; the access-key-id IS the token and the
//	  signature is deliberately not verified.
func tokenFromAuthorization(auth string) (token []byte, ok bool) {
	const bearer = "Bearer "
	const sigv4 = "AWS4-HMAC-SHA256"
	switch {
	case len(auth) > len(bearer) && strings.EqualFold(auth[:len(bearer)], bearer):
		tok := strings.TrimSpace(auth[len(bearer):])
		if tok == "" {
			return nil, false
		}
		return []byte(tok), true
	case len(auth) > len(sigv4) && strings.EqualFold(auth[:len(sigv4)], sigv4):
		// Credential=<access-key-id>/<date>/<region>/<service>/aws4_request
		rest := auth[len(sigv4):]
		i := strings.Index(rest, "Credential=")
		if i < 0 {
			return nil, false
		}
		cred := rest[i+len("Credential="):]
		if j := strings.IndexByte(cred, ','); j >= 0 {
			cred = cred[:j]
		}
		akid, _, found := strings.Cut(cred, "/")
		if !found || akid == "" {
			return nil, false
		}
		return []byte(akid), true
	default:
		return nil, false
	}
}

// splitObjectPath splits a path-style URL into (bucket, key). Virtual-hosted
// addressing is not supported — with endpoint_override pointing at an IP or
// bare host, aws-sdk clients use path-style anyway.
func splitObjectPath(p string) (bucket, key string) {
	bucket, key, _ = strings.Cut(strings.TrimPrefix(p, "/"), "/")
	return bucket, key
}

// decodeObjectKey applies the object-key rule: exactly 64 hex characters →
// the 32-byte block key. hex.Decode makes the match case-insensitive.
func decodeObjectKey(s string) (key [32]byte, ok bool) {
	if len(s) != len(key)*2 {
		return key, false
	}
	if _, err := hex.Decode(key[:], []byte(s)); err != nil {
		return key, false
	}
	return key, true
}

// etagOf renders the block's xxh3_64 as the quoted ETag (16 hex chars). Not
// an MD5 — documented in the file header.
func etagOf(sum uint64) string { return fmt.Sprintf("%q", fmt.Sprintf("%016x", sum)) }

// rangeSpec classifies a Range header against an object.
type rangeSpec int

const (
	rangeIgnore        rangeSpec = iota // malformed / multi-range / non-bytes: serve the whole object
	rangeValid                          // [start, end] inclusive, in bounds
	rangeUnsatisfiable                  // well-formed but selects nothing → 416
)

// parseRange interprets a SINGLE "bytes=" range against size (RFC 9110 §14.1
// int-range and suffix-range forms). S3 supports exactly one range and
// ignores anything it cannot parse, returning the whole object; unsatisfiable
// ranges (start beyond EOF, zero-length suffix, any range on an empty
// object) are 416.
func parseRange(hdr string, size int64) (start, end int64, spec rangeSpec) {
	const unit = "bytes="
	if len(hdr) < len(unit) || !strings.EqualFold(hdr[:len(unit)], unit) {
		return 0, 0, rangeIgnore
	}
	r := strings.TrimSpace(hdr[len(unit):])
	if strings.Contains(r, ",") { // multi-range: S3 ignores → whole object
		return 0, 0, rangeIgnore
	}
	firstStr, lastStr, found := strings.Cut(r, "-")
	if !found {
		return 0, 0, rangeIgnore
	}
	firstStr, lastStr = strings.TrimSpace(firstStr), strings.TrimSpace(lastStr)
	if firstStr == "" { // suffix form "-N": the final N bytes
		n, err := strconv.ParseInt(lastStr, 10, 64)
		if err != nil || n < 0 {
			return 0, 0, rangeIgnore
		}
		if n == 0 || size == 0 {
			return 0, 0, rangeUnsatisfiable
		}
		if n > size {
			n = size
		}
		return size - n, size - 1, rangeValid
	}
	first, err := strconv.ParseInt(firstStr, 10, 64)
	if err != nil || first < 0 {
		return 0, 0, rangeIgnore
	}
	if first >= size { // includes every range on an empty object
		return 0, 0, rangeUnsatisfiable
	}
	last := size - 1
	if lastStr != "" {
		v, verr := strconv.ParseInt(lastStr, 10, 64)
		if verr != nil || v < first {
			return 0, 0, rangeIgnore
		}
		if v < last {
			last = v
		}
	}
	return first, last, rangeValid
}

// s3Error is the S3 REST error document (the subset every SDK parses).
type s3Error struct {
	XMLName  xml.Name `xml:"Error"`
	Code     string   `xml:"Code"`
	Message  string   `xml:"Message"`
	Resource string   `xml:"Resource"`
}

// writeError answers with an S3-shaped XML error. HEAD responses carry the
// status only (a HEAD body would poison Content-Length).
func (h *S3Compat) writeError(w http.ResponseWriter, r *http.Request, status int, code, msg string) {
	if r.Method == http.MethodHead {
		w.WriteHeader(status)
		return
	}
	body, err := xml.Marshal(s3Error{Code: code, Message: msg, Resource: r.URL.Path})
	if err != nil { // cannot happen for a flat struct; stay well-formed anyway
		w.WriteHeader(status)
		return
	}
	full := append([]byte(xml.Header), body...)
	hdr := w.Header()
	hdr.Set("Content-Type", "application/xml")
	hdr.Set("Content-Length", strconv.Itoa(len(full)))
	w.WriteHeader(status)
	_, _ = w.Write(full)
}
