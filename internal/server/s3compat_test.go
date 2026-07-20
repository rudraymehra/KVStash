package server_test

import (
	"bytes"
	"context"
	"encoding/hex"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/kvstash/kvblockd/internal/config"
	"github.com/kvstash/kvblockd/internal/protocol"
	"github.com/kvstash/kvblockd/internal/server"
	"github.com/kvstash/kvblockd/internal/store/ramstub"
)

// The s3compat wall: two tenants against a ramstub store, driven over real
// HTTP (httptest). Everything here goes through ServeHTTP — the same handler
// the separate listener serves.

const (
	s3TokenA = "tok-a-53cr3t"
	s3TokenB = "tok-b-0th3r"
)

// s3TwoTenants builds a two-namespace table through the REAL loader (temp
// YAML), so bucket-name → namespace-id resolution is the production path.
func s3TwoTenants(t *testing.T) *server.Namespaces {
	t.Helper()
	p := filepath.Join(t.TempDir(), "namespaces.yaml")
	doc := "namespaces:\n" +
		"  - {name: bucket-a, id: 1, token: " + s3TokenA + "}\n" +
		"  - {name: bucket-b, id: 2, token: " + s3TokenB + "}\n"
	if err := os.WriteFile(p, []byte(doc), 0o600); err != nil {
		t.Fatal(err)
	}
	ns, err := server.LoadNamespaces(p)
	if err != nil {
		t.Fatal(err)
	}
	return ns
}

func newS3Wall(t *testing.T, cfg config.Config) *httptest.Server {
	t.Helper()
	ts := httptest.NewServer(server.NewS3Compat(cfg, ramstub.New(), s3TwoTenants(t)))
	t.Cleanup(ts.Close)
	return ts
}

// s3Resp is one fully-drained response: body read and closed before the
// helper returns, so no test can leak a keep-alive body.
type s3Resp struct {
	status int
	header http.Header
	body   []byte
}

// s3do fires one request. auth == "" sends no Authorization header; body may
// be nil. Extra headers ride in hdr.
func s3do(t *testing.T, ts *httptest.Server, method, path, auth string, body []byte, hdr map[string]string) s3Resp {
	t.Helper()
	var rd io.Reader
	if body != nil {
		rd = bytes.NewReader(body)
	}
	req, err := http.NewRequest(method, ts.URL+path, rd)
	if err != nil {
		t.Fatal(err)
	}
	if auth != "" {
		req.Header.Set("Authorization", auth)
	}
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	resp, err := ts.Client().Do(req)
	return drain(t, resp, err)
}

// drain consumes an (http.Response, error) pair into an s3Resp.
func drain(t *testing.T, resp *http.Response, err error) s3Resp {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return s3Resp{status: resp.StatusCode, header: resp.Header, body: b}
}

// wantStatus asserts the code and returns the body for follow-up assertions
// (S3 error Code strings).
func wantStatus(t *testing.T, r s3Resp, want int) []byte {
	t.Helper()
	if r.status != want {
		t.Fatalf("status = %d, want %d (body: %s)", r.status, want, r.body)
	}
	return r.body
}

func wantErrCode(t *testing.T, body []byte, code string) {
	t.Helper()
	if !strings.Contains(string(body), "<Code>"+code+"</Code>") {
		t.Fatalf("error body missing <Code>%s</Code>: %s", code, body)
	}
}

// s3key renders a block key per the object-key rule (hex of the 32 bytes).
func s3key(b byte) string {
	k := key(b)
	return hex.EncodeToString(k[:])
}

// s3payload is a deterministic non-repeating-ish pattern so ranged reads
// catch off-by-one slicing.
func s3payload(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*7 + i>>9)
	}
	return b
}

func bearer(tok string) string { return "Bearer " + tok }

// sigv4 builds the aws-sdk-shaped Authorization header (the C-11 path): the
// access-key-id position carries the tenant token; the signature is noise
// the endpoint deliberately ignores.
func sigv4(akid string) string {
	return "AWS4-HMAC-SHA256 Credential=" + akid + "/20260720/us-east-1/s3/aws4_request, " +
		"SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=deadbeef"
}

func TestS3CompatPutGetRoundTrip(t *testing.T) {
	ts := newS3Wall(t, config.Default())
	path := "/bucket-a/" + s3key(0x11)
	data := s3payload(1 << 20)

	r := s3do(t, ts, http.MethodPut, path, bearer(s3TokenA), data, nil)
	wantStatus(t, r, http.StatusOK)
	etag := r.header.Get("ETag")
	if etag == "" {
		t.Fatal("PUT response missing ETag")
	}

	r = s3do(t, ts, http.MethodGet, path, bearer(s3TokenA), nil, nil)
	if r.header.Get("ETag") != etag {
		t.Fatalf("GET ETag %q != PUT ETag %q", r.header.Get("ETag"), etag)
	}
	got := wantStatus(t, r, http.StatusOK)
	if !bytes.Equal(got, data) {
		t.Fatalf("GET returned %d bytes, not byte-identical to the %d PUT", len(got), len(data))
	}

	// Write-once: identical re-PUT is an idempotent success…
	wantStatus(t, s3do(t, ts, http.MethodPut, path, bearer(s3TokenA), data, nil), http.StatusOK)
	// …different bytes on the same key is the corruption alarm.
	other := s3payload(1 << 20)
	other[0] ^= 0xFF
	body := wantStatus(t, s3do(t, ts, http.MethodPut, path, bearer(s3TokenA), other, nil), http.StatusConflict)
	wantErrCode(t, body, "ImmutableConflict")

	// The SigV4-shaped header reads the same block (C-11: aws-sdk clients
	// cannot emit Bearer; the access-key-id is the token).
	got = wantStatus(t, s3do(t, ts, http.MethodGet, path, sigv4(s3TokenA), nil, nil), http.StatusOK)
	if !bytes.Equal(got, data) {
		t.Fatal("SigV4-authenticated GET did not return the stored bytes")
	}
}

func TestS3CompatRangedGet(t *testing.T) {
	ts := newS3Wall(t, config.Default())
	const size = 70000
	path := "/bucket-a/" + s3key(0x22)
	data := s3payload(size)
	wantStatus(t, s3do(t, ts, http.MethodPut, path, bearer(s3TokenA), data, nil), http.StatusOK)

	cases := []struct {
		name, rng  string
		status     int
		want       []byte // 206/200 bodies
		wantCRange string
	}{
		{"first byte", "bytes=0-0", 206, data[0:1], "bytes 0-0/70000"},
		{"interior hundred", "bytes=100-199", 206, data[100:200], "bytes 100-199/70000"},
		{"open tail", "bytes=69990-", 206, data[69990:], "bytes 69990-69999/70000"},
		{"suffix hundred", "bytes=-100", 206, data[size-100:], "bytes 69900-69999/70000"},
		{"end clamped to EOF", "bytes=0-99999", 206, data, "bytes 0-69999/70000"},
		{"start at EOF", "bytes=70000-", 416, nil, "bytes */70000"},
		{"zero-length suffix", "bytes=-0", 416, nil, "bytes */70000"},
		{"inverted is ignored", "bytes=5-2", 200, data, ""},
		{"multi-range is ignored", "bytes=0-1,3-4", 200, data, ""},
		{"alien unit is ignored", "octets=1-2", 200, data, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := s3do(t, ts, http.MethodGet, path, bearer(s3TokenA), nil, map[string]string{"Range": tc.rng})
			if cr := r.header.Get("Content-Range"); cr != tc.wantCRange {
				t.Errorf("Content-Range = %q, want %q", cr, tc.wantCRange)
			}
			body := wantStatus(t, r, tc.status)
			if tc.status == 416 {
				wantErrCode(t, body, "InvalidRange")
				return
			}
			if !bytes.Equal(body, tc.want) {
				t.Fatalf("range %q returned %d bytes, want exactly %d (and byte-identical)", tc.rng, len(body), len(tc.want))
			}
		})
	}
}

func TestS3CompatHead(t *testing.T) {
	ts := newS3Wall(t, config.Default())
	path := "/bucket-a/" + s3key(0x33)
	data := s3payload(4096)
	r := s3do(t, ts, http.MethodPut, path, bearer(s3TokenA), data, nil)
	wantStatus(t, r, http.StatusOK)
	etag := r.header.Get("ETag")

	r = s3do(t, ts, http.MethodHead, path, bearer(s3TokenA), nil, nil)
	if cl := r.header.Get("Content-Length"); cl != strconv.Itoa(len(data)) {
		t.Errorf("HEAD Content-Length = %q, want %d", cl, len(data))
	}
	if r.header.Get("ETag") != etag {
		t.Errorf("HEAD ETag = %q, want %q", r.header.Get("ETag"), etag)
	}
	if body := wantStatus(t, r, http.StatusOK); len(body) != 0 {
		t.Fatalf("HEAD carried a %d-byte body", len(body))
	}

	// HEAD honors Range in headers, still bodyless.
	r = s3do(t, ts, http.MethodHead, path, bearer(s3TokenA), nil, map[string]string{"Range": "bytes=0-9"})
	if cl := r.header.Get("Content-Length"); cl != "10" {
		t.Errorf("ranged HEAD Content-Length = %q, want 10", cl)
	}
	if body := wantStatus(t, r, http.StatusPartialContent); len(body) != 0 {
		t.Fatal("ranged HEAD carried a body")
	}

	// Missing key: 404, still bodyless (a HEAD error body would poison
	// Content-Length).
	r = s3do(t, ts, http.MethodHead, "/bucket-a/"+s3key(0xEE), bearer(s3TokenA), nil, nil)
	if body := wantStatus(t, r, http.StatusNotFound); len(body) != 0 {
		t.Fatal("HEAD 404 carried a body")
	}
}

func TestS3CompatAuth(t *testing.T) {
	ts := newS3Wall(t, config.Default())
	path := "/bucket-a/" + s3key(0x44)
	wantStatus(t, s3do(t, ts, http.MethodPut, path, bearer(s3TokenA), s3payload(64), nil), http.StatusOK)

	deny := []struct{ name, auth string }{
		{"no header", ""},
		{"alien scheme", "Basic dXNlcjpwYXNz"},
		{"empty bearer", "Bearer "},
		{"wrong token", bearer("not-the-token")},
		{"other tenant's token", bearer(s3TokenB)},
		{"sigv4 wrong akid", sigv4("not-the-token")},
		{"sigv4 without credential", "AWS4-HMAC-SHA256 SignedHeaders=host, Signature=deadbeef"},
	}
	for _, tc := range deny {
		t.Run(tc.name, func(t *testing.T) {
			body := wantStatus(t, s3do(t, ts, http.MethodGet, path, tc.auth, nil, nil), http.StatusForbidden)
			wantErrCode(t, body, "AccessDenied")
		})
	}

	// The collapse: an unknown bucket with a real token answers EXACTLY like
	// a wrong token on a real bucket — no bucket-enumeration oracle.
	realBucketWrongTok := s3do(t, ts, http.MethodGet, path, bearer(s3TokenB), nil, nil)
	ghostBucketRealTok := s3do(t, ts, http.MethodGet, "/bucket-ghost/"+s3key(0x44), bearer(s3TokenB), nil, nil)
	if realBucketWrongTok.status != ghostBucketRealTok.status {
		t.Fatalf("oracle: wrong-token %d vs unknown-bucket %d", realBucketWrongTok.status, ghostBucketRealTok.status)
	}
	wantErrCode(t, wantStatus(t, realBucketWrongTok, http.StatusForbidden), "AccessDenied")
	wantErrCode(t, wantStatus(t, ghostBucketRealTok, http.StatusForbidden), "AccessDenied")

	// SigV4 with the right akid is a full-rights credential (C-11 path).
	wantStatus(t, s3do(t, ts, http.MethodHead, path, sigv4(s3TokenA), nil, nil), http.StatusOK)
}

func TestS3CompatCrossNamespaceIsolation(t *testing.T) {
	ts := newS3Wall(t, config.Default())
	k := s3key(0x55)
	dataA := s3payload(8192)

	wantStatus(t, s3do(t, ts, http.MethodPut, "/bucket-a/"+k, bearer(s3TokenA), dataA, nil), http.StatusOK)

	// B's token cannot read bucket A at all: 403 at the bucket gate.
	body := wantStatus(t, s3do(t, ts, http.MethodGet, "/bucket-a/"+k, bearer(s3TokenB), nil, nil), http.StatusForbidden)
	wantErrCode(t, body, "AccessDenied")

	// Through B's OWN bucket the key simply does not exist: NOT_FOUND
	// semantics (disjoint keyspaces), never a FORBIDDEN that would leak that
	// the key lives in namespace A.
	body = wantStatus(t, s3do(t, ts, http.MethodGet, "/bucket-b/"+k, bearer(s3TokenB), nil, nil), http.StatusNotFound)
	wantErrCode(t, body, "NoSuchKey")

	// Write-once is per-namespace: B storing DIFFERENT bytes under the same
	// key hex is a fresh block, not a conflict…
	dataB := s3payload(8192)
	dataB[0] ^= 0xFF
	wantStatus(t, s3do(t, ts, http.MethodPut, "/bucket-b/"+k, bearer(s3TokenB), dataB, nil), http.StatusOK)

	// …and A's block is untouched by it.
	got := wantStatus(t, s3do(t, ts, http.MethodGet, "/bucket-a/"+k, bearer(s3TokenA), nil, nil), http.StatusOK)
	if !bytes.Equal(got, dataA) {
		t.Fatal("namespace A's block changed after namespace B's PUT — isolation broken")
	}
	got = wantStatus(t, s3do(t, ts, http.MethodGet, "/bucket-b/"+k, bearer(s3TokenB), nil, nil), http.StatusOK)
	if !bytes.Equal(got, dataB) {
		t.Fatal("namespace B read back the wrong bytes")
	}
}

func TestS3CompatOversizeRefused(t *testing.T) {
	cfg := config.Default()
	cfg.MaxBlobLen = protocol.FloorMaxBlobLen // 4 MiB: the smallest legal ceiling
	ts := newS3Wall(t, cfg)
	path := "/bucket-a/" + s3key(0x66)

	// One byte over the ceiling: refused before the body is read.
	body := wantStatus(t, s3do(t, ts, http.MethodPut, path, bearer(s3TokenA),
		make([]byte, int(cfg.MaxBlobLen)+1), nil), http.StatusBadRequest)
	wantErrCode(t, body, "EntityTooLarge")

	// Exactly at the ceiling: accepted.
	wantStatus(t, s3do(t, ts, http.MethodPut, path, bearer(s3TokenA),
		make([]byte, int(cfg.MaxBlobLen)), nil), http.StatusOK)

	// No Content-Length (chunked transfer encoding): 411, whole-object PUTs
	// must declare their size. The opaque reader defeats http.NewRequest's
	// ContentLength sniffing, forcing chunked encoding.
	req, err := http.NewRequest(http.MethodPut, ts.URL+"/bucket-a/"+s3key(0x67),
		struct{ io.Reader }{bytes.NewReader(make([]byte, 16))})
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", bearer(s3TokenA))
	resp, err := ts.Client().Do(req)
	body = wantStatus(t, drain(t, resp, err), http.StatusLengthRequired)
	wantErrCode(t, body, "MissingContentLength")
}

func TestS3CompatUnsupportedSurface(t *testing.T) {
	ts := newS3Wall(t, config.Default())
	k := s3key(0x77)
	auth := bearer(s3TokenA)

	cases := []struct {
		name, method, path, auth string
		hdr                      map[string]string
		status                   int
		code                     string
	}{
		{"multipart initiate", http.MethodPost, "/bucket-a/" + k + "?uploads", auth, nil, 501, "NotImplemented"},
		{"multipart part", http.MethodPut, "/bucket-a/" + k + "?partNumber=1&uploadId=x", auth, nil, 501, "NotImplemented"},
		{"delete object", http.MethodDelete, "/bucket-a/" + k, auth, nil, 501, "NotImplemented"},
		{"service root", http.MethodGet, "/", "", nil, 501, "NotImplemented"},
		{"bucket-level list", http.MethodGet, "/bucket-a", auth, nil, 501, "NotImplemented"},
		{"key not hex", http.MethodGet, "/bucket-a/not-a-key", auth, nil, 400, "InvalidArgument"},
		{"key too short", http.MethodGet, "/bucket-a/" + k[:62], auth, nil, 400, "InvalidArgument"},
		{
			"streaming sha payload", http.MethodPut, "/bucket-a/" + k, auth,
			map[string]string{"X-Amz-Content-Sha256": "STREAMING-AWS4-HMAC-SHA256-PAYLOAD"},
			501, "NotImplemented",
		},
		{
			"aws-chunked encoding", http.MethodPut, "/bucket-a/" + k, auth,
			map[string]string{"Content-Encoding": "aws-chunked"},
			501, "NotImplemented",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := wantStatus(t, s3do(t, ts, tc.method, tc.path, tc.auth, []byte("x"), tc.hdr), tc.status)
			wantErrCode(t, body, tc.code)
		})
	}
}

func TestS3CompatZeroLengthObject(t *testing.T) {
	ts := newS3Wall(t, config.Default())
	path := "/bucket-a/" + s3key(0x88)

	wantStatus(t, s3do(t, ts, http.MethodPut, path, bearer(s3TokenA), []byte{}, nil), http.StatusOK)
	if got := wantStatus(t, s3do(t, ts, http.MethodGet, path, bearer(s3TokenA), nil, nil), http.StatusOK); len(got) != 0 {
		t.Fatalf("zero-length object came back with %d bytes", len(got))
	}
	r := s3do(t, ts, http.MethodHead, path, bearer(s3TokenA), nil, nil)
	if cl := r.header.Get("Content-Length"); cl != "0" {
		t.Errorf("HEAD Content-Length = %q, want 0", cl)
	}
	wantStatus(t, r, http.StatusOK)

	// Every range on an empty object is unsatisfiable (S3: 416).
	r = s3do(t, ts, http.MethodGet, path, bearer(s3TokenA), nil, map[string]string{"Range": "bytes=0-0"})
	wantErrCode(t, wantStatus(t, r, http.StatusRequestedRangeNotSatisfiable), "InvalidRange")
}

// TestS3CompatServeLifecycle drives the real listener path: bind :0, serve a
// request, cancel, and require a CLEAN wait (the main.go drain-before-Close
// gate depends on that bool).
func TestS3CompatServeLifecycle(t *testing.T) {
	h := server.NewS3Compat(config.Default(), ramstub.New(), s3TwoTenants(t))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	bound, wait, err := h.Serve(ctx, "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	tr := &http.Transport{}
	client := &http.Client{Transport: tr, Timeout: 5 * time.Second}
	defer tr.CloseIdleConnections()

	resp, err := client.Get("http://" + bound + "/")
	if body := wantStatus(t, drain(t, resp, err), http.StatusNotImplemented); len(body) == 0 {
		t.Fatal("expected an XML error body from the live listener")
	}

	cancel()
	if !wait() {
		t.Fatal("shutdown was not clean with no in-flight handlers")
	}
	if resp, err := client.Get("http://" + bound + "/"); err == nil {
		_ = resp.Body.Close()
		t.Fatal("listener still accepting after clean shutdown")
	}
}

// TestS3CompatSlowBodyBounded is the slow-loris guard: a PUT that declares a
// 1 MiB body, sends 16 bytes, and goes silent must be cut off by the server's
// read deadline (transport BodyReadTimeout's intent) — not pin the
// pre-allocated buffer until someone gives up. It drives the REAL listener,
// because the deadlines live on Serve's http.Server, not on the handler.
func TestS3CompatSlowBodyBounded(t *testing.T) {
	cfg := config.Default()
	// 1s deadlines — below Validate's 5s wire floor, but this cfg is
	// hand-rolled precisely so the test finishes in ~1s, not ~30.
	cfg.StreamTimeoutMS = 1000
	h := server.NewS3Compat(cfg, ramstub.New(), s3TwoTenants(t))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	bound, wait, err := h.Serve(ctx, "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}

	conn, err := net.Dial("tcp", bound)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	req := "PUT /bucket-a/" + s3key(0x99) + " HTTP/1.1\r\n" +
		"Host: " + bound + "\r\n" +
		"Authorization: " + bearer(s3TokenA) + "\r\n" +
		"Content-Length: 1048576\r\n" +
		"\r\n" +
		"sixteen bytes!!!" // …and then silence
	if _, err := conn.Write([]byte(req)); err != nil {
		t.Fatal(err)
	}

	// The conn deadline is the test's own backstop, NOT the assertion — the
	// assertion is that the server answers in read-deadline time, far under it.
	start := time.Now()
	_ = conn.SetReadDeadline(time.Now().Add(15 * time.Second))
	resp, err := io.ReadAll(conn) // returns at server close; the 400 rides along
	elapsed := time.Since(start)
	if len(resp) == 0 {
		t.Fatalf("no response within the backstop (err %v) — the stalled body pinned its handler", err)
	}
	if elapsed > 8*time.Second {
		t.Fatalf("server took %v to cut off a stalled body; want ~the 1s read deadline", elapsed)
	}
	if !bytes.HasPrefix(resp, []byte("HTTP/1.1 400")) || !bytes.Contains(resp, []byte("<Code>IncompleteBody</Code>")) {
		t.Fatalf("stalled body answered %q, want 400 IncompleteBody", resp)
	}

	cancel()
	if !wait() {
		t.Fatal("shutdown was not clean — the stalled handler is still pinned")
	}
}
