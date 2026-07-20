// kvblockd_client.h — C++ wire client for the kvblockd KVB1 protocol.
//
// This is the shared native client core (SPEC 5 cross-adapter note #1): a
// blocking-socket implementation of docs/PROTOCOL.md v1, byte-identical to
// internal/protocol (Go) and python/kvblockd (Python). The golden hex vectors
// in internal/protocol/testdata/frames are the shared oracle; test_protocol.cpp
// pins this codec against them.
//
// Wire law honored here (PROTOCOL.md):
//   - 64-byte fixed header; magic reads "KVB1" in a hexdump; every other
//     integer is little-endian; header CRC32C (Castagnoli) over bytes 0..59.
//   - 8 verb families; batch verbs carry keys in the body (header key zeroed);
//     PUT_STREAM is the only per-key verb (key in the header on all sub-ops).
//   - Credit backpressure (§8): the HELLO response's initial_credit opens the
//     client->server byte window; EVERY request frame debits payload_len (one
//     rule, no exemptions); every received header's credit field replenishes.
//     Send is allowed while the window is positive — overshoot by at most one
//     frame is by design (§8 rule 3).
//   - Payload integrity is per-block xxh3_64 (computed at PUT COMMIT, echoed
//     in every GET descriptor, verified after recv).
//
// Deliberately exception-free on verb paths: verbs return VerbResult, and a
// connection whose stream state is unknown is marked dead and never re-pooled
// (the Python pool's discard rule). Not thread-safe per Conn; the Pool
// guarantees single ownership of a checked-out connection.

#ifndef KVBLOCKD_CLIENT_H
#define KVBLOCKD_CLIENT_H

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace kvb {

// --- constants (mirror internal/protocol/{header,ops}.go) -------------------

constexpr size_t kHeaderSize = 64;
constexpr size_t kPreambleSize = 8;
constexpr size_t kDescSize = 16;
constexpr uint8_t kVersion1 = 0x01;
constexpr uint32_t kMagicLE = 0x3142564B; // bytes on the wire: "KVB1"

enum class Op : uint8_t {
    Nop = 0x00,
    Hello = 0x01,
    BatchExists = 0x02,
    BatchGet = 0x03,
    PutStream = 0x04,
    TouchLease = 0x05,
    Pin = 0x06,
    Delete = 0x07,
    Stats = 0x08,
};

// Flags (header bytes 6-7); bits 4-7 carry the sub-op, 8-15 reserved.
constexpr uint16_t kFlagResp = 0x0001;
constexpr uint16_t kFlagMore = 0x0002;
constexpr uint16_t kFlagFatal = 0x0004;
constexpr uint16_t kFlagForce = 0x0008;

constexpr uint16_t with_subop(uint16_t flags, uint8_t sub) {
    return static_cast<uint16_t>((flags & ~0x00F0u) | (static_cast<uint16_t>(sub & 0xF) << 4));
}
constexpr uint8_t subop(uint16_t flags) { return static_cast<uint8_t>((flags >> 4) & 0xF); }

// PUT_STREAM sub-ops (§3.4).
constexpr uint8_t kPutBegin = 0, kPutChunk = 1, kPutCommit = 2, kPutAbort = 3;
// TOUCH_LEASE sub-ops (§3.5).
constexpr uint8_t kTouchRecency = 0, kLeaseGrant = 1, kLeaseRelease = 2;
// PIN sub-ops (§3.6).
constexpr uint8_t kPinSoft = 0, kPinHard = 1, kUnpin = 2;

// Feature bits (§10).
constexpr uint64_t kFeatOOO = 1ull << 0;
constexpr uint64_t kFeatExistsBitmap = 1ull << 1;

// Status codes (§9) — the codes are wire law; mirrors internal/protocol/ops.go.
enum class Status : uint8_t {
    OK = 0x00,
    OKExists = 0x01,
    NotFound = 0x10,
    Evicted = 0x11, // observability nicety; treat as NOT_FOUND
    ErrAuthRequired = 0x20,
    ErrAuthFailed = 0x21,
    ErrNamespaceUnknown = 0x22,
    ErrForbidden = 0x23,
    ErrQuotaBytes = 0x30,
    ErrPinQuota = 0x31,
    ErrTooLarge = 0x32,
    ErrBatchTooLarge = 0x33,
    ErrBusy = 0x34,
    ErrChecksum = 0x40,
    ErrShortStream = 0x41,
    ErrStaleStream = 0x42,
    ErrImmutableConflict = 0x43,
    ErrLeased = 0x44,
    ErrPinned = 0x45,
    ErrUnsupported = 0x50,
    ErrMalformed = 0x51,
    ErrInternal = 0x60,
    FatalProtocol = 0xF0,
};

constexpr bool status_ok(Status s) { return s == Status::OK || s == Status::OKExists; }
const char *status_name(Status s); // §9 name, for logs and test failures

using Key = std::array<uint8_t, 32>;

// --- codec (pure functions; the golden-vector surface) ----------------------

// CRC32C (Castagnoli, reflected poly 0x82F63B78, init/final xor 0xFFFFFFFF) —
// exactly what Go's hash/crc32 Castagnoli table and the Python client compute.
// One CRC per 64-byte header: a table-driven software implementation is plenty.
uint32_t crc32c(const uint8_t *data, size_t len);

// Little-endian scalar put/get (explicit byte ops — correct on any host).
inline void put_u16(uint8_t *p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
}
inline void put_u32(uint8_t *p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v);
    p[1] = static_cast<uint8_t>(v >> 8);
    p[2] = static_cast<uint8_t>(v >> 16);
    p[3] = static_cast<uint8_t>(v >> 24);
}
inline void put_u64(uint8_t *p, uint64_t v) {
    put_u32(p, static_cast<uint32_t>(v));
    put_u32(p + 4, static_cast<uint32_t>(v >> 32));
}
inline uint16_t get_u16(const uint8_t *p) {
    return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}
inline uint32_t get_u32(const uint8_t *p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
        (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
inline uint64_t get_u64(const uint8_t *p) {
    return static_cast<uint64_t>(get_u32(p)) | (static_cast<uint64_t>(get_u32(p + 4)) << 32);
}

// pad-to-8 rule (§0): pad bytes are 0x00, included in payload_len.
constexpr size_t pad8(size_t n) { return (n + 7) & ~size_t{7}; }

struct Header {
    Op opcode = Op::Nop;
    uint16_t flags = 0;
    uint32_t namespace_id = 0;
    uint32_t credit = 0;
    uint64_t request_id = 0;
    Key key{}; // zeroed for batch verbs (keys ride in the body)
    uint32_t payload_len = 0;

    // Marshal into exactly kHeaderSize bytes: magic+version from constants,
    // CRC32C over bytes 0..59 computed and stored last (Go MarshalTo order).
    void marshal_to(uint8_t out[kHeaderSize]) const;

    // Parse + validate in spec order: magic, version, CRC. Returns false (and
    // fills err) on any violation — all three are connection-fatal (§1).
    static bool parse(const uint8_t in[kHeaderSize], Header &h, std::string &err);
};

struct Desc {
    Status status = Status::NotFound;
    uint32_t len = 0;
    uint64_t xxh3 = 0;
};

struct Preamble {
    Status status = Status::ErrInternal;
    uint32_t count = 0;
};

// Body encoders (append to a byte vector; layouts per PROTOCOL.md §3).
void append_hello_req(std::vector<uint8_t> &dst, uint64_t features, uint32_t max_batch_keys,
                      uint32_t max_frame_len, const std::string &token, const std::string &ns,
                      const std::string &client_name);
void append_keylist(std::vector<uint8_t> &dst, uint32_t aux, const std::vector<Key> &keys);
void append_put_begin(std::vector<uint8_t> &dst, uint32_t total_len, uint32_t ttl_ms,
                      uint64_t xxh3_hint);
void append_put_commit(std::vector<uint8_t> &dst, uint64_t xxh3);
void append_stats_req(std::vector<uint8_t> &dst, uint32_t sections);

// Body decoders. All bounds-checked; false = malformed (stream state unknown).
bool parse_preamble(const uint8_t *body, size_t len, Preamble &out);
bool parse_desc(const uint8_t *p, size_t len, Desc &out);

// HELLO response (§3.1). On a non-OK status only `status` is meaningful.
struct HelloResp {
    Status status = Status::ErrInternal;
    uint8_t proto = 0;
    uint64_t features = 0;
    uint32_t max_batch_keys = 0;
    uint32_t max_frame_len = 0;
    uint32_t max_blob_len = 0;
    uint32_t namespace_id = 0;
    uint32_t initial_credit = 0;
    uint32_t lease_default_ms = 0;
    uint32_t lease_max_ms = 0;
    uint32_t stream_timeout_ms = 0;
    std::string server_name;
};
bool parse_hello_resp(const uint8_t *body, size_t len, HelloResp &out);

// --- verb results ------------------------------------------------------------

// How a verb ended, from the connection's point of view.
enum class VerbErr : uint8_t {
    None = 0,       // verb completed; wire_status carries the batch-level §9 code
    Connection = 1, // socket error / peer closed / timeout — conn is dead
    Protocol = 2,   // malformed or fatal frame, desynced stream — conn is dead
    Verify = 3,     // xxh3 mismatch on a read — payload corrupt, conn drained but distrusted
    Usage = 4,      // caller error (bad args); conn unaffected
};

struct VerbResult {
    VerbErr err = VerbErr::None;
    // Batch-level §9 status when err == None; on a server-fatal Protocol error
    // it carries the status the fatal report frame declared (ERR_AUTH_*, …).
    Status wire_status = Status::OK;
    std::string detail; // human-readable context for logs

    bool ok() const { return err == VerbErr::None && status_ok(wire_status); }
};

// One destination region for BATCH_GET: a block lands at ptr[0..cap).
struct GetDest {
    void *ptr = nullptr;
    size_t cap = 0;
    // Outputs:
    Status status = Status::NotFound; // per-key §9 code (non-OK => no bytes written)
    uint32_t len = 0;                 // bytes written on OK
};

// --- Conn --------------------------------------------------------------------

// One authenticated connection: dial + HELLO, then synchronous verbs (one
// in-flight request per connection; adapters get parallelism by striping
// batches across the Pool, the §7 recommended lane shape).
class Conn {
  public:
    struct Options {
        std::string host = "127.0.0.1";
        uint16_t port = 0;
        std::string ns;    // namespace name (bound to namespace_id at HELLO)
        std::string token; // bearer token, crosses the wire once
        std::string client_name = "kvblockd-nixl";
        // Advertise only what this client implements: it parses the EXISTS
        // bitmap but is synchronous per connection, so it does NOT negotiate OOO
        // (mirrors pkg/client and python client).
        uint64_t features = kFeatExistsBitmap;
        uint32_t max_batch_keys = 0; // 0 = accept server default
        uint32_t max_frame_len = 0;  // 0 = accept server default
        int connect_timeout_ms = 5000;
        int op_timeout_ms = 30000; // SO_RCVTIMEO/SO_SNDTIMEO per syscall
        bool verify_reads = true;  // xxh3-verify every GET payload
    };

    Conn() = default;
    ~Conn() { close(); }
    Conn(const Conn &) = delete;
    Conn &operator=(const Conn &) = delete;

    // Dial + HELLO. On failure the conn is unusable (dead()).
    VerbResult connect(const Options &opt);
    void close();
    bool dead() const { return fd_ < 0 || dead_; }

    // Negotiated limits (valid after connect()).
    const HelloResp &limits() const { return hello_; }
    // Remaining §8 send window in bytes (may briefly go negative by design).
    int64_t credit_window() const { return credit_; }

    // The 8 verbs -------------------------------------------------------------

    // BATCH_EXISTS: keys MUST be in prefix-chain order. n_consecutive = hits
    // from position 0 until the first miss. per_key filled iff the bitmap
    // feature was negotiated (present = OK/OK_EXISTS).
    VerbResult batch_exists(const std::vector<Key> &keys, uint32_t &n_consecutive,
                            std::vector<Status> *per_key);

    // BATCH_GET into caller regions (dests.size() == keys.size()). Reassembles
    // F_MORE frames; verifies xxh3 per OK descriptor when verify_reads. A block
    // longer than its dest cap is a Usage error detected before any recv is
    // lost (the payload is drained; the conn stays in sync; status=NotFound).
    // If `cancel` goes true mid-response, remaining payloads are drained
    // instead of written (caller buffers are never touched again); the frame
    // sequence is still consumed in full, so the conn stays in sync, and the
    // result carries detail "canceled" — the releaseReqH read-abort path.
    VerbResult batch_get(const std::vector<Key> &keys, std::vector<GetDest> &dests,
                         const std::atomic<bool> *cancel = nullptr);

    // PUT_STREAM BEGIN->CHUNK*->COMMIT. Computes xxh3 while streaming; chunks
    // at min(4 MiB, negotiated max_frame_len) per §3.4 guidance. OK_EXISTS is
    // the write-once idempotent hit (no bytes sent after BEGIN). If `cancel`
    // goes true between chunks, the stream is ABORTed instead of committed
    // (staging freed server-side, key never visible); the result then carries
    // the ABORT status with detail "canceled" — the releaseReqH abort path.
    VerbResult put(const Key &key, const void *data, size_t len, uint32_t ttl_ms = 0,
                   const std::atomic<bool> *cancel = nullptr);

    // PUT_STREAM BEGIN -> stream `send_prefix` bytes of data -> ABORT. The
    // clean-abort path releaseReqH exercises: staging is freed server-side,
    // the key never becomes visible. wire_status is the ABORT response (OK on
    // a live stream; ERR_STALE_STREAM if e.g. BEGIN answered OK_EXISTS).
    VerbResult put_abort(const Key &key, const void *data, size_t len, size_t send_prefix);

    // TOUCH_LEASE / PIN / DELETE: per-key §9 statuses land in per_key.
    VerbResult touch_lease(const std::vector<Key> &keys, uint8_t sub, uint32_t ttl_ms,
                           std::vector<Status> &per_key);
    VerbResult pin(const std::vector<Key> &keys, uint8_t sub, std::vector<Status> &per_key);
    VerbResult del(const std::vector<Key> &keys, bool force, std::vector<Status> &per_key);

    // STATS: UTF-8 JSON document (cold path).
    VerbResult stats(std::string &json);

  private:
    // Framing. send_frame debits the §8 window (waiting for replenishment when
    // exhausted) and writes header+payload in one sendmsg. next_header skips
    // unsolicited NOP/CREDIT frames, folds EVERY received credit grant into the
    // window, and surfaces fatal report frames as Protocol errors.
    VerbResult send_frame(Header &h, const uint8_t *payload, size_t len);
    VerbResult send_frame2(Header &h, const uint8_t *a, size_t alen, const uint8_t *b,
                           size_t blen);
    VerbResult next_header(Header &h);
    VerbResult wait_for_credit();
    VerbResult read_exact(void *buf, size_t n);
    VerbResult write_iov(const uint8_t *a, size_t alen, const uint8_t *b, size_t blen);
    VerbResult drain(size_t n, void *xxh_state); // discard n bytes (optionally hashing)
    VerbResult read_body(size_t n, std::vector<uint8_t> &out);
    VerbResult fatal_frame_result(const Header &h);
    uint64_t next_id() { return next_id_++; }
    VerbResult key_status_verb(Op op, uint16_t flags, uint32_t aux, const std::vector<Key> &keys,
                               std::vector<Status> &per_key);
    VerbResult conn_err(VerbErr e, std::string detail);

    int fd_ = -1;
    bool dead_ = false;
    bool verify_ = true;
    uint64_t next_id_ = 1; // 0 is reserved for NOP/CREDIT control frames
    int64_t credit_ = 0;   // §8 window; replenished from every received header
    HelloResp hello_;
    std::vector<uint8_t> scratch_; // reusable metadata/drain buffer (capped)
};

// --- Pool --------------------------------------------------------------------

// A bounded set of live connections to one (endpoint, namespace, token),
// checked out one-at-a-time per verb call. An errored (dead) connection is
// closed and lazily replaced on the next checkout — self-healing, mirroring
// python/kvblockd/pool.py. Default 16 streams (PROTOCOL.md §4 guidance).
class Pool {
  public:
    Pool(Conn::Options opt, size_t streams);
    ~Pool();
    Pool(const Pool &) = delete;
    Pool &operator=(const Pool &) = delete;

    // Dial the first connection eagerly so auth/endpoint failures surface at
    // construction (and negotiated limits become available via limits()).
    VerbResult prime();
    const HelloResp &limits() const { return limits_; }

    // Run fn with exclusive ownership of one connection. If the conn is dead
    // after fn (connection/protocol/verify error), it is discarded; otherwise
    // it is returned for reuse. Blocks while all `streams` conns are busy.
    // The checkin rides a scope guard: if fn throws (std::bad_alloc growing a
    // response buffer, say), the conn's stream state is unknown, so it is
    // closed (dead => discarded) — but the slot is ALWAYS released, or the
    // pool would leak capacity and eventually deadlock every caller.
    template <typename F> auto with_conn(F &&fn) -> decltype(fn(std::declval<Conn &>())) {
        std::unique_ptr<Conn> c;
        VerbResult ck = checkout(c);
        if (ck.err != VerbErr::None) {
            using R = decltype(fn(std::declval<Conn &>()));
            R r{};
            r.err = ck.err;
            r.detail = std::move(ck.detail);
            return r;
        }
        struct CheckinGuard {
            Pool &pool;
            std::unique_ptr<Conn> &conn;
            int base = std::uncaught_exceptions();
            ~CheckinGuard() {
                if (conn && std::uncaught_exceptions() > base) conn->close();
                pool.checkin(std::move(conn));
            }
        } guard{*this, c};
        return fn(*c);
    }

    size_t streams() const { return streams_; }
    void close();

  private:
    VerbResult checkout(std::unique_ptr<Conn> &out);
    void checkin(std::unique_ptr<Conn> c);

    Conn::Options opt_;
    size_t streams_;
    HelloResp limits_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::unique_ptr<Conn>> idle_;
    size_t outstanding_ = 0;
    bool closed_ = false;
};

// xxh3_64 of a buffer — exposed so tests and the backend can pin parity with
// the descriptor checksums (PROTOCOL.md §12 goldens).
uint64_t xxh3_64(const void *data, size_t len);

} // namespace kvb

#endif // KVBLOCKD_CLIENT_H
