// kvblockd_client.cpp — implementation of the KVB1 wire client.
// See kvblockd_client.h for the contract and PROTOCOL.md for the wire law.

#include "kvblockd_client.h"

#define XXH_INLINE_ALL
#include "xxhash.h" // vendored third_party/xxhash (v0.8.3, BSD-2-Clause)

#include <cerrno>
#include <cstdio>

#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

namespace kvb {

// --- crc32c ------------------------------------------------------------------

namespace {

// Castagnoli table, generated once (reflected poly 0x82F63B78) — identical to
// Go crc32.MakeTable(crc32.Castagnoli) and python protocol.crc32c.
struct Crc32cTable {
    uint32_t t[256];
    Crc32cTable() {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? (c >> 1) ^ 0x82F63B78u : c >> 1;
            t[i] = c;
        }
    }
};
const Crc32cTable kCrcTable;

constexpr size_t kMaxScratch = 1 << 20; // reusable metadata/drain buffer cap
constexpr uint32_t kPreHelloFrameCap = 1 << 20; // response cap before negotiation
constexpr size_t kPutChunkCap = 4 << 20;        // §3.4 recommended chunk ceiling

std::string errno_str(const char *what) {
    char buf[128];
    std::snprintf(buf, sizeof buf, "%s: errno %d", what, errno);
    return std::string(buf);
}

} // namespace

uint32_t crc32c(const uint8_t *data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) crc = (crc >> 8) ^ kCrcTable.t[(crc ^ data[i]) & 0xFF];
    return crc ^ 0xFFFFFFFFu;
}

uint64_t xxh3_64(const void *data, size_t len) { return XXH3_64bits(data, len); }

const char *status_name(Status s) {
    switch (s) {
    case Status::OK: return "OK";
    case Status::OKExists: return "OK_EXISTS";
    case Status::NotFound: return "NOT_FOUND";
    case Status::Evicted: return "EVICTED";
    case Status::ErrAuthRequired: return "ERR_AUTH_REQUIRED";
    case Status::ErrAuthFailed: return "ERR_AUTH_FAILED";
    case Status::ErrNamespaceUnknown: return "ERR_NAMESPACE_UNKNOWN";
    case Status::ErrForbidden: return "ERR_FORBIDDEN";
    case Status::ErrQuotaBytes: return "ERR_QUOTA_BYTES";
    case Status::ErrPinQuota: return "ERR_PIN_QUOTA";
    case Status::ErrTooLarge: return "ERR_TOO_LARGE";
    case Status::ErrBatchTooLarge: return "ERR_BATCH_TOO_LARGE";
    case Status::ErrBusy: return "ERR_BUSY";
    case Status::ErrChecksum: return "ERR_CHECKSUM";
    case Status::ErrShortStream: return "ERR_SHORT_STREAM";
    case Status::ErrStaleStream: return "ERR_STALE_STREAM";
    case Status::ErrImmutableConflict: return "ERR_IMMUTABLE_CONFLICT";
    case Status::ErrLeased: return "ERR_LEASED";
    case Status::ErrPinned: return "ERR_PINNED";
    case Status::ErrUnsupported: return "ERR_UNSUPPORTED";
    case Status::ErrMalformed: return "ERR_MALFORMED";
    case Status::ErrInternal: return "ERR_INTERNAL";
    case Status::FatalProtocol: return "FATAL_PROTOCOL";
    }
    return "UNKNOWN_STATUS";
}

// --- header codec --------------------------------------------------------------

void Header::marshal_to(uint8_t out[kHeaderSize]) const {
    put_u32(out + 0, kMagicLE);
    out[4] = kVersion1;
    out[5] = static_cast<uint8_t>(opcode);
    put_u16(out + 6, flags);
    put_u32(out + 8, namespace_id);
    put_u32(out + 12, credit);
    put_u64(out + 16, request_id);
    std::memcpy(out + 24, key.data(), 32);
    put_u32(out + 56, payload_len);
    put_u32(out + 60, crc32c(out, 60)); // CRC over bytes 0..59, stored last
}

bool Header::parse(const uint8_t in[kHeaderSize], Header &h, std::string &err) {
    // Spec-fixed validation order: magic, version, CRC. Nothing after a failed
    // check is trusted — payload_len is only meaningful once the CRC passed.
    if (get_u32(in + 0) != kMagicLE) {
        err = "bad magic (want \"KVB1\")";
        return false;
    }
    if (in[4] != kVersion1) {
        err = "unsupported header version";
        return false;
    }
    if (get_u32(in + 60) != crc32c(in, 60)) {
        err = "header CRC32C mismatch";
        return false;
    }
    h.opcode = static_cast<Op>(in[5]);
    h.flags = get_u16(in + 6);
    h.namespace_id = get_u32(in + 8);
    h.credit = get_u32(in + 12);
    h.request_id = get_u64(in + 16);
    std::memcpy(h.key.data(), in + 24, 32);
    h.payload_len = get_u32(in + 56);
    return true;
}

// --- body codecs -----------------------------------------------------------------

namespace {
void append_u16(std::vector<uint8_t> &d, uint16_t v) {
    uint8_t b[2];
    put_u16(b, v);
    d.insert(d.end(), b, b + 2);
}
void append_u32(std::vector<uint8_t> &d, uint32_t v) {
    uint8_t b[4];
    put_u32(b, v);
    d.insert(d.end(), b, b + 4);
}
void append_u64(std::vector<uint8_t> &d, uint64_t v) {
    uint8_t b[8];
    put_u64(b, v);
    d.insert(d.end(), b, b + 8);
}
void append_pad8(std::vector<uint8_t> &d, size_t region_start) {
    size_t region = d.size() - region_start;
    d.resize(d.size() + (pad8(region) - region), 0);
}
} // namespace

void append_hello_req(std::vector<uint8_t> &dst, uint64_t features, uint32_t max_batch_keys,
                      uint32_t max_frame_len, const std::string &token, const std::string &ns,
                      const std::string &client_name) {
    // proto_min u8 | proto_max u8 | reserved u16 | feature_bits u64 |
    // max_batch_keys u32 | max_frame_len u32 | reserved u64 |
    // token_len u16 | ns_len u16 | client_name_len u16 | reserved u16 |
    // token | ns | client_name | pad to 8 (§3.1).
    size_t start = dst.size();
    dst.push_back(kVersion1); // proto_min
    dst.push_back(kVersion1); // proto_max
    append_u16(dst, 0);
    append_u64(dst, features);
    append_u32(dst, max_batch_keys);
    append_u32(dst, max_frame_len);
    append_u64(dst, 0);
    append_u16(dst, static_cast<uint16_t>(token.size()));
    append_u16(dst, static_cast<uint16_t>(ns.size()));
    append_u16(dst, static_cast<uint16_t>(client_name.size()));
    append_u16(dst, 0);
    dst.insert(dst.end(), token.begin(), token.end());
    dst.insert(dst.end(), ns.begin(), ns.end());
    dst.insert(dst.end(), client_name.begin(), client_name.end());
    append_pad8(dst, start);
}

void append_keylist(std::vector<uint8_t> &dst, uint32_t aux, const std::vector<Key> &keys) {
    // n_keys u32 | aux u32 | key[32] × n — inherently 8-aligned, no pad.
    append_u32(dst, static_cast<uint32_t>(keys.size()));
    append_u32(dst, aux);
    for (const Key &k : keys) dst.insert(dst.end(), k.begin(), k.end());
}

void append_put_begin(std::vector<uint8_t> &dst, uint32_t total_len, uint32_t ttl_ms,
                      uint64_t xxh3_hint) {
    // total_len u32 | ttl_ms u32 | xxh3_64_hint u64 | flags u32 (MUST be 0 in
    // v1) | reserved u32 (§3.4).
    append_u32(dst, total_len);
    append_u32(dst, ttl_ms);
    append_u64(dst, xxh3_hint);
    append_u32(dst, 0);
    append_u32(dst, 0);
}

void append_put_commit(std::vector<uint8_t> &dst, uint64_t xxh3) { append_u64(dst, xxh3); }

void append_stats_req(std::vector<uint8_t> &dst, uint32_t sections) {
    append_u32(dst, sections);
    append_u32(dst, 0);
}

bool parse_preamble(const uint8_t *body, size_t len, Preamble &out) {
    if (len < kPreambleSize) return false;
    out.status = static_cast<Status>(body[0]);
    out.count = get_u32(body + 4);
    return true;
}

bool parse_desc(const uint8_t *p, size_t len, Desc &out) {
    if (len < kDescSize) return false;
    out.status = static_cast<Status>(p[0]);
    out.len = get_u32(p + 4);
    out.xxh3 = get_u64(p + 8);
    return true;
}

bool parse_hello_resp(const uint8_t *body, size_t len, HelloResp &out) {
    // preamble | proto u8, rsvd[3] | features u64 | 8×u32 | name_len u16,
    // rsvd u16 | server_name | pad to 8 (§3.1). helloRespFixed = 56.
    constexpr size_t kFixed = kPreambleSize + 4 + 8 + 32 + 4;
    Preamble p;
    if (!parse_preamble(body, len, p)) return false;
    out.status = p.status;
    if (!status_ok(p.status)) {
        // §3: non-OK responses are exactly the 8-byte preamble with count=0.
        return len == kPreambleSize && p.count == 0;
    }
    if (len < kFixed) return false;
    const uint8_t *b = body + kPreambleSize;
    out.proto = b[0];
    out.features = get_u64(b + 4);
    out.max_batch_keys = get_u32(b + 12);
    out.max_frame_len = get_u32(b + 16);
    out.max_blob_len = get_u32(b + 20);
    out.namespace_id = get_u32(b + 24);
    out.initial_credit = get_u32(b + 28);
    out.lease_default_ms = get_u32(b + 32);
    out.lease_max_ms = get_u32(b + 36);
    out.stream_timeout_ms = get_u32(b + 40);
    uint16_t name_len = get_u16(body + kFixed - 4);
    if (len != pad8(kFixed + name_len)) return false;
    out.server_name.assign(reinterpret_cast<const char *>(body + kFixed), name_len);
    return true;
}

// --- Conn: socket plumbing -----------------------------------------------------

VerbResult Conn::conn_err(VerbErr e, std::string detail) {
    if (e == VerbErr::Connection || e == VerbErr::Protocol || e == VerbErr::Verify) dead_ = true;
    VerbResult r;
    r.err = e;
    r.detail = std::move(detail);
    return r;
}

void Conn::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    dead_ = true;
}

VerbResult Conn::connect(const Options &opt) {
    verify_ = opt.verify_reads;
    // HELLO carries these as u16 length fields (§3.1): reject rather than
    // truncate (a truncated token would also desync the body layout).
    if (opt.token.size() > UINT16_MAX || opt.ns.size() > UINT16_MAX ||
        opt.client_name.size() > UINT16_MAX)
        return conn_err(VerbErr::Usage, "connect: token/ns/client_name exceed u16 wire limit");

    struct addrinfo hints {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo *res = nullptr;
    char portstr[8];
    std::snprintf(portstr, sizeof portstr, "%u", static_cast<unsigned>(opt.port));
    if (int rc = ::getaddrinfo(opt.host.c_str(), portstr, &hints, &res); rc != 0 || !res) {
        return conn_err(VerbErr::Connection, "getaddrinfo failed for " + opt.host);
    }

    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = ::socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        // Bounded connect: nonblocking connect + poll, then back to blocking.
        int fl = ::fcntl(fd, F_GETFL, 0);
        ::fcntl(fd, F_SETFL, fl | O_NONBLOCK);
        int rc = ::connect(fd, ai->ai_addr, ai->ai_addrlen);
        if (rc != 0 && errno == EINPROGRESS) {
            struct pollfd pfd {fd, POLLOUT, 0};
            rc = ::poll(&pfd, 1, opt.connect_timeout_ms) == 1 ? 0 : -1;
            if (rc == 0) {
                int soerr = 0;
                socklen_t slen = sizeof soerr;
                ::getsockopt(fd, SOL_SOCKET, SO_ERROR, &soerr, &slen);
                if (soerr != 0) rc = -1;
            }
        }
        if (rc == 0) {
            ::fcntl(fd, F_SETFL, fl); // back to blocking
            break;
        }
        ::close(fd);
        fd = -1;
    }
    ::freeaddrinfo(res);
    if (fd < 0) return conn_err(VerbErr::Connection, "connect failed to " + opt.host);

    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
#ifdef SO_NOSIGPIPE
    ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
    struct timeval tv {};
    tv.tv_sec = opt.op_timeout_ms / 1000;
    tv.tv_usec = (opt.op_timeout_ms % 1000) * 1000;
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);

    fd_ = fd;
    dead_ = false;

    // HELLO MUST be the first frame; namespace_id=0, key zeroed, nonzero
    // request_id (0 is reserved for NOP/CREDIT).
    std::vector<uint8_t> body;
    append_hello_req(body, opt.features, opt.max_batch_keys, opt.max_frame_len, opt.token,
                     opt.ns, opt.client_name);
    Header h;
    h.opcode = Op::Hello;
    h.request_id = next_id();
    // Pre-negotiation there is no credit window yet; HELLO itself is exempt
    // from the debit (the window opens AT the HELLO response). Give the ledger
    // a temporary allowance so send_frame's one rule stays uniform.
    credit_ = static_cast<int64_t>(body.size()) + 1;
    if (VerbResult r = send_frame(h, body.data(), body.size()); r.err != VerbErr::None) return r;

    Header rh;
    if (VerbResult r = next_header(rh); r.err != VerbErr::None) return r;
    if (rh.payload_len > kPreHelloFrameCap)
        return conn_err(VerbErr::Protocol, "oversized HELLO response");
    std::vector<uint8_t> resp;
    if (VerbResult r = read_body(rh.payload_len, resp); r.err != VerbErr::None) return r;
    if (!parse_hello_resp(resp.data(), resp.size(), hello_))
        return conn_err(VerbErr::Protocol, "malformed HELLO response body");
    if (!status_ok(hello_.status)) {
        VerbResult r; // auth rejection: the server closes (F_FATAL); report the status
        r.err = VerbErr::None;
        r.wire_status = hello_.status;
        r.detail = std::string("HELLO rejected: ") + status_name(hello_.status);
        dead_ = true;
        return r;
    }
    // §8 rule 1: the response's initial_credit opens the window. Grants that
    // rode this header were pre-window noise; start exactly at initial_credit.
    credit_ = static_cast<int64_t>(hello_.initial_credit);
    VerbResult okr;
    okr.wire_status = Status::OK;
    return okr;
}

VerbResult Conn::read_exact(void *buf, size_t n) {
    uint8_t *p = static_cast<uint8_t *>(buf);
    size_t got = 0;
    while (got < n) {
        ssize_t r = ::recv(fd_, p + got, n - got, 0);
        if (r > 0) {
            got += static_cast<size_t>(r);
            continue;
        }
        if (r == 0) return conn_err(VerbErr::Connection, "peer closed mid-frame");
        if (errno == EINTR) continue;
        return conn_err(VerbErr::Connection, errno_str("recv"));
    }
    return {};
}

VerbResult Conn::write_iov(const uint8_t *a, size_t alen, const uint8_t *b, size_t blen) {
    struct iovec iov[2];
    int iovcnt = 0;
    if (alen) iov[iovcnt++] = {const_cast<uint8_t *>(a), alen};
    if (blen) iov[iovcnt++] = {const_cast<uint8_t *>(b), blen};
    while (iovcnt > 0) {
        struct msghdr msg {};
        msg.msg_iov = iov;
        msg.msg_iovlen = static_cast<decltype(msg.msg_iovlen)>(iovcnt);
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags = MSG_NOSIGNAL; // Linux; macOS uses SO_NOSIGPIPE set at connect
#endif
        ssize_t w = ::sendmsg(fd_, &msg, flags);
        if (w < 0) {
            if (errno == EINTR) continue;
            return conn_err(VerbErr::Connection, errno_str("sendmsg"));
        }
        size_t rem = static_cast<size_t>(w);
        int i = 0;
        while (i < iovcnt && rem >= iov[i].iov_len) {
            rem -= iov[i].iov_len;
            i++;
        }
        if (i > 0) {
            for (int j = i; j < iovcnt; j++) iov[j - i] = iov[j];
            iovcnt -= i;
        }
        if (iovcnt > 0 && rem > 0) {
            iov[0].iov_base = static_cast<uint8_t *>(iov[0].iov_base) + rem;
            iov[0].iov_len -= rem;
        }
    }
    return {};
}

VerbResult Conn::fatal_frame_result(const Header &h) {
    // §9 protocol-fatal report: opcode 0, F_RESP|F_FATAL, payload = 8-byte
    // preamble carrying the fatal status. Best-effort read; then dead.
    Status st = Status::FatalProtocol;
    if (h.payload_len >= kPreambleSize && h.payload_len <= kPreHelloFrameCap) {
        std::vector<uint8_t> body;
        if (read_body(h.payload_len, body).err == VerbErr::None) {
            Preamble p;
            if (parse_preamble(body.data(), body.size(), p)) st = p.status;
        }
    }
    // Surface the reported §9 status alongside the Protocol error so callers
    // (a rejected HELLO especially) can distinguish auth failures from CRC
    // garbage without string-matching detail.
    VerbResult r = conn_err(VerbErr::Protocol, std::string("server fatal: ") + status_name(st));
    r.wire_status = st;
    return r;
}

VerbResult Conn::next_header(Header &h) {
    for (;;) {
        uint8_t buf[kHeaderSize];
        if (VerbResult r = read_exact(buf, kHeaderSize); r.err != VerbErr::None) return r;
        std::string perr;
        if (!Header::parse(buf, h, perr)) return conn_err(VerbErr::Protocol, perr);
        // §8 rule 4: EVERY server->client header may carry a credit grant.
        credit_ += h.credit;
        if (h.flags & kFlagFatal) return fatal_frame_result(h);
        if (h.opcode == Op::Nop) {
            // Keepalive / unsolicited credit. A NOP never carries a body, but
            // tolerate one (skip via payload_len) per §3.
            if (h.payload_len) {
                if (VerbResult r = drain(h.payload_len, nullptr); r.err != VerbErr::None)
                    return r;
            }
            continue;
        }
        uint32_t cap = hello_.max_frame_len ? hello_.max_frame_len : kPreHelloFrameCap;
        if (h.payload_len > cap)
            return conn_err(VerbErr::Protocol, "response payload_len exceeds negotiated cap");
        return {};
    }
}

VerbResult Conn::wait_for_credit() {
    // The expected inbound traffic here is NOP/CREDIT: this runs only before a
    // request or between PUT chunks, when no response is due (BEGIN was
    // answered before the first chunk; COMMIT/ABORT not yet sent). Pump frames
    // until the window is positive again; each read is bounded by SO_RCVTIMEO.
    while (credit_ <= 0) {
        uint8_t buf[kHeaderSize];
        if (VerbResult r = read_exact(buf, kHeaderSize); r.err != VerbErr::None) {
            r.detail = "awaiting credit: " + r.detail;
            return r;
        }
        Header h;
        std::string perr;
        if (!Header::parse(buf, h, perr)) return conn_err(VerbErr::Protocol, perr);
        credit_ += h.credit;
        if (h.flags & kFlagFatal) return fatal_frame_result(h);
        // A non-NOP frame is a §2-tolerated stray (a response to a request_id
        // this client never sent), not a framing violation: its credit grant
        // was folded in above, so skip its body via payload_len exactly like
        // the verb response loops do, rather than killing a healthy conn.
        if (h.payload_len) {
            if (VerbResult r = drain(h.payload_len, nullptr); r.err != VerbErr::None) return r;
        }
    }
    return {};
}

VerbResult Conn::send_frame(Header &h, const uint8_t *payload, size_t len) {
    return send_frame2(h, payload, len, nullptr, 0);
}

VerbResult Conn::send_frame2(Header &h, const uint8_t *a, size_t alen, const uint8_t *b,
                             size_t blen) {
    if (dead_) return conn_err(VerbErr::Connection, "connection is dead");
    size_t len = alen + blen;
    // §8 rules 2+3: every request frame debits its payload bytes (headers are
    // free; one rule, no exemptions). Send is allowed while W > 0 — overshoot
    // by at most one frame is by design; at W <= 0 wait for replenishment.
    // A zero-payload frame (ABORT, NOP keepalive) may proceed even at W <= 0:
    // it debits nothing, so it cannot grow the overshoot past rule 5's
    // byte-based enforcement threshold — and gating ABORT on credit could
    // deadlock a cancel (the frame that frees server staging would be waiting
    // on the very window that drain replenishes).
    if (len > 0 && credit_ <= 0) {
        if (VerbResult r = wait_for_credit(); r.err != VerbErr::None) return r;
    }
    credit_ -= static_cast<int64_t>(len);
    h.payload_len = static_cast<uint32_t>(len);
    uint8_t hdr[kHeaderSize];
    h.marshal_to(hdr);
    if (blen == 0) return write_iov(hdr, kHeaderSize, a, alen);
    // Three regions: header + two payload parts. Send header+a, then b.
    if (VerbResult r = write_iov(hdr, kHeaderSize, a, alen); r.err != VerbErr::None) return r;
    return write_iov(b, blen, nullptr, 0);
}

VerbResult Conn::drain(size_t n, void *xxh_state) {
    if (scratch_.size() < kMaxScratch) scratch_.resize(kMaxScratch);
    while (n > 0) {
        size_t chunk = n < kMaxScratch ? n : kMaxScratch;
        if (VerbResult r = read_exact(scratch_.data(), chunk); r.err != VerbErr::None) return r;
        if (xxh_state) XXH3_64bits_update(static_cast<XXH3_state_t *>(xxh_state),
                                          scratch_.data(), chunk);
        n -= chunk;
    }
    return {};
}

VerbResult Conn::read_body(size_t n, std::vector<uint8_t> &out) {
    out.resize(n);
    if (n == 0) return {};
    return read_exact(out.data(), n);
}

// --- verbs ---------------------------------------------------------------------
// Response loops discard spec-tolerated strays: a response whose request_id
// this client never sent is skipped via payload_len and logged nowhere (§2 —
// "discards it and SHOULD log"; this client is silent by design, the pool
// layer owns observability).

VerbResult Conn::batch_exists(const std::vector<Key> &keys, uint32_t &n_consecutive,
                              std::vector<Status> *per_key) {
    n_consecutive = 0;
    if (per_key) per_key->clear();
    bool want_bitmap = (hello_.features & kFeatExistsBitmap) != 0;

    std::vector<uint8_t> body;
    append_keylist(body, 0, keys);
    Header h;
    h.opcode = Op::BatchExists;
    h.namespace_id = hello_.namespace_id;
    h.request_id = next_id();
    if (VerbResult r = send_frame(h, body.data(), body.size()); r.err != VerbErr::None) return r;

    Header rh;
    for (;;) { // skip spec-tolerated stray responses (request_id we never sent)
        if (VerbResult r = next_header(rh); r.err != VerbErr::None) return r;
        if (rh.request_id == h.request_id) break;
        if (VerbResult r = drain(rh.payload_len, nullptr); r.err != VerbErr::None) return r;
    }
    std::vector<uint8_t> resp;
    if (VerbResult r = read_body(rh.payload_len, resp); r.err != VerbErr::None) return r;

    Preamble p;
    if (!parse_preamble(resp.data(), resp.size(), p))
        return conn_err(VerbErr::Protocol, "short EXISTS response");
    VerbResult out;
    out.wire_status = p.status;
    if (!status_ok(p.status)) return out; // §3: preamble-only on non-OK
    size_t want = kPreambleSize + 8 + (want_bitmap ? pad8(p.count) : 0);
    if (resp.size() != want || p.count != keys.size())
        return conn_err(VerbErr::Protocol, "EXISTS response length/count mismatch");
    n_consecutive = get_u32(resp.data() + kPreambleSize);
    if (want_bitmap && per_key) {
        per_key->reserve(p.count);
        for (uint32_t i = 0; i < p.count; i++)
            per_key->push_back(static_cast<Status>(resp[kPreambleSize + 8 + i]));
    }
    return out;
}

VerbResult Conn::batch_get(const std::vector<Key> &keys, std::vector<GetDest> &dests,
                           const std::atomic<bool> *cancel) {
    if (dests.size() != keys.size())
        return conn_err(VerbErr::Usage, "batch_get: dests.size() != keys.size()");
    for (GetDest &d : dests) {
        d.status = Status::NotFound;
        d.len = 0;
    }
    if (keys.empty()) return {};
    if (hello_.max_batch_keys && keys.size() > hello_.max_batch_keys)
        return conn_err(VerbErr::Usage, "batch_get: n_keys over negotiated max_batch_keys");

    std::vector<uint8_t> body;
    append_keylist(body, 0, keys);
    Header h;
    h.opcode = Op::BatchGet;
    h.namespace_id = hello_.namespace_id;
    h.request_id = next_id();
    if (VerbResult r = send_frame(h, body.data(), body.size()); r.err != VerbErr::None) return r;

    const size_t n = keys.size();
    size_t seen = 0;
    bool canceled = false;
    VerbResult usage;        // deferred caller error (undersized dest) — stream stays in sync
    std::vector<Desc> descs; // per-frame descriptor table, parsed up front (see below)
    for (;;) {
        Header rh;
        for (;;) {
            if (VerbResult r = next_header(rh); r.err != VerbErr::None) return r;
            if (rh.request_id == h.request_id) break;
            if (VerbResult r = drain(rh.payload_len, nullptr); r.err != VerbErr::None) return r;
        }
        if (rh.opcode != Op::BatchGet || !(rh.flags & kFlagResp))
            return conn_err(VerbErr::Protocol, "unexpected frame in GET response");

        // Incremental region parse: preamble, then (on OK) index + descriptor
        // table, then payloads streamed straight into dests. Every consumed
        // byte is reconciled against rh.payload_len so a lying frame can never
        // leave the stream position unknown-but-trusted: any mismatch is a
        // Protocol error, which marks the conn dead (never re-pooled).
        if (rh.payload_len < kPreambleSize)
            return conn_err(VerbErr::Protocol, "GET response shorter than preamble");
        uint8_t pre[kPreambleSize];
        if (VerbResult r = read_exact(pre, kPreambleSize); r.err != VerbErr::None) return r;
        Preamble p;
        parse_preamble(pre, sizeof pre, p);
        if (!status_ok(p.status)) {
            // §3: non-OK responses are exactly the 8-byte preamble with count=0.
            if (rh.payload_len != kPreambleSize || p.count != 0)
                return conn_err(VerbErr::Protocol, "non-OK GET response not preamble-only");
            VerbResult out;
            out.wire_status = p.status; // frame fully consumed
            return out;
        }
        // Bounding count by the request size stops a bogus u32 from driving a
        // giant allocation. An F_MORE frame that covers zero keys can never
        // advance `seen` — reject it instead of spinning forever.
        if (p.count > n - seen)
            return conn_err(VerbErr::Protocol, "GET descriptor count exceeds request window");
        if (p.count == 0 && (rh.flags & kFlagMore))
            return conn_err(VerbErr::Protocol, "GET F_MORE frame with zero descriptors");
        size_t region = 8 + kDescSize * p.count;
        if (rh.payload_len < kPreambleSize + region)
            return conn_err(VerbErr::Protocol, "GET descriptor table exceeds payload_len");
        if (scratch_.size() < region) scratch_.resize(region);
        if (VerbResult r = read_exact(scratch_.data(), region); r.err != VerbErr::None) return r;
        uint32_t first = get_u32(scratch_.data());
        uint32_t total = get_u32(scratch_.data() + 4);
        if (total != n || first != seen || first + p.count > n)
            return conn_err(VerbErr::Protocol, "GET frame window invalid");

        // Parse the WHOLE descriptor table before consuming any payload:
        // drain() below reuses scratch_, so a lazily-parsed table would be
        // overwritten by the first drained payload (wrong statuses for every
        // later key + silent stream desync — the F1 clobber).
        descs.resize(p.count);
        for (uint32_t j = 0; j < p.count; j++)
            parse_desc(scratch_.data() + 8 + j * kDescSize, kDescSize, descs[j]);

        // Reconcile the frame's declared length against what we will consume:
        // preamble + index + descriptors + the OK payloads, nothing else
        // (§3.3/§12 — payloads are concatenated raw, no per-block padding).
        // A mismatch means the next frame boundary is unknowable — fatal.
        uint64_t expect = kPreambleSize + 8 + uint64_t{kDescSize} * p.count;
        for (const Desc &d : descs)
            if (status_ok(d.status)) expect += d.len;
        if (expect != rh.payload_len)
            return conn_err(VerbErr::Protocol,
                            "GET payload_len does not reconcile with descriptor table");

        for (uint32_t j = 0; j < p.count; j++) {
            const Desc &d = descs[j];
            GetDest &dst = dests[first + j];
            if (!canceled && cancel && cancel->load(std::memory_order_relaxed)) canceled = true;
            if (!status_ok(d.status)) {
                dst.status = d.status; // payload-free per-key outcome (NOT_FOUND/EVICTED/BUSY/…)
                continue;
            }
            if (canceled) {
                // Caller abandoned the transfer (releaseReqH): its buffers may
                // be on their way out, so stop touching them; keep consuming
                // the response so the connection stays in sync and reusable.
                if (VerbResult r = drain(d.len, nullptr); r.err != VerbErr::None) return r;
                continue;
            }
            XXH3_state_t st;
            if (verify_) XXH3_64bits_reset(&st);
            if (dst.ptr == nullptr || dst.cap < d.len) {
                // Caller bug: keep the stream in sync (drain the payload),
                // report Usage once at the end; the key reads as a miss.
                if (VerbResult r = drain(d.len, verify_ ? &st : nullptr);
                    r.err != VerbErr::None)
                    return r;
                usage = conn_err(VerbErr::Usage, "GET dest smaller than block");
                continue; // Usage never marks the conn dead; stream stays in sync
            }
            if (VerbResult r = read_exact(dst.ptr, d.len); r.err != VerbErr::None) return r;
            if (verify_) {
                XXH3_64bits_update(&st, dst.ptr, d.len);
                if (XXH3_64bits_digest(&st) != d.xxh3)
                    return conn_err(VerbErr::Verify, "GET xxh3 mismatch (corrupt payload)");
            }
            dst.status = d.status;
            dst.len = d.len;
        }
        seen = first + p.count;
        if (!(rh.flags & kFlagMore)) break;
    }
    if (seen != n) return conn_err(VerbErr::Protocol, "GET response covered fewer keys than sent");
    if (canceled) {
        VerbResult out; // mirrors put(): abandoned by the caller, conn healthy
        out.detail = "canceled";
        return out;
    }
    if (usage.err != VerbErr::None) return usage;
    return {};
}

VerbResult Conn::put(const Key &key, const void *data, size_t len, uint32_t ttl_ms,
                     const std::atomic<bool> *cancel) {
    // total_len is a u32 wire field (§3.4): reject rather than truncate, even
    // when the server forgot to advertise max_blob_len.
    if (len > UINT32_MAX) return conn_err(VerbErr::Usage, "put: blob exceeds u32 wire limit");
    if (hello_.max_blob_len && len > hello_.max_blob_len)
        return conn_err(VerbErr::Usage, "put: blob over negotiated max_blob_len");
    uint64_t sum = xxh3_64(data, len);

    // BEGIN (this client waits for the BEGIN response rather than streaming
    // optimistically — one RTT, in exchange for never wasting chunk bytes on a
    // tombstoned stream; OK_EXISTS then costs nothing).
    std::vector<uint8_t> body;
    append_put_begin(body, static_cast<uint32_t>(len), ttl_ms, sum);
    Header h;
    h.opcode = Op::PutStream;
    h.flags = with_subop(0, kPutBegin);
    h.namespace_id = hello_.namespace_id;
    h.request_id = next_id();
    h.key = key;
    if (VerbResult r = send_frame(h, body.data(), body.size()); r.err != VerbErr::None) return r;

    Header rh;
    for (;;) {
        if (VerbResult r = next_header(rh); r.err != VerbErr::None) return r;
        if (rh.request_id == h.request_id) break;
        if (VerbResult r = drain(rh.payload_len, nullptr); r.err != VerbErr::None) return r;
    }
    std::vector<uint8_t> resp;
    if (VerbResult r = read_body(rh.payload_len, resp); r.err != VerbErr::None) return r;
    Preamble p;
    if (!parse_preamble(resp.data(), resp.size(), p))
        return conn_err(VerbErr::Protocol, "short PUT BEGIN response");
    if (p.status == Status::OKExists) {
        VerbResult r;
        r.wire_status = Status::OKExists; // write-once idempotent hit; stop sending
        return r;
    }
    if (p.status != Status::OK) {
        VerbResult r;
        r.wire_status = p.status;
        return r;
    }

    // CHUNKs: same connection, in order, each ≤ min(4 MiB, max_frame_len).
    size_t cap = hello_.max_frame_len ? hello_.max_frame_len : kPutChunkCap;
    if (cap > kPutChunkCap) cap = kPutChunkCap;
    const uint8_t *pdata = static_cast<const uint8_t *>(data);
    Header ch;
    ch.opcode = Op::PutStream;
    ch.flags = with_subop(0, kPutChunk);
    ch.namespace_id = hello_.namespace_id;
    ch.request_id = h.request_id; // chunks bind to the stream via key+request_id
    ch.key = key;
    bool canceled = false;
    for (size_t off = 0; off < len;) {
        if (cancel && cancel->load(std::memory_order_relaxed)) {
            canceled = true;
            break;
        }
        size_t chunk = len - off < cap ? len - off : cap;
        if (VerbResult r = send_frame(ch, pdata + off, chunk); r.err != VerbErr::None) return r;
        off += chunk;
    }
    // Re-check after the last chunk: a cancel that landed while it was being
    // sent must ABORT rather than COMMIT (releaseReqH means the caller has
    // abandoned the transfer — never publish a block it no longer wants).
    if (!canceled && cancel && cancel->load(std::memory_order_relaxed)) canceled = true;
    if (canceled) {
        // Abandon the stream cleanly: ABORT frees staging immediately and the
        // key never becomes visible (§5). Exactly one response is due.
        Header ab;
        ab.opcode = Op::PutStream;
        ab.flags = with_subop(0, kPutAbort);
        ab.namespace_id = hello_.namespace_id;
        ab.request_id = h.request_id;
        ab.key = key;
        if (VerbResult r = send_frame(ab, nullptr, 0); r.err != VerbErr::None) return r;
        for (;;) {
            if (VerbResult r = next_header(rh); r.err != VerbErr::None) return r;
            if (rh.request_id == h.request_id) break;
            if (VerbResult r = drain(rh.payload_len, nullptr); r.err != VerbErr::None) return r;
        }
        if (VerbResult r = read_body(rh.payload_len, resp); r.err != VerbErr::None) return r;
        if (!parse_preamble(resp.data(), resp.size(), p))
            return conn_err(VerbErr::Protocol, "short PUT ABORT response");
        VerbResult out;
        out.wire_status = p.status;
        out.detail = "canceled";
        return out;
    }

    // COMMIT: authoritative xxh3_64.
    std::vector<uint8_t> cbody;
    append_put_commit(cbody, sum);
    Header cm;
    cm.opcode = Op::PutStream;
    cm.flags = with_subop(0, kPutCommit);
    cm.namespace_id = hello_.namespace_id;
    cm.request_id = h.request_id;
    cm.key = key;
    if (VerbResult r = send_frame(cm, cbody.data(), cbody.size()); r.err != VerbErr::None)
        return r;
    for (;;) {
        if (VerbResult r = next_header(rh); r.err != VerbErr::None) return r;
        if (rh.request_id == h.request_id) break;
        if (VerbResult r = drain(rh.payload_len, nullptr); r.err != VerbErr::None) return r;
    }
    if (VerbResult r = read_body(rh.payload_len, resp); r.err != VerbErr::None) return r;
    if (!parse_preamble(resp.data(), resp.size(), p))
        return conn_err(VerbErr::Protocol, "short PUT COMMIT response");
    VerbResult out;
    out.wire_status = p.status;
    return out;
}

VerbResult Conn::put_abort(const Key &key, const void *data, size_t len, size_t send_prefix) {
    if (len > UINT32_MAX) // u32 total_len, as in put()
        return conn_err(VerbErr::Usage, "put_abort: blob exceeds u32 wire limit");
    std::vector<uint8_t> body;
    append_put_begin(body, static_cast<uint32_t>(len), 0, 0);
    Header h;
    h.opcode = Op::PutStream;
    h.flags = with_subop(0, kPutBegin);
    h.namespace_id = hello_.namespace_id;
    h.request_id = next_id();
    h.key = key;
    if (VerbResult r = send_frame(h, body.data(), body.size()); r.err != VerbErr::None) return r;

    Header rh;
    for (;;) {
        if (VerbResult r = next_header(rh); r.err != VerbErr::None) return r;
        if (rh.request_id == h.request_id) break;
        if (VerbResult r = drain(rh.payload_len, nullptr); r.err != VerbErr::None) return r;
    }
    std::vector<uint8_t> resp;
    if (VerbResult r = read_body(rh.payload_len, resp); r.err != VerbErr::None) return r;
    Preamble p;
    if (!parse_preamble(resp.data(), resp.size(), p))
        return conn_err(VerbErr::Protocol, "short PUT BEGIN response");
    Status begin_status = p.status;

    if (begin_status == Status::OK && send_prefix > 0) {
        // Stream a prefix, then abandon — the mid-transfer abort shape.
        size_t cap = hello_.max_frame_len ? hello_.max_frame_len : kPutChunkCap;
        if (cap > kPutChunkCap) cap = kPutChunkCap;
        size_t limit = send_prefix < len ? send_prefix : len;
        const uint8_t *pdata = static_cast<const uint8_t *>(data);
        Header ch;
        ch.opcode = Op::PutStream;
        ch.flags = with_subop(0, kPutChunk);
        ch.namespace_id = hello_.namespace_id;
        ch.request_id = h.request_id;
        ch.key = key;
        for (size_t off = 0; off < limit;) {
            size_t chunk = limit - off < cap ? limit - off : cap;
            if (VerbResult r = send_frame(ch, pdata + off, chunk); r.err != VerbErr::None)
                return r;
            off += chunk;
        }
    }

    // ABORT: empty payload; exactly one response (OK on a live stream,
    // ERR_STALE_STREAM on a tombstoned one — e.g. after OK_EXISTS BEGIN).
    Header ab;
    ab.opcode = Op::PutStream;
    ab.flags = with_subop(0, kPutAbort);
    ab.namespace_id = hello_.namespace_id;
    ab.request_id = h.request_id;
    ab.key = key;
    if (VerbResult r = send_frame(ab, nullptr, 0); r.err != VerbErr::None) return r;
    for (;;) {
        if (VerbResult r = next_header(rh); r.err != VerbErr::None) return r;
        if (rh.request_id == h.request_id) break;
        if (VerbResult r = drain(rh.payload_len, nullptr); r.err != VerbErr::None) return r;
    }
    if (VerbResult r = read_body(rh.payload_len, resp); r.err != VerbErr::None) return r;
    if (!parse_preamble(resp.data(), resp.size(), p))
        return conn_err(VerbErr::Protocol, "short PUT ABORT response");
    VerbResult out;
    out.wire_status = p.status;
    out.detail = std::string("begin=") + status_name(begin_status);
    return out;
}

VerbResult Conn::key_status_verb(Op op, uint16_t flags, uint32_t aux,
                                 const std::vector<Key> &keys, std::vector<Status> &per_key) {
    per_key.clear();
    std::vector<uint8_t> body;
    append_keylist(body, aux, keys);
    Header h;
    h.opcode = op;
    h.flags = flags;
    h.namespace_id = hello_.namespace_id;
    h.request_id = next_id();
    if (VerbResult r = send_frame(h, body.data(), body.size()); r.err != VerbErr::None) return r;

    Header rh;
    for (;;) {
        if (VerbResult r = next_header(rh); r.err != VerbErr::None) return r;
        if (rh.request_id == h.request_id) break;
        if (VerbResult r = drain(rh.payload_len, nullptr); r.err != VerbErr::None) return r;
    }
    std::vector<uint8_t> resp;
    if (VerbResult r = read_body(rh.payload_len, resp); r.err != VerbErr::None) return r;
    Preamble p;
    if (!parse_preamble(resp.data(), resp.size(), p))
        return conn_err(VerbErr::Protocol, "short key-status response");
    VerbResult out;
    out.wire_status = p.status;
    if (!status_ok(p.status)) return out;
    if (resp.size() != kPreambleSize + pad8(p.count) || p.count != keys.size())
        return conn_err(VerbErr::Protocol, "key-status response length/count mismatch");
    per_key.reserve(p.count);
    for (uint32_t i = 0; i < p.count; i++)
        per_key.push_back(static_cast<Status>(resp[kPreambleSize + i]));
    return out;
}

VerbResult Conn::touch_lease(const std::vector<Key> &keys, uint8_t sub, uint32_t ttl_ms,
                             std::vector<Status> &per_key) {
    return key_status_verb(Op::TouchLease, with_subop(0, sub), ttl_ms, keys, per_key);
}

VerbResult Conn::pin(const std::vector<Key> &keys, uint8_t sub, std::vector<Status> &per_key) {
    return key_status_verb(Op::Pin, with_subop(0, sub), 0, keys, per_key);
}

VerbResult Conn::del(const std::vector<Key> &keys, bool force, std::vector<Status> &per_key) {
    return key_status_verb(Op::Delete, force ? kFlagForce : 0, 0, keys, per_key);
}

VerbResult Conn::stats(std::string &json) {
    json.clear();
    std::vector<uint8_t> body;
    append_stats_req(body, 0);
    Header h;
    h.opcode = Op::Stats;
    h.namespace_id = hello_.namespace_id;
    h.request_id = next_id();
    if (VerbResult r = send_frame(h, body.data(), body.size()); r.err != VerbErr::None) return r;

    Header rh;
    for (;;) {
        if (VerbResult r = next_header(rh); r.err != VerbErr::None) return r;
        if (rh.request_id == h.request_id) break;
        if (VerbResult r = drain(rh.payload_len, nullptr); r.err != VerbErr::None) return r;
    }
    std::vector<uint8_t> resp;
    if (VerbResult r = read_body(rh.payload_len, resp); r.err != VerbErr::None) return r;
    Preamble p;
    if (!parse_preamble(resp.data(), resp.size(), p))
        return conn_err(VerbErr::Protocol, "short STATS response");
    VerbResult out;
    out.wire_status = p.status;
    if (!status_ok(p.status)) return out;
    size_t take = p.count;
    if (kPreambleSize + take > resp.size()) take = resp.size() - kPreambleSize;
    json.assign(reinterpret_cast<const char *>(resp.data() + kPreambleSize), take);
    return out;
}

// --- Pool ----------------------------------------------------------------------

Pool::Pool(Conn::Options opt, size_t streams)
    : opt_(std::move(opt)), streams_(streams == 0 ? 1 : streams) {}

Pool::~Pool() { close(); }

VerbResult Pool::prime() {
    auto c = std::make_unique<Conn>();
    VerbResult r = c->connect(opt_);
    if (r.err != VerbErr::None || !r.ok()) return r;
    limits_ = c->limits();
    checkin(std::move(c));
    // prime() is called before any checkout, so outstanding_ was never
    // incremented for this conn; checkin() bumps idle_ only.
    return r;
}

VerbResult Pool::checkout(std::unique_ptr<Conn> &out) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return closed_ || !idle_.empty() || outstanding_ < streams_; });
    if (closed_) {
        VerbResult r;
        r.err = VerbErr::Connection;
        r.detail = "pool closed";
        return r;
    }
    if (!idle_.empty()) {
        out = std::move(idle_.back());
        idle_.pop_back();
        outstanding_++;
        return {};
    }
    outstanding_++; // reserve the slot, dial outside the lock
    lk.unlock();
    auto c = std::make_unique<Conn>();
    VerbResult r = c->connect(opt_);
    if (r.err != VerbErr::None || !r.ok()) {
        std::lock_guard<std::mutex> lg(mu_);
        outstanding_--;
        cv_.notify_one();
        if (r.err == VerbErr::None) r.err = VerbErr::Connection; // HELLO rejection
        return r;
    }
    out = std::move(c);
    return {};
}

void Pool::checkin(std::unique_ptr<Conn> c) {
    std::lock_guard<std::mutex> lg(mu_);
    if (outstanding_ > 0) outstanding_--;
    // A dead conn is dropped here and lazily replaced on the next checkout —
    // the python pool's discard rule (unknown stream state is never reused).
    if (!closed_ && c && !c->dead()) idle_.push_back(std::move(c));
    cv_.notify_one();
}

void Pool::close() {
    std::lock_guard<std::mutex> lg(mu_);
    closed_ = true;
    idle_.clear();
    cv_.notify_all();
}

} // namespace kvb
