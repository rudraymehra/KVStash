// kvblockd_backend.cpp — see kvblockd_backend.h.

#include "kvblockd_backend.h"

#include <cstdio>
#include <cstdlib>

namespace {

// Parse the OBJ descriptor metaInfo into a 32-byte wire key: either 32 raw
// bytes or 64 hex characters (the s3compat object-key convention).
bool parse_key(const std::string &mi, kvb::Key &out) {
    if (mi.size() == 32) {
        std::memcpy(out.data(), mi.data(), 32);
        return true;
    }
    if (mi.size() == 64) {
        auto nib = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        for (size_t i = 0; i < 32; i++) {
            int hi = nib(mi[2 * i]), lo = nib(mi[2 * i + 1]);
            if (hi < 0 || lo < 0) return false;
            out[i] = static_cast<uint8_t>(hi << 4 | lo);
        }
        return true;
    }
    return false;
}

bool parse_bool(const std::string &s, bool dflt) {
    if (s == "true" || s == "1" || s == "yes") return true;
    if (s == "false" || s == "0" || s == "no") return false;
    return dflt;
}

void debug_log(const char *msg, const std::string &detail) {
    if (std::getenv("KVB_NIXL_DEBUG") != nullptr)
        std::fprintf(stderr, "[kvblockd-nixl] %s: %s\n", msg, detail.c_str());
}

constexpr int kWireErrBase = 100; // first_err encoding: 100 + wire Status

} // namespace

nixlKvblockdEngine::nixlKvblockdEngine(const nixlBackendInitParams *init_params)
    : nixlBackendEngine(init_params) {
    std::string endpoint, ns, token, v;

    if (getInitParam("endpoint", endpoint) != NIXL_SUCCESS || endpoint.empty() ||
        getInitParam("namespace", ns) != NIXL_SUCCESS || ns.empty() ||
        getInitParam("token", token) != NIXL_SUCCESS) {
        debug_log("init", "missing required param (endpoint/namespace/token)");
        initErr = true;
        return;
    }

    kvb::Conn::Options opt;
    size_t colon = endpoint.rfind(':');
    if (colon == std::string::npos || colon + 1 >= endpoint.size()) {
        debug_log("init", "endpoint must be host:port, got " + endpoint);
        initErr = true;
        return;
    }
    opt.host = endpoint.substr(0, colon);
    opt.port = static_cast<uint16_t>(std::strtoul(endpoint.c_str() + colon + 1, nullptr, 10));
    opt.ns = ns;
    opt.token = token;
    opt.client_name = "kvblockd-nixl/" + localAgent;

    if (getInitParam("num_connections", v) == NIXL_SUCCESS && !v.empty()) {
        long n = std::strtol(v.c_str(), nullptr, 10);
        if (n >= 1 && n <= 64) num_connections_ = static_cast<size_t>(n);
    }
    if (getInitParam("verify_reads", v) == NIXL_SUCCESS && !v.empty())
        opt.verify_reads = parse_bool(v, true);
    if (getInitParam("put_ttl_ms", v) == NIXL_SUCCESS && !v.empty())
        put_ttl_ms_ = static_cast<uint32_t>(std::strtoul(v.c_str(), nullptr, 10));
    if (getInitParam("op_timeout_ms", v) == NIXL_SUCCESS && !v.empty()) {
        long t = std::strtol(v.c_str(), nullptr, 10);
        if (t > 0) opt.op_timeout_ms = static_cast<int>(t);
    }

    pool_ = std::make_unique<kvb::Pool>(opt, num_connections_);
    kvb::VerbResult r = pool_->prime();
    if (!r.ok()) {
        debug_log("HELLO failed", r.detail + " status=" + kvb::status_name(r.wire_status));
        initErr = true;
        return;
    }
    executor_ = std::make_unique<kvb::Executor>(num_connections_);
}

nixlKvblockdEngine::~nixlKvblockdEngine() {
    // Drain in-flight tasks first (their conns must outlive them), then the pool.
    if (executor_) executor_->shutdown();
    if (pool_) pool_->close();
}

nixl_status_t nixlKvblockdEngine::registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem,
                                              nixlBackendMD *&out) {
    out = nullptr;
    if (nixl_mem == DRAM_SEG) {
        auto *md = new nixlKvblockdMD(true);
        md->memType = DRAM_SEG;
        md->base = mem.addr;
        md->len = mem.len;
        out = md;
        return NIXL_SUCCESS;
    }
    if (nixl_mem == OBJ_SEG) {
        auto *md = new nixlKvblockdMD(true);
        md->memType = OBJ_SEG;
        if (!parse_key(mem.metaInfo, md->key)) {
            delete md;
            debug_log("registerMem", "OBJ metaInfo must be 32 raw bytes or 64 hex chars");
            return NIXL_ERR_INVALID_PARAM;
        }
        out = md;
        return NIXL_SUCCESS;
    }
    return NIXL_ERR_NOT_SUPPORTED;
}

nixl_status_t nixlKvblockdEngine::deregisterMem(nixlBackendMD *meta) {
    delete static_cast<nixlKvblockdMD *>(meta);
    return NIXL_SUCCESS;
}

nixl_status_t nixlKvblockdEngine::prepXfer(const nixl_xfer_op_t &operation,
                                           const nixl_meta_dlist_t &local,
                                           const nixl_meta_dlist_t &remote,
                                           const std::string &remote_agent,
                                           nixlBackendReqH *&handle,
                                           const nixl_opt_b_args_t *opt_args) const {
    (void)remote_agent;
    (void)opt_args;
    handle = nullptr;
    if (operation != NIXL_READ && operation != NIXL_WRITE) return NIXL_ERR_INVALID_PARAM;
    if (local.getType() != DRAM_SEG || remote.getType() != OBJ_SEG) {
        debug_log("prepXfer", "expected local=DRAM_SEG, remote=OBJ_SEG");
        return NIXL_ERR_INVALID_PARAM;
    }
    const int n = local.descCount();
    if (n <= 0 || remote.descCount() != n) return NIXL_ERR_INVALID_PARAM;

    auto st = std::make_shared<kvbXferState>();
    st->op = operation;
    st->put_ttl_ms = put_ttl_ms_;
    size_t n_tiles = num_connections_ < static_cast<size_t>(n) ? num_connections_
                                                               : static_cast<size_t>(n);
    if (n_tiles == 0) n_tiles = 1;
    st->tiles.resize(n_tiles);

    for (int i = 0; i < n; i++) {
        const nixlMetaDesc &ld = local[i];
        const nixlMetaDesc &rd = remote[i];
        const auto *rmd = static_cast<const nixlKvblockdMD *>(rd.metadataP);
        if (rmd == nullptr || rmd->memType != OBJ_SEG) {
            debug_log("prepXfer", "remote desc lacks a registered OBJ key");
            return NIXL_ERR_INVALID_PARAM;
        }
        if (rd.addr != 0) {
            // kvblockd blocks are write-once whole objects; ranged access is
            // not part of the v1 backend (documented divergence, README).
            debug_log("prepXfer", "OBJ desc addr (offset) must be 0");
            return NIXL_ERR_NOT_SUPPORTED;
        }
        if (ld.len > UINT32_MAX) return NIXL_ERR_INVALID_PARAM;
        const auto *lmd = static_cast<const nixlKvblockdMD *>(ld.metadataP);
        if (lmd != nullptr && lmd->memType == DRAM_SEG && lmd->len > 0) {
            // Bounds-check against the registered window when we have it —
            // subtraction form so a hostile addr/len pair cannot wrap
            // uintptr_t and slip past an additive check.
            if (ld.addr < lmd->base || ld.len > lmd->len ||
                ld.addr - lmd->base > lmd->len - ld.len) {
                debug_log("prepXfer", "local desc outside its registered region");
                return NIXL_ERR_INVALID_PARAM;
            }
        }
        kvbXferItem item;
        item.ptr = reinterpret_cast<void *>(ld.addr);
        item.len = ld.len;
        item.key = rmd->key;
        st->tiles[static_cast<size_t>(i) % n_tiles].push_back(item);
    }

    handle = new nixlKvblockdReqH(std::move(st));
    return NIXL_SUCCESS;
}

nixl_status_t nixlKvblockdEngine::postXfer(const nixl_xfer_op_t &operation,
                                           const nixl_meta_dlist_t &local,
                                           const nixl_meta_dlist_t &remote,
                                           const std::string &remote_agent,
                                           nixlBackendReqH *&handle,
                                           const nixl_opt_b_args_t *opt_args) const {
    (void)local;
    (void)remote;
    (void)remote_agent;
    (void)opt_args;
    auto *h = static_cast<nixlKvblockdReqH *>(handle);
    if (h == nullptr || !h->state) return NIXL_ERR_INVALID_PARAM;
    std::shared_ptr<kvbXferState> st = h->state;
    if (operation != st->op) return NIXL_ERR_INVALID_PARAM; // handles are op-specific
    if (!executor_) return NIXL_ERR_BACKEND;

    // One in-flight transfer per handle; repost allowed only once terminal.
    int phase = st->phase.load(std::memory_order_acquire);
    for (;;) {
        if (phase == kvbXferState::INFLIGHT) return NIXL_ERR_REPOST_ACTIVE;
        if (st->phase.compare_exchange_weak(phase, kvbXferState::INFLIGHT,
                                            std::memory_order_acq_rel))
            break;
    }
    st->aborted.store(false, std::memory_order_relaxed);
    st->first_err.store(0, std::memory_order_relaxed);
    st->remaining.store(static_cast<uint32_t>(st->tiles.size()), std::memory_order_release);

    for (size_t t = 0; t < st->tiles.size(); t++) {
        const nixlKvblockdEngine *self = this;
        bool queued = executor_->submit([self, st, t] {
            try {
                if (!st->aborted.load(std::memory_order_relaxed)) {
                    if (st->op == NIXL_WRITE)
                        self->run_write_tile(st, t);
                    else
                        self->run_read_tile(st, t);
                }
            } catch (...) {
                // Executor tasks are noexcept-by-contract (nothing may unwind
                // across the worker loop — that would std::terminate the host
                // inference process). A std::bad_alloc from a tile's buffers
                // lands here and becomes a backend error on this transfer.
                st->record_err(kWireErrBase + static_cast<int>(kvb::Status::ErrInternal));
            }
            if (st->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                int err = st->first_err.load(std::memory_order_relaxed);
                st->phase.store(err ? kvbXferState::ERR : kvbXferState::DONE,
                                std::memory_order_release);
            }
        });
        if (!queued) {
            // Executor is shutting down: account for the never-queued tile so
            // the transfer still reaches a terminal phase.
            st->record_err(static_cast<int>(kvb::VerbErr::Connection));
            if (st->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                st->phase.store(kvbXferState::ERR, std::memory_order_release);
            }
        }
    }
    return NIXL_IN_PROG;
}

void nixlKvblockdEngine::run_write_tile(std::shared_ptr<kvbXferState> st, size_t tile) const {
    for (const kvbXferItem &item : st->tiles[tile]) {
        if (st->aborted.load(std::memory_order_relaxed)) return; // unstarted blocks: nothing to undo
        kvb::VerbResult r = pool_->with_conn([&](kvb::Conn &c) {
            return c.put(item.key, item.ptr, item.len, st->put_ttl_ms, &st->aborted);
        });
        if (r.err != kvb::VerbErr::None) {
            st->record_err(static_cast<int>(r.err));
            debug_log("PUT", r.detail);
            return;
        }
        if (r.detail == "canceled") return; // aborted mid-stream; server staging freed
        if (!kvb::status_ok(r.wire_status)) {
            st->record_err(kWireErrBase + static_cast<int>(r.wire_status));
            debug_log("PUT status", kvb::status_name(r.wire_status));
            return;
        }
    }
}

void nixlKvblockdEngine::run_read_tile(std::shared_ptr<kvbXferState> st, size_t tile) const {
    const std::vector<kvbXferItem> &items = st->tiles[tile];
    uint32_t cap = pool_->limits().max_batch_keys;
    if (cap == 0) cap = 512;

    for (size_t base = 0; base < items.size(); base += cap) {
        // Reads leave no server state; batch_get also watches the flag at
        // every descriptor boundary (drain instead of write once aborted).
        if (st->aborted.load(std::memory_order_relaxed)) return;
        size_t n = items.size() - base < cap ? items.size() - base : cap;
        std::vector<kvb::Key> keys(n);
        std::vector<kvb::GetDest> dests(n);
        for (size_t i = 0; i < n; i++) {
            keys[i] = items[base + i].key;
            dests[i].ptr = items[base + i].ptr;
            dests[i].cap = items[base + i].len;
        }
        kvb::VerbResult r = pool_->with_conn(
            [&](kvb::Conn &c) { return c.batch_get(keys, dests, &st->aborted); });
        if (r.err != kvb::VerbErr::None) {
            st->record_err(static_cast<int>(r.err));
            debug_log("GET", r.detail);
            return;
        }
        // Aborted mid-batch (releaseReqH): the client stopped writing caller
        // memory at the next descriptor boundary and drained the rest, so the
        // conn is healthy — but dests are partial; skip their validation.
        if (r.detail == "canceled") return;
        if (!kvb::status_ok(r.wire_status)) {
            st->record_err(kWireErrBase + static_cast<int>(r.wire_status));
            return;
        }
        for (size_t i = 0; i < n; i++) {
            if (!kvb::status_ok(dests[i].status)) {
                // A miss on a requested block is a failed transfer, not a
                // partial success — NIXL transfers are all-or-nothing.
                st->record_err(kWireErrBase + static_cast<int>(dests[i].status));
                return;
            }
            if (dests[i].len != items[base + i].len) {
                // NIXL descs are exact-size; a length drift means the caller's
                // view of the block and the store disagree.
                st->record_err(kWireErrBase + static_cast<int>(kvb::Status::ErrMalformed));
                return;
            }
        }
    }
}

nixl_status_t nixlKvblockdEngine::checkXfer(nixlBackendReqH *handle) const {
    auto *h = static_cast<nixlKvblockdReqH *>(handle);
    if (h == nullptr || !h->state) return NIXL_ERR_INVALID_PARAM;
    switch (h->state->phase.load(std::memory_order_acquire)) {
    case kvbXferState::PREPPED:
        return NIXL_ERR_NOT_POSTED;
    case kvbXferState::INFLIGHT:
        return NIXL_IN_PROG;
    case kvbXferState::DONE:
        return NIXL_SUCCESS;
    default: {
        int err = h->state->first_err.load(std::memory_order_relaxed);
        if (err == static_cast<int>(kvb::VerbErr::Connection)) return NIXL_ERR_REMOTE_DISCONNECT;
        if (err == kWireErrBase + static_cast<int>(kvb::Status::NotFound) ||
            err == kWireErrBase + static_cast<int>(kvb::Status::Evicted))
            return NIXL_ERR_NOT_FOUND;
        return NIXL_ERR_BACKEND;
    }
    }
}

nixl_status_t nixlKvblockdEngine::releaseReqH(nixlBackendReqH *handle) const {
    auto *h = static_cast<nixlKvblockdReqH *>(handle);
    if (h == nullptr) return NIXL_ERR_INVALID_PARAM;
    if (h->state) {
        // Non-blocking abort: in-flight PUT streams see the flag at the next
        // chunk boundary (and once more before COMMIT) and send ABORT (server
        // staging freed, key never visible); reads stop writing caller memory
        // at the next descriptor boundary and drain the rest. Tasks keep the
        // state alive via their shared_ptr — deleting the handle now is safe.
        // NOT a quiesce: a tile mid-syscall may still touch registered memory
        // briefly after this returns (README caveat 3 — buffers must outlive
        // the engine; full quiescence-on-release is a GA follow-up).
        h->state->aborted.store(true, std::memory_order_relaxed);
    }
    delete h;
    return NIXL_SUCCESS;
}
