// kvblockd_backend.h — NIXL storage backend engine for kvblockd (beta).
//
// Follows the GDS/POSIX/obj local-storage pattern: supportsLocal-only, DRAM
// and OBJ segments, prep-once/repost-many request handles, non-blocking
// postXfer (work rides the kvb::Executor thread pool), pollable checkXfer,
// non-blocking releaseReqH (abort via a shared flag; in-flight PUT streams
// are ABORTed at the next chunk boundary so server staging is freed; reads
// stop writing caller memory at the next descriptor boundary). The abort is
// NOT a quiesce — registered buffers must outlive the engine object, whose
// destructor joins all in-flight tasks (README caveat 3).
//
// Descriptor convention (matches the obj plugin):
//   local  side: DRAM_SEG descs — addr/len of registered host memory.
//   remote side: OBJ_SEG descs — metaInfo carries the 32-byte kvblockd block
//     key, either as 32 raw bytes or as 64 hex characters. The key is opaque
//     to the server (T3) and computed by whoever owns the key schema (KVBM /
//     the caller); this backend never derives keys. addr MUST be 0: kvblockd
//     blocks are write-once whole objects, there are no ranged writes, and v1
//     of this backend does not implement ranged reads.
//
//   WRITE = PUT_STREAM BEGIN->CHUNK*->COMMIT per block (xxh3 computed while
//   streaming); READ = BATCH_GET tiled across the connection pool with xxh3
//   verification into the registered memory.
//
// Init params (see get_backend_options in kvblockd_plugin.cpp for defaults):
//   endpoint         "host:port" of the kvblockd data listener   (required)
//   namespace        tenant namespace name                        (required)
//   token            bearer token for that namespace              (required)
//   num_connections  pool size, striped per §7 lanes              (default 16)
//   verify_reads     xxh3-verify GET payloads ("true"/"false")    (default true)
//   put_ttl_ms       TTL passed on PUT BEGIN (0 = namespace default)
//   op_timeout_ms    per-syscall socket timeout                   (default 30000)

#ifndef KVBLOCKD_BACKEND_H
#define KVBLOCKD_BACKEND_H

#include "backend/backend_engine.h"

#include "executor.h"
#include "kvblockd_client.h"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

// Registered-memory record. DRAM: the registered [base, base+len) window every
// transfer descriptor must fall inside. OBJ: the parsed 32-byte block key.
class nixlKvblockdMD : public nixlBackendMD {
  public:
    explicit nixlKvblockdMD(bool is_private) : nixlBackendMD(is_private) {}
    ~nixlKvblockdMD() override = default;

    nixl_mem_t memType = DRAM_SEG;
    uintptr_t base = 0; // DRAM only
    size_t len = 0;     // DRAM only
    kvb::Key key{};     // OBJ only
};

// One block move: DRAM region <-> keyed block.
struct kvbXferItem {
    void *ptr = nullptr;
    size_t len = 0;
    kvb::Key key{};
};

// Shared transfer state. The handle, the executor tasks, and nothing else
// hold references; releaseReqH can therefore drop the handle mid-flight
// (abort flag set) without racing task completion — the last reference frees
// the state (the reason this is shared_ptr-held rather than handle-owned).
struct kvbXferState {
    // Lifecycle: PREPPED --post--> INFLIGHT --last task--> DONE | ERR
    // (repost allowed from PREPPED/DONE/ERR; never from INFLIGHT — one
    // in-flight transfer per handle).
    enum Phase : int { PREPPED = 0, INFLIGHT = 1, DONE = 2, ERR = 3 };

    std::atomic<int> phase{PREPPED};
    std::atomic<uint32_t> remaining{0}; // unfinished tiles for the current post
    std::atomic<bool> aborted{false};
    std::atomic<int> first_err{0}; // kvb::VerbErr, or 100+wire-status for verb failures

    nixl_xfer_op_t op = NIXL_WRITE;
    uint32_t put_ttl_ms = 0;
    // Items tiled round-robin across pool connections at prep time.
    std::vector<std::vector<kvbXferItem>> tiles;

    void record_err(int code) {
        int expected = 0;
        first_err.compare_exchange_strong(expected, code);
    }
};

class nixlKvblockdReqH : public nixlBackendReqH {
  public:
    explicit nixlKvblockdReqH(std::shared_ptr<kvbXferState> s) : state(std::move(s)) {}
    ~nixlKvblockdReqH() override = default;
    std::shared_ptr<kvbXferState> state;
};

class nixlKvblockdEngine : public nixlBackendEngine {
  public:
    explicit nixlKvblockdEngine(const nixlBackendInitParams *init_params);
    ~nixlKvblockdEngine() override;

    bool supportsRemote() const override { return false; }
    bool supportsLocal() const override { return true; }
    bool supportsNotif() const override { return false; }
    nixl_mem_list_t getSupportedMems() const override { return {DRAM_SEG, OBJ_SEG}; }

    nixl_status_t registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem,
                              nixlBackendMD *&out) override;
    nixl_status_t deregisterMem(nixlBackendMD *meta) override;

    // Local storage engine: connections are internal (the wire pool), so the
    // agent-level connect/disconnect are trivially satisfied (GDS pattern).
    nixl_status_t connect(const std::string & /*remote_agent*/) override { return NIXL_SUCCESS; }
    nixl_status_t disconnect(const std::string & /*remote_agent*/) override {
        return NIXL_SUCCESS;
    }
    nixl_status_t unloadMD(nixlBackendMD * /*input*/) override { return NIXL_SUCCESS; }
    nixl_status_t loadLocalMD(nixlBackendMD *input, nixlBackendMD *&output) override {
        output = input; // pass-through: local MD is already complete
        return NIXL_SUCCESS;
    }

    nixl_status_t prepXfer(const nixl_xfer_op_t &operation, const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote, const std::string &remote_agent,
                           nixlBackendReqH *&handle,
                           const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t postXfer(const nixl_xfer_op_t &operation, const nixl_meta_dlist_t &local,
                           const nixl_meta_dlist_t &remote, const std::string &remote_agent,
                           nixlBackendReqH *&handle,
                           const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t checkXfer(nixlBackendReqH *handle) const override;
    nixl_status_t releaseReqH(nixlBackendReqH *handle) const override;

  private:
    void run_write_tile(std::shared_ptr<kvbXferState> st, size_t tile) const;
    void run_read_tile(std::shared_ptr<kvbXferState> st, size_t tile) const;

    std::unique_ptr<kvb::Pool> pool_;
    std::unique_ptr<kvb::Executor> executor_;
    uint32_t put_ttl_ms_ = 0;
    size_t num_connections_ = 16;
};

#endif // KVBLOCKD_BACKEND_H
