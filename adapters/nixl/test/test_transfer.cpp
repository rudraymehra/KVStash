// test_transfer.cpp — the Week-11 gate: 32 × 2.5 MB write -> read -> verify ->
// abort against a REAL kvblockd daemon that this harness builds/spawns itself
// (the W5/W12 conftest pattern: temp config, fresh ports, /healthz wait,
// SIGTERM teardown).
//
// Two flavors from one file:
//   * core (always built): drives the shared C++ client core (kvb::Pool +
//     kvb::Executor) through all 8 verbs — the wire-level gate. The daemon
//     config pins initial_credit at the §4 floor (16 MiB) so a 24 MB PUT
//     genuinely exhausts the credit window and exercises the ledger's
//     wait-for-replenishment path, not just the happy path.
//   * KVB_HAVE_NIXL (CI, -Dnixl_path/-Dnixl_build_path given): additionally
//     drives the nixlKvblockdEngine lifecycle exactly as the NIXL agent would
//     (registerMem -> prepXfer -> postXfer -> checkXfer poll -> releaseReqH),
//     including a mid-flight releaseReqH abort — the daemon must survive and
//     keep serving.
//
// Environment:
//   KVB_DAEMON      path to a prebuilt kvblockd binary (preferred)
//   KVB_REPO_ROOT   repo root for a `go build ./cmd/kvblockd` fallback
// Exit 77 (meson SKIP) when neither yields a daemon.

#include "executor.h"
#include "kvblockd_client.h"

#ifdef KVB_HAVE_NIXL
#include "kvblockd_backend.h"
#endif

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <csignal>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace kvb;

static int g_failures = 0;

// T_-prefixed: absl's log/check.h also defines CHECK and the nixl flavor
// pulls it in transitively. Conditions are evaluated exactly ONCE — several
// checked expressions (postXfer, registerMem) have side effects.
#define T_CHECK(cond, ...)                                                                      \
    do {                                                                                        \
        if (!(cond)) {                                                                          \
            g_failures++;                                                                       \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                                    \
            std::printf(__VA_ARGS__);                                                           \
            std::printf("\n");                                                                  \
        }                                                                                       \
    } while (0)

#define T_REQUIRE(cond, ...)                                                                    \
    do {                                                                                        \
        const bool t_req_ok_ = static_cast<bool>(cond);                                         \
        T_CHECK(t_req_ok_, __VA_ARGS__);                                                        \
        if (!t_req_ok_) return false;                                                           \
    } while (0)

namespace {

constexpr size_t kBlockLen = 2621440; // 2.5 MB, the upper end of the 0.4-2.5 MB band
constexpr size_t kNumBlocks = 32;
constexpr size_t kPoolStreams = 8;

// --- deterministic incompressible payloads (xxh3-seeded splitmix64) ----------

uint64_t splitmix64(uint64_t &s) {
    s += 0x9E3779B97F4A7C15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

void fill_block(uint8_t *dst, size_t len, uint64_t seed_input) {
    // Seed each block from xxh3 (the "xxh3-derived" requirement), then expand
    // with splitmix64 — fast, deterministic, incompressible.
    uint64_t seed = xxh3_64(&seed_input, sizeof seed_input);
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        uint64_t w = splitmix64(seed);
        std::memcpy(dst + i, &w, 8);
    }
    if (i < len) {
        uint64_t w = splitmix64(seed);
        std::memcpy(dst + i, &w, len - i);
    }
}

Key make_key(const char *tag, size_t i) {
    // Test keys are opaque 32-byte values (the server never inspects them —
    // T3); derive them from xxh3 so runs are deterministic.
    Key k;
    for (size_t w = 0; w < 4; w++) {
        std::string s = std::string(tag) + "/" + std::to_string(i) + "/" + std::to_string(w);
        uint64_t h = xxh3_64(s.data(), s.size());
        std::memcpy(k.data() + w * 8, &h, 8);
    }
    return k;
}

// --- daemon harness -----------------------------------------------------------

uint16_t free_port() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;
    ::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof addr);
    socklen_t len = sizeof addr;
    ::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len);
    uint16_t port = ntohs(addr.sin_port);
    ::close(fd);
    return port;
}

bool healthz_ok(uint16_t port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(port);
    if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof addr) != 0) {
        ::close(fd);
        return false;
    }
    const char *req = "GET /healthz HTTP/1.0\r\nHost: localhost\r\n\r\n";
    (void)!::send(fd, req, std::strlen(req), 0);
    char buf[64] = {0};
    ssize_t n = ::recv(fd, buf, sizeof buf - 1, 0);
    ::close(fd);
    return n > 0 && std::strstr(buf, " 200") != nullptr;
}

struct Daemon {
    pid_t pid = -1;
    uint16_t data_port = 0;
    uint16_t metrics_port = 0;
    std::filesystem::path dir;

    // RAII teardown: every exit path (early T_REQUIRE returns from main
    // included) must reap the daemon and remove the tempdir — a start() that
    // timed out on /healthz has ALREADY forked a live process.
    ~Daemon() { stop(); }

    void remove_dir() {
        if (dir.empty()) return;
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        dir.clear();
    }

    bool start() {
        char tmpl[] = "/tmp/kvb-nixl-XXXXXX";
        const char *d = ::mkdtemp(tmpl);
        if (d == nullptr) return false;
        dir = d;

        std::string bin;
        if (const char *env = std::getenv("KVB_DAEMON"); env && *env) {
            bin = env;
        } else {
            const char *root = std::getenv("KVB_REPO_ROOT");
            if (root == nullptr || !*root) {
                std::printf("SKIP: neither KVB_DAEMON nor KVB_REPO_ROOT set\n");
                remove_dir(); // std::exit skips destructors
                std::exit(77);
            }
            bin = (dir / "kvblockd").string();
            std::string cmd = "cd '" + std::string(root) + "' && go build -o '" + bin +
                              "' ./cmd/kvblockd";
            if (std::system(cmd.c_str()) != 0) {
                std::printf("SKIP: go build ./cmd/kvblockd failed (toolchain missing?)\n");
                remove_dir(); // std::exit skips destructors
                std::exit(77);
            }
        }

        data_port = free_port();
        metrics_port = free_port();
        {
            std::ofstream ns(dir / "ns.yaml");
            ns << "namespaces:\n  - { name: t, id: 7, token: sekret }\n";
        }
        {
            std::ofstream cfg(dir / "cfg.yaml");
            cfg << "listen_addr: \"127.0.0.1:" << data_port << "\"\n"
                << "metrics_addr: \"127.0.0.1:" << metrics_port << "\"\n"
                << "dram_arena_bytes: 536870912\n" // 512 MiB: 2×80 MB batches + staging
                << "pinned_bytes_cap: 67108864\n"
                << "initial_credit: 16777216\n" // the §4 floor — forces credit waits
                << "namespaces_path: \"" << (dir / "ns.yaml").string() << "\"\n";
        }

        pid = ::fork();
        if (pid < 0) {
            remove_dir(); // no child to reap; don't leak the tempdir
            return false;
        }
        if (pid == 0) {
            // Child: daemon output -> log file for post-mortem upload.
            FILE *log = std::fopen((dir / "daemon.log").c_str(), "w");
            if (log != nullptr) {
                ::dup2(fileno(log), 1);
                ::dup2(fileno(log), 2);
            }
            std::string cfgp = (dir / "cfg.yaml").string();
            ::execlp(bin.c_str(), bin.c_str(), "-config", cfgp.c_str(),
                     static_cast<char *>(nullptr));
            std::_Exit(127);
        }
        for (int i = 0; i < 150; i++) { // 15 s
            if (healthz_ok(metrics_port)) return true;
            if (::waitpid(pid, nullptr, WNOHANG) == pid) {
                pid = -1; // child already exited; nothing to kill
                dump_log();
                stop();
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        // Startup timeout: the daemon is STILL RUNNING — reap it (and the
        // tempdir) here or it outlives the test as an orphan.
        dump_log();
        stop();
        return false;
    }

    bool alive() const { return pid > 0 && ::waitpid(pid, nullptr, WNOHANG) == 0; }

    void dump_log() const {
        std::ifstream log(dir / "daemon.log");
        if (!log) return;
        std::string line;
        std::printf("--- daemon.log ---\n");
        int n = 0;
        while (std::getline(log, line) && n++ < 50) std::printf("%s\n", line.c_str());
    }

    void stop() { // idempotent: safe from every failure path and the dtor
        if (pid > 0) {
            ::kill(pid, SIGTERM);
            for (int i = 0; i < 50; i++) {
                if (::waitpid(pid, nullptr, WNOHANG) == pid) {
                    pid = -1;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (pid > 0) {
                ::kill(pid, SIGKILL);
                ::waitpid(pid, nullptr, 0);
                pid = -1;
            }
        }
        remove_dir();
    }
};

// --- core (wire-level) phases ----------------------------------------------------

struct Ctx {
    Pool *pool = nullptr;
    std::vector<Key> keys;
    std::vector<uint8_t> src; // kNumBlocks × kBlockLen
    std::vector<uint8_t> dst;
};

bool phase_parallel_put(Ctx &c) {
    // Tile the 32 blocks across executor workers the way the backend does.
    Executor ex(kPoolStreams);
    std::atomic<uint32_t> remaining{kPoolStreams};
    std::atomic<int> errors{0};
    for (size_t t = 0; t < kPoolStreams; t++) {
        ex.submit([&, t] {
            for (size_t i = t; i < kNumBlocks; i += kPoolStreams) {
                VerbResult r = c.pool->with_conn([&](Conn &cn) {
                    return cn.put(c.keys[i], c.src.data() + i * kBlockLen, kBlockLen);
                });
                if (!r.ok()) {
                    errors.fetch_add(1);
                    std::printf("put[%zu]: err=%d status=%s %s\n", i, int(r.err),
                                status_name(r.wire_status), r.detail.c_str());
                }
            }
            remaining.fetch_sub(1);
        });
    }
    while (remaining.load() != 0) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    ex.shutdown();
    T_REQUIRE(errors.load() == 0, "parallel PUT had %d failures", errors.load());
    return true;
}

bool phase_exists(Ctx &c) {
    uint32_t n_consec = 0;
    std::vector<Status> per_key;
    VerbResult r = c.pool->with_conn(
        [&](Conn &cn) { return cn.batch_exists(c.keys, n_consec, &per_key); });
    T_REQUIRE(r.ok(), "EXISTS failed: %s", r.detail.c_str());
    T_REQUIRE(n_consec == kNumBlocks, "n_consecutive=%u want %zu", n_consec, kNumBlocks);
    T_REQUIRE(per_key.size() == kNumBlocks, "bitmap size %zu", per_key.size());
    for (size_t i = 0; i < kNumBlocks; i++)
        T_CHECK(status_ok(per_key[i]), "per_key[%zu]=%s", i, status_name(per_key[i]));
    return true;
}

bool phase_get_verify(Ctx &c) {
    // One BATCH_GET (single conn) — xxh3-verified by the client, then
    // byte-compared against the source here.
    std::vector<GetDest> dests(kNumBlocks);
    for (size_t i = 0; i < kNumBlocks; i++) {
        dests[i].ptr = c.dst.data() + i * kBlockLen;
        dests[i].cap = kBlockLen;
    }
    VerbResult r = c.pool->with_conn([&](Conn &cn) { return cn.batch_get(c.keys, dests); });
    T_REQUIRE(r.ok(), "GET failed: err=%d %s", int(r.err), r.detail.c_str());
    for (size_t i = 0; i < kNumBlocks; i++) {
        T_REQUIRE(dests[i].status == Status::OK, "GET[%zu] status=%s", i,
                status_name(dests[i].status));
        T_REQUIRE(dests[i].len == kBlockLen, "GET[%zu] len=%u", i, dests[i].len);
    }
    T_REQUIRE(std::memcmp(c.src.data(), c.dst.data(), c.src.size()) == 0,
            "read-back bytes differ from written bytes");
    return true;
}

bool phase_undersized_dest(const Daemon &d) {
    // Descriptor-clobber PoC (review F1): two stored blocks, dest[0] deliberately
    // undersized. Draining the oversized payload reuses the connection's scratch
    // buffer — with a lazily-parsed descriptor table the drain overwrote desc[1]
    // before it was read, corrupting key 1's status/len and silently desyncing
    // the stream. Contract under test: batch_get returns Usage, dest[0] reads as
    // a miss, dest[1] still lands intact and byte-correct, and the SAME
    // connection stays in sync (services a follow-up verb).
    constexpr size_t kLen = 131072; // 128 KiB: the drain clobbers far past the 40B table
    std::vector<uint8_t> blk0(kLen), blk1(kLen);
    fill_block(blk0.data(), kLen, 0xF10);
    fill_block(blk1.data(), kLen, 0xF11);
    std::vector<Key> keys{make_key("clobber", 0), make_key("clobber", 1)};

    Conn::Options opt;
    opt.host = "127.0.0.1";
    opt.port = d.data_port;
    opt.ns = "t";
    opt.token = "sekret";
    Conn cn;
    VerbResult r = cn.connect(opt);
    T_REQUIRE(r.ok(), "PoC conn HELLO failed: %s", r.detail.c_str());
    r = cn.put(keys[0], blk0.data(), kLen);
    T_REQUIRE(r.ok(), "PoC put[0] failed: %s", r.detail.c_str());
    r = cn.put(keys[1], blk1.data(), kLen);
    T_REQUIRE(r.ok(), "PoC put[1] failed: %s", r.detail.c_str());

    std::vector<uint8_t> out0(4096), out1(kLen);
    std::vector<GetDest> dests(2);
    dests[0].ptr = out0.data();
    dests[0].cap = out0.size(); // undersized: block is 128 KiB
    dests[1].ptr = out1.data();
    dests[1].cap = out1.size();
    r = cn.batch_get(keys, dests);
    T_CHECK(r.err == VerbErr::Usage, "undersized dest: err=%d want Usage(4) (%s)", int(r.err),
          r.detail.c_str());
    T_CHECK(dests[0].status == Status::NotFound && dests[0].len == 0,
          "undersized dest[0] must read as a miss (status=%s len=%u)",
          status_name(dests[0].status), dests[0].len);
    T_REQUIRE(dests[1].status == Status::OK,
            "dest[1] status=%s want OK (descriptor table clobbered by drain?)",
            status_name(dests[1].status));
    T_REQUIRE(dests[1].len == kLen, "dest[1] len=%u want %zu", dests[1].len, kLen);
    T_REQUIRE(std::memcmp(out1.data(), blk1.data(), kLen) == 0, "dest[1] bytes differ");
    T_REQUIRE(!cn.dead(), "conn marked dead after an in-sync Usage error");

    // The stream must be in perfect sync: before the fix, 128 KiB of unread
    // payload poisoned the next verb on this connection.
    uint32_t n_consec = 0;
    r = cn.batch_exists(keys, n_consec, nullptr);
    T_REQUIRE(r.ok(), "follow-up EXISTS on the same conn failed: err=%d %s", int(r.err),
            r.detail.c_str());
    T_REQUIRE(n_consec == 2, "follow-up EXISTS n_consecutive=%u want 2", n_consec);

    // Cleanup: drop the GET-granted read leases, then delete the PoC keys.
    std::vector<Status> per_key;
    r = cn.touch_lease(keys, kLeaseRelease, 0, per_key);
    T_REQUIRE(r.ok(), "PoC lease release failed");
    r = cn.del(keys, false, per_key);
    T_REQUIRE(r.ok(), "PoC delete failed");
    return true;
}

bool phase_idempotent_reput(Ctx &c) {
    VerbResult r = c.pool->with_conn(
        [&](Conn &cn) { return cn.put(c.keys[0], c.src.data(), kBlockLen); });
    T_REQUIRE(r.err == VerbErr::None, "re-PUT errored: %s", r.detail.c_str());
    T_REQUIRE(r.wire_status == Status::OKExists, "re-PUT status=%s want OK_EXISTS",
            status_name(r.wire_status));
    return true;
}

bool phase_credit_window(Ctx &c) {
    // 24 MB block against the 16 MiB floor window: the chunk stream MUST hit
    // W <= 0 and recover via server NOP/CREDIT replenishment (§8 rule 4).
    const size_t big = 24u << 20;
    std::vector<uint8_t> blob(big);
    fill_block(blob.data(), big, 0xC4ED17);
    Key k = make_key("credit", 0);
    int64_t window = 0;
    VerbResult r = c.pool->with_conn([&](Conn &cn) {
        VerbResult rr = cn.put(k, blob.data(), big);
        window = cn.credit_window();
        return rr;
    });
    T_REQUIRE(r.ok(), "24MB PUT failed: err=%d status=%s %s", int(r.err),
            status_name(r.wire_status), r.detail.c_str());
    std::printf("  credit window after 24MB PUT: %lld bytes\n",
                static_cast<long long>(window));
    // Read it back to prove integrity end-to-end across the credit waits.
    std::vector<uint8_t> back(big);
    std::vector<GetDest> dests(1);
    dests[0].ptr = back.data();
    dests[0].cap = big;
    std::vector<Key> keys{k};
    r = c.pool->with_conn([&](Conn &cn) { return cn.batch_get(keys, dests); });
    T_REQUIRE(r.ok() && dests[0].status == Status::OK && dests[0].len == big,
            "24MB read-back failed");
    T_REQUIRE(std::memcmp(blob.data(), back.data(), big) == 0, "24MB bytes differ");
    return true;
}

bool phase_abort(Ctx &c, const Daemon &d) {
    // BEGIN + half the chunks + ABORT: staging freed, key never visible,
    // daemon alive (§5 — the releaseReqH wire shape).
    Key k = make_key("abort", 1);
    VerbResult r = c.pool->with_conn([&](Conn &cn) {
        return cn.put_abort(k, c.src.data(), kBlockLen, kBlockLen / 2);
    });
    T_REQUIRE(r.err == VerbErr::None, "ABORT errored: %s", r.detail.c_str());
    T_REQUIRE(r.wire_status == Status::OK, "ABORT on live stream=%s want OK (%s)",
            status_name(r.wire_status), r.detail.c_str());

    uint32_t n_consec = 1;
    std::vector<Key> keys{k};
    VerbResult e = c.pool->with_conn(
        [&](Conn &cn) { return cn.batch_exists(keys, n_consec, nullptr); });
    T_REQUIRE(e.ok(), "EXISTS after abort failed");
    T_REQUIRE(n_consec == 0, "aborted key became visible (n_consecutive=%u)", n_consec);
    T_REQUIRE(d.alive() && healthz_ok(d.metrics_port), "daemon unhealthy after abort");
    return true;
}

bool phase_cancel_flag(Ctx &c) {
    // The engine's releaseReqH path at wire level: the cancel flag is already
    // set when the chunk loop starts, so the client MUST take the
    // BEGIN -> ABORT branch deterministically (chunk-boundary check) and the
    // key must never become visible. (put_abort above covers the
    // some-chunks-already-sent shape; together they bracket releaseReqH.)
    std::atomic<bool> cancel{true};
    Key k = make_key("cancel", 2);
    const size_t big = 16u << 20;
    std::vector<uint8_t> blob(big);
    fill_block(blob.data(), big, 0xCA9CE1);
    VerbResult r = c.pool->with_conn(
        [&](Conn &cn) { return cn.put(k, blob.data(), big, 0, &cancel); });
    T_REQUIRE(r.err == VerbErr::None, "cancel PUT errored: %s", r.detail.c_str());
    T_REQUIRE(r.detail == "canceled", "pre-set cancel flag did not abort (detail=%s status=%s)",
            r.detail.c_str(), status_name(r.wire_status));
    T_REQUIRE(r.wire_status == Status::OK, "ABORT on live canceled stream=%s want OK",
            status_name(r.wire_status));
    uint32_t n_consec = 1;
    std::vector<Key> keys{k};
    VerbResult e = c.pool->with_conn(
        [&](Conn &cn) { return cn.batch_exists(keys, n_consec, nullptr); });
    T_REQUIRE(e.ok() && n_consec == 0, "canceled key became visible");
    return true;
}

bool phase_lease_pin_delete(Ctx &c) {
    std::vector<Status> per_key;
    std::vector<Key> first(c.keys.begin(), c.keys.begin() + 4);

    // PIN_SOFT the first 4. keys[0] is ALSO still read-leased by the earlier
    // BATCH_GET (5 s auto-grant) — with both protections live the server may
    // report either ERR_LEASED or ERR_PINNED (§3.7 fixes no precedence).
    VerbResult r =
        c.pool->with_conn([&](Conn &cn) { return cn.pin(first, kPinSoft, per_key); });
    T_REQUIRE(r.ok(), "PIN_SOFT failed");
    std::vector<Key> one{c.keys[0]};
    r = c.pool->with_conn([&](Conn &cn) { return cn.del(one, false, per_key); });
    T_REQUIRE(r.ok() && per_key.size() == 1, "DELETE(leased+pinned) verb failed");
    T_REQUIRE(per_key[0] == Status::ErrLeased || per_key[0] == Status::ErrPinned,
            "DELETE on leased+pinned=%s want ERR_LEASED|ERR_PINNED", status_name(per_key[0]));

    // Drop the lease; now the soft pin must be the (exact) blocker.
    r = c.pool->with_conn([&](Conn &cn) { return cn.touch_lease(one, kLeaseRelease, 0, per_key); });
    T_REQUIRE(r.ok(), "LEASE_RELEASE(keys[0]) failed");
    r = c.pool->with_conn([&](Conn &cn) { return cn.del(one, false, per_key); });
    T_REQUIRE(r.ok() && per_key.size() == 1, "DELETE(pinned) verb failed");
    T_REQUIRE(per_key[0] == Status::ErrPinned, "DELETE on soft-pinned=%s want ERR_PINNED",
            status_name(per_key[0]));

    r = c.pool->with_conn([&](Conn &cn) { return cn.pin(first, kUnpin, per_key); });
    T_REQUIRE(r.ok(), "UNPIN failed");

    // Drop the read leases GET granted, then delete everything.
    r = c.pool->with_conn(
        [&](Conn &cn) { return cn.touch_lease(c.keys, kLeaseRelease, 0, per_key); });
    T_REQUIRE(r.ok(), "LEASE_RELEASE failed");
    r = c.pool->with_conn([&](Conn &cn) { return cn.del(c.keys, false, per_key); });
    T_REQUIRE(r.ok() && per_key.size() == kNumBlocks, "DELETE failed");
    for (size_t i = 0; i < per_key.size(); i++)
        T_CHECK(status_ok(per_key[i]), "DELETE[%zu]=%s", i, status_name(per_key[i]));

    uint32_t n_consec = 1;
    r = c.pool->with_conn([&](Conn &cn) { return cn.batch_exists(c.keys, n_consec, nullptr); });
    T_REQUIRE(r.ok() && n_consec == 0, "keys still visible after DELETE");
    return true;
}

bool phase_stats(Ctx &c) {
    std::string json;
    VerbResult r = c.pool->with_conn([&](Conn &cn) { return cn.stats(json); });
    T_REQUIRE(r.ok(), "STATS failed");
    T_REQUIRE(!json.empty() && json.find("schema") != std::string::npos,
            "STATS JSON missing schema field: %.80s", json.c_str());
    return true;
}

} // namespace

// --- NIXL engine flavor ------------------------------------------------------------

#ifdef KVB_HAVE_NIXL
namespace {

std::string key_hex(const Key &k) {
    static const char *hex = "0123456789abcdef";
    std::string s(64, '0');
    for (size_t i = 0; i < 32; i++) {
        s[2 * i] = hex[k[i] >> 4];
        s[2 * i + 1] = hex[k[i] & 0xF];
    }
    return s;
}

nixl_status_t poll_done(nixlKvblockdEngine &eng, nixlBackendReqH *h, int timeout_ms) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        nixl_status_t st = eng.checkXfer(h); // never blocks
        if (st != NIXL_IN_PROG) return st;
        if (std::chrono::steady_clock::now() > deadline) return NIXL_ERR_UNKNOWN;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

bool phase_nixl_engine(const Daemon &d) {
    nixl_b_params_t params = {
        {"endpoint", "127.0.0.1:" + std::to_string(d.data_port)},
        {"namespace", "t"},
        {"token", "sekret"},
        {"num_connections", "8"},
    };
    nixlBackendInitParams init;
    init.localAgent = "KvbTester";
    init.type = "KVBLOCKD";
    init.customParams = &params;
    init.enableProgTh = false;
    init.pthrDelay = 0;
    init.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE;
    init.enableTelemetry_ = false;

    // Registered memory: source + destination DRAM windows (declared BEFORE
    // the engine: its destructor joins the abort-released tasks below, which
    // may still be reading these buffers — reverse destruction order would be
    // a use-after-free, exactly the lifetime rule README caveat 3 states).
    std::vector<uint8_t> src(kNumBlocks * kBlockLen), dst(kNumBlocks * kBlockLen);
    for (size_t i = 0; i < kNumBlocks; i++)
        fill_block(src.data() + i * kBlockLen, kBlockLen, 0x71E00 + i);

    nixlKvblockdEngine eng(&init);
    T_REQUIRE(!eng.getInitErr(), "engine init failed (daemon at %u?)", d.data_port);
    T_REQUIRE(eng.supportsLocal() && !eng.supportsRemote() && !eng.supportsNotif(),
            "capability surface wrong");
    nixl_mem_list_t mems = eng.getSupportedMems();
    T_REQUIRE(mems.size() == 2 && mems[0] == DRAM_SEG && mems[1] == OBJ_SEG,
            "getSupportedMems != {DRAM_SEG, OBJ_SEG}");

    // 32 OBJ keys (hex metaInfo — the s3compat object-key convention).

    nixlBlobDesc src_bd(reinterpret_cast<uintptr_t>(src.data()), src.size(), 0, "");
    nixlBlobDesc dst_bd(reinterpret_cast<uintptr_t>(dst.data()), dst.size(), 0, "");
    nixlBackendMD *src_md = nullptr, *dst_md = nullptr;
    T_REQUIRE(eng.registerMem(src_bd, DRAM_SEG, src_md) == NIXL_SUCCESS, "registerMem(src)");
    T_REQUIRE(eng.registerMem(dst_bd, DRAM_SEG, dst_md) == NIXL_SUCCESS, "registerMem(dst)");

    std::vector<nixlBackendMD *> obj_mds(kNumBlocks);
    for (size_t i = 0; i < kNumBlocks; i++) {
        Key k = make_key("nixl", i);
        nixlBlobDesc od(0, kBlockLen, static_cast<uint64_t>(i), key_hex(k));
        T_REQUIRE(eng.registerMem(od, OBJ_SEG, obj_mds[i]) == NIXL_SUCCESS, "registerMem(obj %zu)",
                i);
    }

    auto make_lists = [&](std::vector<uint8_t> &buf, nixlBackendMD *dram_md,
                          nixl_meta_dlist_t &local, nixl_meta_dlist_t &remote) {
        for (size_t i = 0; i < kNumBlocks; i++) {
            local.addDesc(nixlMetaDesc(reinterpret_cast<uintptr_t>(buf.data() + i * kBlockLen),
                                       kBlockLen, 0, dram_md));
            remote.addDesc(nixlMetaDesc(0, kBlockLen, static_cast<uint64_t>(i), obj_mds[i]));
        }
    };

    // WRITE all 32, poll to completion.
    {
        nixl_meta_dlist_t local(DRAM_SEG), remote(OBJ_SEG);
        make_lists(src, src_md, local, remote);
        nixlBackendReqH *h = nullptr;
        T_REQUIRE(eng.prepXfer(NIXL_WRITE, local, remote, "KvbTester", h) == NIXL_SUCCESS,
                "prepXfer(WRITE)");
        T_REQUIRE(eng.checkXfer(h) == NIXL_ERR_NOT_POSTED, "checkXfer before post");
        T_REQUIRE(eng.postXfer(NIXL_WRITE, local, remote, "KvbTester", h) == NIXL_IN_PROG,
                "postXfer(WRITE)");
        // One in-flight transfer per handle: an immediate repost must refuse.
        T_CHECK(eng.postXfer(NIXL_WRITE, local, remote, "KvbTester", h) ==
                  NIXL_ERR_REPOST_ACTIVE,
              "repost while inflight not refused");
        T_REQUIRE(poll_done(eng, h, 120000) == NIXL_SUCCESS, "WRITE did not complete");
        T_REQUIRE(eng.releaseReqH(h) == NIXL_SUCCESS, "releaseReqH(WRITE)");
    }

    // READ back into dst, byte-compare.
    {
        nixl_meta_dlist_t local(DRAM_SEG), remote(OBJ_SEG);
        make_lists(dst, dst_md, local, remote);
        nixlBackendReqH *h = nullptr;
        T_REQUIRE(eng.prepXfer(NIXL_READ, local, remote, "KvbTester", h) == NIXL_SUCCESS,
                "prepXfer(READ)");
        T_REQUIRE(eng.postXfer(NIXL_READ, local, remote, "KvbTester", h) == NIXL_IN_PROG,
                "postXfer(READ)");
        T_REQUIRE(poll_done(eng, h, 120000) == NIXL_SUCCESS, "READ did not complete");
        T_REQUIRE(eng.releaseReqH(h) == NIXL_SUCCESS, "releaseReqH(READ)");
        T_REQUIRE(std::memcmp(src.data(), dst.data(), src.size()) == 0,
                "NIXL read-back differs from written bytes");
    }

    // Mid-flight abort: fresh keys, release immediately after post. The
    // daemon must stay healthy; the engine must reach no stuck state (its
    // dtor joins whatever the abort left running).
    std::vector<nixlBackendMD *> abort_mds(kNumBlocks);
    {
        for (size_t i = 0; i < kNumBlocks; i++) {
            Key k = make_key("nixl-abort", i);
            nixlBlobDesc od(0, kBlockLen, static_cast<uint64_t>(i), key_hex(k));
            T_REQUIRE(eng.registerMem(od, OBJ_SEG, abort_mds[i]) == NIXL_SUCCESS,
                    "registerMem(abort obj)");
        }
        nixl_meta_dlist_t local(DRAM_SEG), remote(OBJ_SEG);
        for (size_t i = 0; i < kNumBlocks; i++) {
            local.addDesc(nixlMetaDesc(reinterpret_cast<uintptr_t>(src.data() + i * kBlockLen),
                                       kBlockLen, 0, src_md));
            remote.addDesc(nixlMetaDesc(0, kBlockLen, static_cast<uint64_t>(i), abort_mds[i]));
        }
        nixlBackendReqH *h = nullptr;
        T_REQUIRE(eng.prepXfer(NIXL_WRITE, local, remote, "KvbTester", h) == NIXL_SUCCESS,
                "prepXfer(abort WRITE)");
        T_REQUIRE(eng.postXfer(NIXL_WRITE, local, remote, "KvbTester", h) == NIXL_IN_PROG,
                "postXfer(abort WRITE)");
        T_REQUIRE(eng.releaseReqH(h) == NIXL_SUCCESS, "releaseReqH mid-flight");
        std::printf("  released WRITE handle mid-flight (abort)\n");
    }

    // Prove the engine and daemon still work end-to-end after the abort.
    {
        nixl_meta_dlist_t local(DRAM_SEG), remote(OBJ_SEG);
        local.addDesc(nixlMetaDesc(reinterpret_cast<uintptr_t>(dst.data()), kBlockLen, 0,
                                   dst_md));
        remote.addDesc(nixlMetaDesc(0, kBlockLen, 0, obj_mds[0]));
        nixlBackendReqH *h = nullptr;
        T_REQUIRE(eng.prepXfer(NIXL_READ, local, remote, "KvbTester", h) == NIXL_SUCCESS,
                "prepXfer(post-abort READ)");
        T_REQUIRE(eng.postXfer(NIXL_READ, local, remote, "KvbTester", h) == NIXL_IN_PROG,
                "postXfer(post-abort READ)");
        T_REQUIRE(poll_done(eng, h, 60000) == NIXL_SUCCESS, "post-abort READ failed");
        T_REQUIRE(eng.releaseReqH(h) == NIXL_SUCCESS, "releaseReqH(post-abort READ)");
        T_REQUIRE(std::memcmp(dst.data(), src.data(), kBlockLen) == 0,
                "post-abort READ bytes differ");
    }
    T_REQUIRE(d.alive() && healthz_ok(d.metrics_port), "daemon unhealthy after NIXL phases");

    for (nixlBackendMD *md : obj_mds) eng.deregisterMem(md);
    for (nixlBackendMD *md : abort_mds) eng.deregisterMem(md);
    eng.deregisterMem(src_md);
    eng.deregisterMem(dst_md);
    return true;
}

} // namespace
#endif // KVB_HAVE_NIXL

int main() {
    std::signal(SIGPIPE, SIG_IGN);

    Daemon d;
    if (!d.start()) {
        std::printf("FAIL: could not start kvblockd daemon\n");
        return 1;
    }
    std::printf("daemon up: data=%u metrics=%u\n", d.data_port, d.metrics_port);

    {
        Conn::Options opt;
        opt.host = "127.0.0.1";
        opt.port = d.data_port;
        opt.ns = "t";
        opt.token = "sekret";
        Pool pool(opt, kPoolStreams);
        VerbResult pr = pool.prime();
        if (!pr.ok()) {
            std::printf("FAIL: HELLO/prime: %s (%s)\n", pr.detail.c_str(),
                        status_name(pr.wire_status));
            d.dump_log();
            d.stop();
            return 1;
        }
        std::printf("negotiated: batch=%u frame=%u blob=%u credit=%u server=%s\n",
                    pool.limits().max_batch_keys, pool.limits().max_frame_len,
                    pool.limits().max_blob_len, pool.limits().initial_credit,
                    pool.limits().server_name.c_str());
        T_CHECK(pool.limits().initial_credit == (16u << 20),
              "test daemon should pin initial_credit at the 16 MiB floor");

        Ctx c;
        c.pool = &pool;
        c.src.resize(kNumBlocks * kBlockLen);
        c.dst.resize(kNumBlocks * kBlockLen);
        c.keys.reserve(kNumBlocks);
        for (size_t i = 0; i < kNumBlocks; i++) {
            c.keys.push_back(make_key("blk", i));
            fill_block(c.src.data() + i * kBlockLen, kBlockLen, i);
        }

        struct {
            const char *name;
            bool ok;
        } phases[] = {
            {"parallel PUT 32x2.5MB", phase_parallel_put(c)},
            {"BATCH_EXISTS", phase_exists(c)},
            {"BATCH_GET verify", phase_get_verify(c)},
            {"undersized-dest PoC (desc clobber)", phase_undersized_dest(d)},
            {"idempotent re-PUT", phase_idempotent_reput(c)},
            {"credit window (24MB vs 16MiB floor)", phase_credit_window(c)},
            {"PUT ABORT mid-stream", phase_abort(c, d)},
            {"cancel-flag abort", phase_cancel_flag(c)},
            {"lease/pin/delete ladder", phase_lease_pin_delete(c)},
            {"STATS", phase_stats(c)},
        };
        for (const auto &p : phases)
            std::printf("%-38s %s\n", p.name, p.ok ? "PASS" : "FAIL");
    }

#ifdef KVB_HAVE_NIXL
    std::printf("--- NIXL engine flavor ---\n");
    bool nixl_ok = phase_nixl_engine(d);
    T_CHECK(nixl_ok, "NIXL engine flavor failed");
    std::printf("%-38s %s\n", "nixlBackendEngine lifecycle", nixl_ok ? "PASS" : "FAIL");
#else
    std::printf("--- NIXL engine flavor skipped (built without -Dnixl_path) ---\n");
#endif

    bool alive = d.alive() && healthz_ok(d.metrics_port);
    T_CHECK(alive, "daemon dead or unhealthy at teardown");
    if (g_failures != 0) d.dump_log();
    d.stop();

    if (g_failures == 0) {
        std::printf("test_transfer: PASS (daemon survived, zero crashes)\n");
        return 0;
    }
    std::printf("test_transfer: %d FAILURE(S)\n", g_failures);
    return 1;
}
