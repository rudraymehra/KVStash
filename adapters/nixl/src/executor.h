// executor.h — non-blocking thread-pool executor for the KVBLOCKD NIXL backend.
//
// Modeled on the obj plugin's async-executor shape: a submission queue fed by
// postXfer and drained by worker threads; completion is tracked by the request
// state itself (atomic counters), not by the executor — so submit() NEVER
// blocks (SPEC 5 §3.1 hard contract: postXfer must never block) and checkXfer
// is a lock-free poll.

#ifndef KVBLOCKD_EXECUTOR_H
#define KVBLOCKD_EXECUTOR_H

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace kvb {

class Executor {
  public:
    // Spawns `threads` workers (minimum 1).
    explicit Executor(size_t threads);
    ~Executor();
    Executor(const Executor &) = delete;
    Executor &operator=(const Executor &) = delete;

    // Enqueue a task. Never blocks (unbounded queue; backpressure belongs to
    // the wire's credit ledger, not the local queue). Returns false only
    // after shutdown() — callers treat that as NIXL_ERR_BACKEND.
    bool submit(std::function<void()> task);

    // Stop accepting tasks, run what is already queued, join the workers.
    void shutdown();

    size_t threads() const { return workers_.size(); }

  private:
    void worker_loop();

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> queue_;
    std::vector<std::thread> workers_;
    bool stopping_ = false;
};

} // namespace kvb

#endif // KVBLOCKD_EXECUTOR_H
