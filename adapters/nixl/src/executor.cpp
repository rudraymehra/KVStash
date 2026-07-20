// executor.cpp — see executor.h.

#include "executor.h"

namespace kvb {

Executor::Executor(size_t threads) {
    if (threads == 0) threads = 1;
    workers_.reserve(threads);
    for (size_t i = 0; i < threads; i++) workers_.emplace_back([this] { worker_loop(); });
}

Executor::~Executor() { shutdown(); }

bool Executor::submit(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lg(mu_);
        if (stopping_) return false;
        queue_.push_back(std::move(task));
    }
    cv_.notify_one();
    return true;
}

void Executor::shutdown() {
    {
        std::lock_guard<std::mutex> lg(mu_);
        if (stopping_) return;
        stopping_ = true;
    }
    cv_.notify_all();
    for (std::thread &t : workers_) {
        if (t.joinable()) t.join();
    }
}

void Executor::worker_loop() {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stopping_ || !queue_.empty(); });
            if (queue_.empty()) return; // stopping and drained
            task = std::move(queue_.front());
            queue_.pop_front();
        }
        task(); // tasks are noexcept by contract (they record errors in the
                // request state; nothing may unwind across the pool)
    }
}

} // namespace kvb
