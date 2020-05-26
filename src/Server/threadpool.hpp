#ifndef THREADPOOL_HPP
#define THREADPOOL_HPP

#include <atomic>
#include <functional>
#include <memory>
#include <vector>
#include <thread>

#include "queue.hpp"

using work_t = std::function<void()>;

constexpr size_t CACHE_LINE_SIZE = 64;

class ThreadPool {
public:
    const int size;
    ThreadPool(int num_threads);
    ~ThreadPool(void);

    // Delete copy and assignment operators
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator= (const ThreadPool&) = delete;

    void stop(bool = false);
    void start(void);
     
    template<typename Func, typename... Args>
    bool submit(Func&& f, Args&&... args) noexcept;

private:
    std::atomic<bool> started_;
    char pad1[CACHE_LINE_SIZE - sizeof(std::atomic<bool>)];
    std::atomic<bool> stopped_;
    char pad2[CACHE_LINE_SIZE - sizeof(std::atomic<bool>)];
    // Here I want to make sure that the queue and the stopped_ controller are on different cache lines
    std::shared_ptr<Queue<work_t>> q;
    std::vector<std::thread> pool_;
};

#include "threadpool.cpp"

#endif
