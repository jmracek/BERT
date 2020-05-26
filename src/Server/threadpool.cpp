#include <functional>
#include <future>
#include <iostream>
#include <optional>
#include <thread>
#include <vector>
#include <unistd.h>

#include "queue.hpp"

using work_t = std::function<void()>;

ThreadPool::ThreadPool(int num_threads): 
    size(num_threads), 
    started_(std::atomic<bool>(false)),
    pad1{0},
    stopped_(std::atomic<bool>(false)),
    pad2{0},
    q(std::make_shared<Queue<work_t>>()),
    pool_(std::vector<std::thread>())
{}

ThreadPool::~ThreadPool(void) {
    if (!stopped_.load()) {
        this->stop();
    }
}

void ThreadPool::stop(bool wait_for_complete) {
    if (wait_for_complete)
        while (!q->empty()) continue;

    stopped_ = true; // Send the signal to all the workers to pack it up
    for (auto& worker : pool_) worker.join();
}
    
void ThreadPool::start(void) {
    for (int i = 0; i < size; ++i) {
        pool_.emplace_back([this] (int i) noexcept {
            while (!this->stopped_.load()) {
                std::optional<work_t> task = this->q->dequeue();
                if (!this->stopped_.load() && task) {
                    (*task)();
                }
            }
        }, i);
    }
    started_ = true;
}
    
// We assume that there is no return value from f, or that f itself is handling the results of it's own internal work
template<typename Func, typename... Args>
bool ThreadPool::submit(Func&& f, Args&&... args) noexcept {
    using return_type = typename std::invoke_result<Func&&, Args&&...>::type;
    // Idiot checks at compile and run-time
    static_assert(
        std::is_invocable<Func, Args...>::value, 
        "[ERROR]: Cannot submit non-callable object as work to thread pool."
    );
    if (not started_.load()) {
        std::cout << "[ERROR]: Cannot submit work. Threadpool not yet started!" << std::endl;
        return false;
    }

    // Make the task to execute and put it in the queue.
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        [&](void) {
            std::forward<Func>(f)(std::forward<Args>(args)...);
        }
    );
    q->enqueue(
        [task] (void) noexcept {
            (*task)();
        }
    );
    return true;
}
