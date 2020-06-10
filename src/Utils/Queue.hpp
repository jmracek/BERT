#ifndef QUEUE_HPP
#define QUEUE_HPP

#include <queue>
#include <mutex>
#include <optional>
#include <utility>

template<typename T>
class Queue {
private:
    std::mutex lock_;
    std::queue<T> q_;

public:
    bool empty(void) {
        lock_.lock();
        bool out = q_.empty();
        lock_.unlock();
        return out;
    }
    
    template<typename S>
    void enqueue(S&& obj) {
        lock_.lock();
        q_.push(std::forward<S>(obj));
        lock_.unlock();
    }

    std::optional<T> dequeue(void) {
        lock_.lock();
        if (q_.empty()) {
            lock_.unlock();
            return {};
        }

        T item = q_.front();
        q_.pop();
        lock_.unlock();
        return item;
    }
};

#endif
