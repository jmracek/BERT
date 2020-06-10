#ifndef MEMORY_HPP 
#define MEMORY_HPP 

#include <atomic>
#include <future>
#include <memory>
#include <vector>

#include "queue.hpp"

namespace {
constexpr int CACHE_LINE_SIZE = 64;
}

template<typename T>
class ObjectPool {
private:
    std::vector<T*> blocks_;
    alignas(64) std::atomic<T*> current_;
    char pad[::CACHE_LINE_SIZE - sizeof(std::atomic<T*>)];
    
    T* last_;
    std::future<T*> next_;
    Queue<T*> free_;

    inline T* getPtrFromBuffer(void);
    inline bool noFreePtrsAvail(void);

public:
    ObjectPool(void);
    ~ObjectPool(void);
    T* alloc(void);
    template<typename... Args> T* alloc(Args&&...);
    void free(T* obj);
    void clean(T* obj);
};

#include "memory.cpp"

#endif

