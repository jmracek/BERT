#include <atomic>
#include <future>
#include <memory>
#include <utility>
#include <vector>

#include "queue.hpp"

constexpr int DEFAULT_BUFFER_SIZE = 256;

template<typename T>
ObjectPool<T>::ObjectPool(void):
    // This will initialize our buffer for space to hold DEFAULT_BUFFER_SIZE objects
    blocks_{new T[DEFAULT_BUFFER_SIZE]},
    // This is the location of the current pointer to allocate
    current_(std::atomic<T*>(blocks_[0])),
    // This is the location of the last pointer in the block
    last_(std::next(blocks_[0], DEFAULT_BUFFER_SIZE - 1)),
    // This launches the task to fetch the next block
    next_(std::async(
        std::launch::async, 
        [] (void) { 
            return new T[DEFAULT_BUFFER_SIZE];
        })
    ),
    // This list contains previously freed memory to be reused
    free_(Queue<T*>())
{}

template<typename T>
ObjectPool<T>::~ObjectPool(void) {
    for (auto& ptr : blocks_) delete[] ptr;
    delete[] next_.get();
}

template<typename T>
inline T* ObjectPool<T>::getPtrFromBuffer(void) {
    // If we're currently in the middle of a block swap, come back later.
    if (current_.load() == nullptr)
        return nullptr;

    T* tmplast = last_;
    T* ptr = nullptr;
    do {
        // Check whether we've reached the end of the block.  If we have, then set current_ to
        // be equal to nullptr and do the work of changing blocks.  While a block swap is in
        // progress, this function will return nullptr for other threads.  This is desired     
        // behaviour, because then alloc will continue looping and attempt to get another pointer.
        if (std::atomic_compare_exchange_strong<T*>(&current_, &tmplast, nullptr)) {
            blocks_.push_back(next_.get());
            next_ = std::async(std::launch::async, [](void) {
                return new T[DEFAULT_BUFFER_SIZE];
            });
            last_ = std::next(blocks_.back(), DEFAULT_BUFFER_SIZE - 1);
            current_ = blocks_.back();
            return tmplast;
        }
        ptr = current_.load();
        // A block swap might have been initiated. If that's the case, come back later.
        if (ptr == nullptr)
            return ptr;
    } while (!std::atomic_compare_exchange_weak<T*>(&current_, &ptr, std::next(ptr)));
    return ptr;
}

// CAVEAT EMPTOR: There is currently nothing to prevent bugs resulting from pointer reuse.  
template<typename T>
T* ObjectPool<T>::alloc(void) {
    T* obj_ptr = nullptr;
    do {
        if (free_.empty()) {
            obj_ptr = getPtrFromBuffer();
        }
        else {
            // It's possible that another thread took the last free ptr
            // before this thread got to it.  We just need to check that
            // the dequeue successfully returned an object from the list.
            auto maybe_ptr = free_.dequeue();
            if (maybe_ptr.has_value())
                obj_ptr = *maybe_ptr;
        }
    } while (obj_ptr == nullptr);
    return obj_ptr;
}

// Allocate and construct in place.
template<typename T>
template<typename... Args>
T* ObjectPool<T>::alloc(Args&&... args) {
    T* obj_ptr = alloc();
    *obj_ptr = T(std::forward<Args>(args)...);
    return obj_ptr;
}

template<typename T>
void ObjectPool<T>::free(T* obj) {
    // Call the destructor for the object to avoid leaking memory
    obj->~T();
    free_.enqueue(obj);
    return;
}

template<typename T>
void ObjectPool<T>::clean(T* obj) {
    // Free an object without calling the destructor.  
    // This is useful when reusing protobuf objects.
    // TODO: ADD A WIPE/CLEAN ASPECT TO THIS FUNCTION
    free_.enqueue(obj);
    return;
}

