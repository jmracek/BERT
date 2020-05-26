#ifndef BUFFER_HPP
#define BUFFER_HPP

namespace {
constexpr int BUFFER_SIZE = 4096;
}

struct Buffer {
    char bytes[::BUFFER_SIZE];
    char& operator[] (size_t idx) {
        return bytes[idx];
    }
};

#endif
