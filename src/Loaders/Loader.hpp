#include <cuda_fp16.h>
#include <vector_types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef LOADER_HPP
#define LOADER_HPP

using copy_t = int4;

enum class tile_t {
    a_type,
    b_type
};

template<size_t height = 64, size_t width = 64>
class GlmemToShmemLoader {
private:
    static constexpr size_t h = height;
    static constexpr size_t w = width;
    static constexpr size_t COPY_TILE_SIZE = 64 * 4;
    static constexpr size_t N_COPY_TILES = w * h / (64 * 4);
    static constexpr size_t tiles_per_row = w / 4;
    static constexpr size_t tiles_per_col = h / 4;

    const int lane;
    const int warp;
    const int nwarps;
    const int c;
    const int s;
    const int shmem_row;
    const int shmem_bank;

    __device__
    void load_a_type(half* src, half* dst, int ld_src) {
        int glmem_offset = c * sizeof(copy_t) / sizeof(half) + s * ld_src;
        int shmem_offset = shmem_bank + shmem_row * 8;
        int tile_i, tile_j;
        half* glmem_stream_ptr = nullptr; 
        half* shmem_stream_ptr = nullptr;
#pragma unroll
        for (int tile_idx = warp; tile_idx < N_COPY_TILES; tile_idx += nwarps) {
            // Compute the row and column of the first element of the tile in terms of the tile_idx
            tile_i = tile_idx / tiles_per_row;
            tile_j = tile_idx % tiles_per_row;
            glmem_stream_ptr = src + tile_i * 64 + tile_j * 4 * ld_src;
            shmem_stream_ptr = dst + COPY_TILE_SIZE * tile_idx;
            *((copy_t *)shmem_stream_ptr + shmem_offset) = *((copy_t *)(glmem_stream_ptr + glmem_offset));
        } 
    }
    
    __device__
    void load_b_type(half* src, half* dst, int ld_src) {
        int glmem_offset = c * sizeof(copy_t) / sizeof(half) + s * ld_src;
        int shmem_offset = shmem_bank + shmem_row * 8;
        int tile_i, tile_j;
        half* glmem_stream_ptr, *shmem_stream_ptr;
#pragma unroll
        for (int tile_idx = warp; tile_idx < N_COPY_TILES; tile_idx += nwarps) {
            // Compute the row and column of the first element of the tile in terms of the tile_idx
            tile_i = tile_idx % tiles_per_col;
            tile_j = tile_idx / tiles_per_col;
            glmem_stream_ptr = src + tile_i * 4 * ld_src + tile_j * 64;
            shmem_stream_ptr = dst + COPY_TILE_SIZE * tile_idx;
            *((copy_t *)shmem_stream_ptr + shmem_offset) = *((copy_t *)(glmem_stream_ptr + glmem_offset));
        }
    }

public:
    __device__
    GlmemToShmemLoader(int lane_, int warp_, int nwarps_): 
        lane(lane_),
        warp(warp_),
        nwarps(nwarps_),
        c(lane & 7),
        s(lane >> 3),
        shmem_row( (c & 1) | (c >> 1 & 2) ),
        shmem_bank( c << 1 & 4 | s ^ shmem_row )
    {}

    __device__
    void load(half* src, half* dst, int ld_src, tile_t matrix_type) {
        switch (matrix_type) {
        case tile_t::a_type: 
            load_a_type(src, dst, ld_src);
            break;
        case tile_t::b_type:
            load_b_type(src, dst, ld_src);
            break;
        }
    }
};

#endif
