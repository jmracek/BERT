#include <cuda_fp16.h>
#include <vector_types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef LOADER_HPP
#define LOADER_HPP

using copy_t = int4;

enum class tile_t {
    row_type,
    col_type
};

template<size_t height = 64, size_t width = 64>
class SwizzledGlmemToShmemLoader {
private:
    static constexpr size_t h = height;
    static constexpr size_t w = width;
    static constexpr size_t SHMEM_TILE_SIZE = w * h;
    static constexpr size_t N_COPY_TILES = w * h / (64 * 4);
    static constexpr size_t tiles_per_row = w / 4;
    const int lane;
    const int warp;
    const int nwarps;
    const int c;
    const int s;
    const int shmem_row;
    const int shmem_bank;
public:
    __device__
    SwizzledGlmemToShmemLoader(int lane_, int warp_, int nwarps_): 
        lane(lane_),
        warp(warp_),
        nwarps(nwarps_),
        c(lane & 7),
        s(lane >> 3),
        shmem_row( (c & 1) | (c >> 1 & 2) ),
        shmem_bank( c << 1 & 4 | s ^ shmem_row )
    {}

    __device__
    void load(half* src, half* dst, int ld_src) {
        int glmem_offset = c * sizeof(copy_t) / sizeof(half) + s * ld_src;
        int shmem_offset = shmem_bank + shmem_row * 8;
        int tile_i, tile_j;
        half* glmem_stream_ptr, *shmem_stream_ptr;
        for (int tile_idx = warp; tile_idx < N_COPY_TILES; tile_idx += nwarps) {
            // Compute the row and column of the first element of the tile in terms of the tile_idx
            tile_i = tile_idx / tiles_per_row;
            tile_j = tile_idx % tiles_per_row;
            glmem_stream_ptr = src + tile_i * 64 + tile_j * 4 * ld_src;
            shmem_stream_ptr = dst + SHMEM_TILE_SIZE * tile_idx;
            *((copy_t *)shmem_stream_ptr + shmem_offset) = *((copy_t *)(glmem_stream_ptr + glmem_offset));
        }
    }
};

#endif
