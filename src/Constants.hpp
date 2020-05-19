#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

namespace Constants {

constexpr int THREADS_PER_WARP = 32;
constexpr int GLOBAL_TILE_WIDTH = 128;
constexpr int GLOBAL_TILE_HEIGHT = 128;
constexpr int GLOBAL_TILE_K = 128;
constexpr int GLOBAL_TILE_K2 = 64;
constexpr int WARP_TILE_WIDTH = 32;
constexpr int WARP_TILE_HEIGHT = 32;
constexpr int SHMEM_TILE_SIZE = 4 * 64;

constexpr int MMA_TILE_M = 8;
constexpr int MMA_TILE_N = 8;
constexpr int MMA_TILE_K = 4;

}

#endif
