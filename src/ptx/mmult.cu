#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector_types.h>

extern "C" __device__
void mmultShmemTiles(
    half*   ptr_a,
    half*   ptr_b,
    float4& mma0_acc_03, 
    float4& mma0_acc_47, 
    float4& mma1_acc_03, 
    float4& mma1_acc_47, 
    float4& mma2_acc_03, 
    float4& mma2_acc_47, 
    float4& mma3_acc_03, 
    float4& mma3_acc_47
);

constexpr int THREADS_PER_WARP = 32;
constexpr int GLOBAL_TILE_WIDTH = 128;
constexpr int GLOBAL_TILE_HEIGHT = 128;
constexpr int WARP_TILE_WIDTH = 32;
constexpr int WARP_TILE_HEIGHT = 32;
constexpr int SHMEM_TILE_SIZE = 4 * 64;

using copy_t = float4;

struct thread_copy_t {
    float elts[16];
};

/*
* This kernel computes out = activation(X * W + b) using the PTX mma.sync operation
* ARGS
*   W : Pointer to row major ordered weight matrix
*   X : Column major ordered matrix of samples, where each sample is along a row
*   b : Bias row-vector
*   activation : Any standard activation function, or possibly a cast to a different precision.
*
* Steps:
*   1. Cooperative copy of bias vector from global to shared memory
*   2. Each thread grabs the elements of the bias vector it needs from shared memory
*      and places them into accumulator fragments.
*   3. Perform the matrix multiplication.
*       a) For each tile in global memory, cooperative copy the tile from global memory to shared
*          memory using the strided access pattern.
*       b) Compute the specific address in shmem for the matrix elements of A and B required by
*          each thread to call mma.sync.
*       c) Call our device function mmultShmemTiles using our accumulator fragments and shmem ptrs
*   4. Apply any post-multiplication functor to all elements in the fragments
*   5. Store the result to the output from fragments
*/
__global__
void forward(half* W, half* X, float* b, float* out) {
    extern __shared__ half shmem[];

    // Some constants we will need to compute offsets into memory...
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int wid = tid / THREADS_PER_WARP;
    const int lane = tid % THREADS_PER_WARP;
    const int quad_pair = (lane >> 2) % 4;
    
    // Each global tile is a 128x128 block of the output matrix.  Each warp computes
    // a 32x32 submatrix of the output.  The assignment of warps to warp blocks happens 
    // using row major order.
    const int warp_bias_offset = (wid % 4) * WARP_TILE_WIDTH;

    const int global_col_offset = blockIdx.x * GLOBAL_TILE_WIDTH;
    const int global_row_offset = blockIdx.y * GLOBAL_TILE_HEIGHT;

    const float* shmem_ptr_float = (float *)&shmem[0];
    
    // STEP 1
    // This should get coalesced to only 4 glmem accesses
    if (tid < GLOBAL_TILE_WIDTH / sizeof(copy_t)) {
        *((copy_t *)shmem_ptr_float + tid) = *((copy_t *)(b + global_col_offset) + tid);
    }
    __syncthreads();

    // STEP 2
    float4 mma_acc[4][2];
    // This is a temporary buffer for retrieving elements from shmem.  They'll be appropriately
    // copied/reorganized into the fragments above.
    float thread_bias_elts[16];
    // Perform the copy from shmem to registers.  There should be zero bank conflicts here, because
    // all threads in a given quad pair access the same memory locations.
    *((thread_copy_t *)&thread_bias_etls[0]) = *((thread_copy_t *)(shmem_ptr_float + warp_bias_offset) + (quad_pair >> 1));
    // Now, rearrange the elements so they are put into their appropriate fragments.
#pragma unroll
    for (int i = 0; i < 4; i += 2) {
        mma_acc[i][0] = *((copy_t *)&thread_bias_elts[0]);
        mma_acc[i][1] = *((copy_t *)&thread_bias_elts[8]);
        mma_acc[i + 1][0] = *((copy_t *)&thread_bias_elts[4]);
        mma_acc[i + 1][1] = *((copy_t *)&thread_bias_elts[12]);
    }

    // STEP 3
    const int c = lane % 8;
    const int s = lane / 8;
    
    // This looks really weird, but is the shared memory swizzling function
    const int shmem_row = (c & 1) | ((c >> 1) & 2);
    const int bank      =  ((c << 1) & 4) | s ^ smem_row;

    const int gmem_offset  = 8 * c + s * ldx;
    const int shmem_offset = 8 * bank + shmem_row * 64; // 64 = width of shmem tile

    const half* start_a = &shmem[0]
    const half* start_b = start_a + GLOBAL_TILE_WIDTH * GLOBAL_TILE_HEIGHT;

    half* ptr_a = nullptr;
    half* ptr_b = nullptr;
    half* shmem_stream_ptr = nullptr;
    half* glmem_stream_ptr = nullptr;
    
    auto globalStripe = Matrix<half>::glmemTilePairIter(X, W, global_row_offset, global_col_offset);

    for (const auto& [glmem_tile_a, glmem_tile_b] : globalStripe) {
        // STEP 3A
        // Warps 0 - 7 copy the A matrix to shmem, warps 8 - 15 copy the B matrix
        if (wid <= 7) {
            // A matrix is 128x128 column ordered.  Each warp copies a 64x32 sized block
            const int tile_row_offset = wid >> 2;
            const int tile_col_offset = (wid % 4) * 8;

            // This is a complicated expression, but basically is the result of a memory ordering
            // whereby we use shared memory chunks of size 64x4, and lay them out in a row-ordered
            // manner
            shmem_stream_ptr = start_a + 4 * 8 * SHMEM_TILE_SIZE * tile_row_offset + SHMEM_TILE_SIZE * tile_col_offset;
            // Remember, we overloaded & to return the pointer to the first element of the tile
            glmem_stream_ptr = &glmem_tile_a + 4 * tile_col_offset * ldx + tile_row_offset * 64;

#pragma unroll
            for (int i = 0; i < 32; i += 4) {
                *((copy_t *)(shmem_stream_ptr + shmem_offset)) = *((copy_t *)(glmem_stream_ptr + glmem_offset));
                // Move to the right by one tile (4 columns)
                glmem_stream_ptr += 4 * ldx;
                shmem_stream_ptr += SHMEM_TILE_SIZE;
            }
        }
        else {
            // TODO: Complete this!
            const int copy_row_offset = ((wid % 8) >> 1) * 32;
            const int copy_col_offset = (wid  % 2) * 64;
                    
            stream_ptr = start_b + copy_row_offset * GLOBAL_TILE_WIDTH + copy_col_offset;
            glmem_stream_ptr = W + (global_col_offset + copy_col_offset) * ldx + (global_row_offset + copy_row_offset);

#pragma unroll
            for (int i = 0; i < 64; i += 4) {
                
            }
        }

        
        
        for (auto& [shmem_tile_a, shmem_tile_b] : shmemStripe) {
            // Each warp performs 4 8x8x4 matrix multiplies

        }

        
        mmultShmemTiles(
            ptr_a,
            ptr_b,
            mma_acc[0][0],
            mma_acc[0][1],
            mma_acc[1][0],
            mma_acc[1][1],
            mma_acc[2][0],
            mma_acc[2][1],
            mma_acc[3][0],
            mma_acc[3][1]
        );
    }


    // STEP 4

    return;
}


W*X = (WX)^T^T = (X^TW^T)^T 
