#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector_types.h>
#include <iostream>
#include <stdio.h>

#include "../Constants.hpp"
#include "../Loaders/Loader.hpp"

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace Constants;

extern "C" __device__
void mma_sync_32_32(
    unsigned* A,
    unsigned* B,
    float* C,
    float* D
);


// 12 warps per block

// Steps:
// K = X * WK
// Q = X * WQ
// V = X * WV
// At the end of here, we have K, Q, and V in accumulator fragments spread across the warp
// S = softmax(QK^T / sqrt(d), axis = 1) * V

/*
* This kernel performs three matrix multiplications to compute Keys, Queries, and Values of a single head of an
*  attention layer.  Additionally, this kernel transposes the keys when writing to output.
*  Requirements:
*   1) Q is half precision, column major
*   2) K^T is half precision, row major (NOTE: K in column major order is the same thing as K^T in row major order)
*   3) V is half precision, row major
*   TODO: Modify mmult kernel so that it outputs column major ordered instead of row major ordered
*/


/*
__global__
void attention_front(
    half* X, 
    half* WK, 
    half* WQ, 
    half* WV, 
    half* KT, 
    half* Q, 
    half* V, 
    int ldx, 
    int ldk, 
    int ldq, 
    int ldv, 
    int max_seq_len, 
    int embed_dim
) {
    __shared__ half shmem[64 * 64 * 4];

    const int tid = threadIdx.x;
    const int wid = tid / THREADS_PER_WARP;
    const int lane = tid % THREADS_PER_WARP;
    const int quad_pair = (lane >> 2) % 4;

    const int warp_row  = wid >> 1;
    const int warp_col  = wid & 1;

    const int kqv_label = wid / 4;
    const int kqv_index = wid % 4;

    const int warp_bias_offset = warp_col * WARP_TILE_WIDTH;
    const int global_col_offset = blockIdx.y * 64; // Since d = 64 is the default configuration, this will be zero most of the time
    const int global_row_offset = blockIdx.x * 64;
    const int nwarps = blockDim.x / THREADS_PER_WARP; // For this kernel, this is 12 warps
    
    auto shmem_loader = GlmemToShmemLoader<64, 64>(lane, wid, nwarps);
    
    float* shmem_ptr_float = (float *)&shmem[0];
    
    // STEP 2
    float accumulators[32] = {0};
    float* C = &accumulators[0];
    float* D = &accumulators[16];
    
    half* start_a = &shmem[0];
    half* start_k = start_a + 64 * 64;
    half* start_q = start_a + 64 * 64 * 2;
    half* start_v = start_a + 64 * 64 * 3;

    half* glmem_tile_a = X + global_row_offset;
    half* glmem_tile_k = WK + global_col_offset;
    half* glmem_tile_q = WQ + global_col_offset;
    half* glmem_tile_v = WV + global_col_offset;

    half* ptr_a = nullptr;
    half* ptr_b = nullptr;

    // This logic is describing how each thread accesses the elements from shmem it needs to perform
    // the mma.sync instruction
    const int lane_row_a  = lane >> 4 | (warp_row & 1) << 1;
    const int lane_bank_a = (lane >> 4) ^ (lane & 7) ^ ((warp_row & 1) << 1);
    const int lane_row_b  = (lane >> 4) | (warp_col & 1) << 1;
    const int lane_bank_b_row_0 = (lane & 3) | ((lane & 15) >> 3) << 2;
    const int lane_bank_b = lane_bank_b_row_0 ^ lane_row_b;

    half A_elts[8]; 
    half B_elts[8];
    unsigned* A = reinterpret_cast<unsigned *>(&A_elts[0]);
    unsigned* B = reinterpret_cast<unsigned *>(&B_elts[0]);

    for (int idx = 0; idx < k; idx += GLOBAL_TILE_K2) {
        shmem_loader.load(glmem_tile_a, start_a, ldx, tile_t::a_type);
        shmem_loader.load(glmem_tile_k, start_k, ldk, tile_t::b_type);
        shmem_loader.load(glmem_tile_q, start_q, ldq, tile_t::b_type);
        shmem_loader.load(glmem_tile_v, start_v, ldv, tile_t::b_type);
        __syncthreads();

        half* ptr_a = start_a + lane_row_a * 64 + lane_bank_a * 8;
        half* ptr_b = start_b + lane_row_b * 64 + lane_bank_b * 8;

#pragma unroll
        for (int tile_idx = 0; tile_idx < 64 / MMA_TILE_K; ++tile_idx) {
            *((copy_t *)A) = *((copy_t *)ptr_a);
            *((copy_t *)B) = *((copy_t *)ptr_b);
            
            mma_sync_32_32(A, B, C, D);
            
            ptr_a += SHMEM_TILE_SIZE;
            ptr_b += SHMEM_TILE_SIZE;
        }
        glmem_tile_a += ldx * GLOBAL_TILE_K2;
        glmem_tile_b += ldw * GLOBAL_TILE_K2;
        __syncthreads();
    }
    
    // TODO: MOVE THIS LOGIC INTO A CUSTOM LOADER CLASS; RegToGlmemLoader
    const int lane_row_offset       = (lane & 3 | (lane >> 3) << 2) | ((quad_pair & 1) << 4);
    const int shmem_warp_row_offset = warp_row * 32;
    const int shmem_warp_col_offset = warp_col * 32;
    
    // Copy first set of elements (C
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        *((copy_t *)(shmem_ptr_float + shmem_warp_col_offset + 
            64 * (shmem_warp_row_offset + lane_row_offset ^ ((i >> 2) << 2))) + ((lane + i) & 3) + ((quad_pair >> 1) << 2)) =
                *((copy_t *)(&accumulators[0]) + ((((lane + i) & 3) | ((quad_pair >> 1) << 2)) ^ ((i >> 2) << 2)));
    }
    __syncthreads(); // Need to sync here, because now we have a block cooperative load from shmem to glmem
    
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        out[ldo * (global_row_offset + i + warp_row * 32) + global_col_offset + 32 * warp_col + lane] = 
            shmem_ptr_float[ 64 * (i + warp_row * 32) + 32 * warp_col + lane ];
    }
     
    return;
}
*/

/* This kernel computes softmax(QK^T / sqrt(attention_dim), axis = 1)
 * First, we compute the product QK^T, then take the softmax
 * INPUTS:
 *  Q:  Column major order matrix of queries - dimensions (max_seq_len x attention_dim)
 *  KT: Row major order matrix of transposed keys (i.e. column major ordered keys)  - dimensions (attention_dim x max_seq_len)
 * OUTPUTS:
 *  out: Column major ordered softmax(QK^T / sqrt(attention_dim), axis = 1)
 *  
 * Things to try later:
 *  - Make each block compute a 64x256 strip of the output matrix, with the intention that max_seq_len is 256.
 *    this partitions the output into strips where each block handles 64 entire rows of the output.  This makes
 *    the parallel reduction part of the kernel easier since we only require intra-block synchronization.
 *      * How much shmem does this require? Requires to load 1 64x64 tile of A, and 4 64x64 tiles of B.
 *      * 5 * 64 * 64 * 2B / hf = 40,960B required, which is do-able
 *      * This kind of design may clog up the math instruction pipeline though, since we concentrate all the
 *        arithmetic on a much smaller number of blocks.
 *  - Try out __shfl__up(down)_sync() instructions
 */
__global__
void attention_middle(
    half* Q, 
    half* KT, 
    float* out,
    float* reduce_glmem,
    int max_seq_len, 
    int attention_dim, 
    int ldo
) {
    // May want to consider later making this 128x128
    __shared__ half shmem[64 * 64 * 2];

    // This kernel requires grid level synchronization
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();

    const int tid = threadIdx.x;
    const int wid = tid / THREADS_PER_WARP;
    const int lane = tid % THREADS_PER_WARP;
    const int quad_pair = (lane >> 2) % 4;
    const int warp_row  = wid >> 1;
    const int warp_col  = wid & 1;
    const int global_col_offset = blockIdx.y * 64;
    const int global_row_offset = blockIdx.x * 64;
    const int nwarps = blockDim.x / THREADS_PER_WARP;
    
    float* shmem_ptr_float = (float *)&shmem[0];
    
    float accumulators[32] = {0};
    float* C = &accumulators[0];
    float* D = &accumulators[16];
    
    half* start_a = &shmem[0];
    half* start_b = start_a + 64 * 64;
    half* ptr_a = nullptr;
    half* ptr_b = nullptr;
    half* glmem_tile_a = Q + global_row_offset;
    half* glmem_tile_b = KT + global_col_offset;

    // This is local storage for the fragments retrieved from shmem by each thread for mma.sync
    half A_elts[8]; 
    half B_elts[8];
    unsigned* A = reinterpret_cast<unsigned *>(&A_elts[0]);
    unsigned* B = reinterpret_cast<unsigned *>(&B_elts[0]);

    // This logic is describing how each thread accesses the elements from shmem it needs to perform
    // the mma.sync instruction
    const int lane_row_a  = lane >> 4 | (warp_row & 1) << 1;
    const int lane_bank_a = (lane >> 4) ^ (lane & 7) ^ ((warp_row & 1) << 1);
    const int lane_row_b  = (lane >> 4) | (warp_col & 1) << 1;
    const int lane_bank_b_row_0 = (lane & 3) | ((lane & 15) >> 3) << 2;
    const int lane_bank_b = lane_bank_b_row_0 ^ lane_row_b;

    auto shmem_loader = GlmemToShmemLoader<64, 64>(lane, wid, nwarps);

    for (int idx = 0; idx < attention_dim; idx += GLOBAL_TILE_K2) {
        shmem_loader.load(glmem_tile_a, start_a, max_seq_len, tile_t::a_type);
        shmem_loader.load(glmem_tile_b, start_b, max_seq_len, tile_t::b_type);
        __syncthreads();

        ptr_a = start_a + lane_row_a * 64 + lane_bank_a * 8;
        ptr_b = start_b + lane_row_b * 64 + lane_bank_b * 8;

#pragma unroll
        for (int tile_idx = 0; tile_idx < 64 / MMA_TILE_K; ++tile_idx) {
            *((copy_t *)A) = *((copy_t *)ptr_a);
            *((copy_t *)B) = *((copy_t *)ptr_b);

            mma_sync_32_32(A, B, C, D);

            ptr_a += SHMEM_TILE_SIZE;
            ptr_b += SHMEM_TILE_SIZE;
        }
        glmem_tile_a += max_seq_len * GLOBAL_TILE_K2;
        glmem_tile_b += max_seq_len * GLOBAL_TILE_K2;
        __syncthreads(); // This instruction may be un-necessary; Double check!
    }

    // At this point, each thread is holding 32 elements in two rows of the output matrix.  We need to compute
    // the softmax along the columns.  Since each block covers a 64x64 sized tile of the output, there will be
    // multiple blocks required to cover groups of rows, and we must synchronize between them.
    
    // First compute exponentials of all elements in the registers and sum them.

    float c_sum = 0, d_sum = 0;
    float rsqrt_d = rsqrtf(attention_dim);
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        accumulators[i] = expf(accumulators[i] * rsqrt_d);
        (i < 16 ? c_sum : d_sum) += accumulators[i];
    }

    const int lane_row_offset = (lane & 3 | (lane >> 3) << 2) | ((quad_pair & 1) << 4);
    const int warp_row_offset = 32 * warp_row;

    const int row_offset = global_row_offset + warp_row_offset + lane_row_offset;

    atomicAdd(&reduce_glmem[row_offset], ((quad_pair & 1) ? d_sum : c_sum));
    atomicAdd(&reduce_glmem[row_offset ^ 4], ((quad_pair & 1) ? c_sum : d_sum));
    grid.sync();
    
    // Read the result and finish the softmax computation
    if ( wid < 2 ) {
        shmem_ptr_float[tid] = reduce_glmem[global_row_offset + tid];
    }
    __syncthreads();

    float block_c_sum = quad_pair & 1 ? 
        shmem_ptr_float[(warp_row_offset + lane_row_offset) ^ 4] : shmem_ptr_float[warp_row_offset + lane_row_offset];
    float block_d_sum = quad_pair & 1 ? 
        shmem_ptr_float[warp_row_offset + lane_row_offset] : shmem_ptr_float[(warp_row_offset + lane_row_offset) ^ 4];

#pragma unroll
    for (int i = 0; i < 16; ++i) C[i] /= block_c_sum;
#pragma unroll
    for (int i = 0; i < 16; ++i) D[i] /= block_d_sum;
    
    // Now, we write the result to the output.
    const int shmem_warp_row_offset = warp_row * 32;
    const int shmem_warp_col_offset = warp_col * 32;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        *((copy_t *)(shmem_ptr_float + shmem_warp_col_offset + 
            64 * (shmem_warp_row_offset + lane_row_offset ^ ((i >> 2) << 2))) + ((lane + i) & 3) + ((quad_pair >> 1) << 2)) =
                *((copy_t *)(&accumulators[0]) + ((((lane + i) & 3) | ((quad_pair >> 1) << 2)) ^ ((i >> 2) << 2)));
    }
    __syncthreads(); // Need to sync here, because now we have a block cooperative load from shmem to glmem

#pragma unroll
    for (int i = 0; i < 32; ++i) {
        out[ldo * (global_row_offset + i + warp_row * 32) + global_col_offset + 32 * warp_col + lane] = 
            shmem_ptr_float[ 64 * (i + warp_row * 32) + 32 * warp_col + lane ];
    }
     
    return;
}

__host__
void attention_middle_launcher(
    half* Q, 
    half* KT, 
    float* out,
    int max_seq_len, 
    int attention_dim, 
    int ldo
) {
    half *d_A, *d_B;
    float *d_out, *d_glmem;

    gpuErrCheck(cudaMalloc((void **) &d_A, max_seq_len * attention_dim * sizeof(half)));
    gpuErrCheck(cudaMalloc((void **) &d_B, max_seq_len * attention_dim * sizeof(half)));
    gpuErrCheck(cudaMalloc((void **) &d_out, max_seq_len * max_seq_len * sizeof(float)));
    gpuErrCheck(cudaMalloc((void **) &d_glmem, max_seq_len * sizeof(float)));
    
    gpuErrCheck(cudaMemcpy(d_A, Q, max_seq_len * attention_dim * sizeof(half), cudaMemcpyHostToDevice));
    gpuErrCheck(cudaMemcpy(d_B, KT, max_seq_len * attention_dim * sizeof(half), cudaMemcpyHostToDevice));
    gpuErrCheck(cudaMemset(d_out, 0.0, max_seq_len * max_seq_len * sizeof(float)));
    gpuErrCheck(cudaMemset(d_glmem, 0.0, max_seq_len * sizeof(float)));
    
    dim3 nblocks = dim3(max_seq_len / 64, max_seq_len / 64);
    dim3 nthreads = 32 * 4;
    
    void *args[] = {
        (void *)&d_A,
        (void *)&d_B,
        (void *)&d_out,
        (void *)&d_glmem,
        (void *)&max_seq_len,
        (void *)&attention_dim,
        (void *)&max_seq_len,
    };

    gpuErrCheck(cudaLaunchCooperativeKernel(
        (void *)attention_middle,
        nblocks,
        nthreads,
        &args[0],
        0,
        NULL
    ));
    
    cudaMemcpy(out, d_out, max_seq_len * max_seq_len * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);
}
    
// What are my options for sum reduction?
// 1) Have each thread reduce its own registers then share the result globally
//  a) atomicAdd each row reduction to global memory, then sync block, then each thread gets the sum it needs.
//  b) Use shfl_up(down)_sync() intrinsics to reduce inside warp first
// 2) Have each thread write its registers to shmem, then perform a parallel reduction 64 times.
//  * This may be overkill, since the size of the reduction is only 256 elements * 64 times

