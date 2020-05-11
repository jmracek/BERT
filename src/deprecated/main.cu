#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <vector_types.h>

#include <random>
#include <iostream>
#include <stdio.h>

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
constexpr int GLOBAL_TILE_K = 128;
constexpr int WARP_TILE_WIDTH = 32;
constexpr int WARP_TILE_HEIGHT = 32;
constexpr int SHMEM_TILE_SIZE = 4 * 64;

constexpr int MMA_TILE_M = 8;
constexpr int MMA_TILE_N = 8;
constexpr int MMA_TILE_K = 4;

using copy_t = float4;

// Since this struct has size which is a power of two, when we cast memory locations to thread_copy_t*
// and dereference, the compiler will generate vectorized load/store instructions.  Note that the underlying
// pointer must be 16-byte aligned for this to work.
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
void forward(half* X, half* W, float* b, float* out, int ldx, int ldw, int ldo, int k) {
    extern __shared__ half shmem[];

    // Some constants we will need to compute offsets into memory...
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int wid = tid / THREADS_PER_WARP;
    const int lane = tid % THREADS_PER_WARP;
    const int quad_pair = (lane >> 2) % 4;
    
    // Each global tile is a 128x128 block of the output matrix.  Each warp computes
    // a 32x32 submatrix of the output.  The assignment of warps to warp blocks happens 
    // using row major order.
    const int warp_row  = wid >> 2;
    const int warp_col  = wid & 3;
    const int warp_bias_offset = (wid % 4) * WARP_TILE_WIDTH;
    const int global_col_offset = blockIdx.x * GLOBAL_TILE_WIDTH;
    const int global_row_offset = blockIdx.y * GLOBAL_TILE_HEIGHT;

    const float* shmem_ptr_float = (float *)&shmem[0];
    
    // STEP 1
    // This should get coalesced to only 4 glmem accesses
    if ( tid * sizeof(copy_t) / sizeof(float) < GLOBAL_TILE_WIDTH ) {
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
    *((thread_copy_t *)&thread_bias_elts[0]) = *((thread_copy_t *)(shmem_ptr_float + warp_bias_offset) + (quad_pair >> 1));
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
    const int shmem_row  = (c & 1) | (c >> 1 & 2);
    const int shmem_bank =  c << 1 & 4 | s ^ shmem_row;

    const int glmem_offset  = c * sizeof(copy_t) / sizeof(half) + s * ldx;
    const int shmem_offset = shmem_bank + shmem_row * 8; // 64 = width of shmem tile

    half* start_a = &shmem[0];
    half* start_b = start_a + GLOBAL_TILE_WIDTH * GLOBAL_TILE_HEIGHT;

    half* ptr_a = nullptr;
    half* ptr_b = nullptr;
    half* shmem_stream_ptr = nullptr;
    half* glmem_stream_ptr = nullptr;
    
    //auto globalStripe = Matrix<half>::glmemTilePairIter(X, W, global_row_offset, global_col_offset);
    half* glmem_tile_a = X + global_row_offset;
    half* glmem_tile_b = W + global_col_offset;

    for (int idx = 0; idx < k; idx += GLOBAL_TILE_K) {
        // STEP 3A: Copy tile_a and tile_b matrices from global memory to shared memory in a swizzled fashion.
        // Warps 0 - 7 copy the A matrix to shmem, warps 8 - 15 copy the B matrix
        if (wid < 8) {
            const int N_SHMEM_TILES_PER_COL = 8;
            const int N_SHMEM_TILES_PER_ROW = 4 * N_SHMEM_TILES_PER_COL;
            const int COPY_TILE_HEIGHT = 64;
            const int COPY_TILE_WIDTH  = 32;
            const int copy_row = wid >> 2;
            const int copy_col = wid & 3;

            shmem_stream_ptr = start_a + SHMEM_TILE_SIZE * (copy_row * N_SHMEM_TILES_PER_ROW + copy_col * N_SHMEM_TILES_PER_COL);
            glmem_stream_ptr = glmem_tile_a + copy_col * COPY_TILE_WIDTH * ldx + copy_row * COPY_TILE_HEIGHT;
#pragma unroll
            for (int i = 0; i < 32; i += 4) {
                *((copy_t *)shmem_stream_ptr + shmem_offset) = *((copy_t *)(glmem_stream_ptr + glmem_offset));
                // Move to the right by one tile (4 columns)
                glmem_stream_ptr += 4 * ldx;
                shmem_stream_ptr += SHMEM_TILE_SIZE;
            }
        }
        else {
            // We tile B using 32x64 blocks (copy tiles).  The blocks are each composed of eight 4x64 shmem tiles.  We use the
            // following layout for shared memory.  The 4x64 shmem tiles are swizzled into a contiguous block of memory.
            // Within a copy tile, shmem tiles appear in order from top to bottom (i.e. the shmem tile appearing 4 rows below
            // a given one is offset by 64*4 = 256 elements in shared memory).  Copy tiles are assigned to warps in row major order, filling
            // the entire 128x128 global tile.  Adjacent copy tiles are offset from one another in shmem by 8 * 256 elements.
            const int N_SHMEM_TILES_PER_COL = 8;
            const int N_SHMEM_TILES_PER_ROW = 2 * N_SHMEM_TILES_PER_COL;
            const int COPY_TILE_HEIGHT = 32;
            const int COPY_TILE_WIDTH  = 64;
            const int copy_row = (wid & 7) >> 1;
            const int copy_col = wid & 1;
            
            shmem_stream_ptr = start_b + SHMEM_TILE_SIZE * (copy_row * N_SHMEM_TILES_PER_ROW + copy_col * N_SHMEM_TILES_PER_COL);
            glmem_stream_ptr = glmem_tile_b + (copy_row * COPY_TILE_HEIGHT) * ldw + COPY_TILE_WIDTH * copy_col;
#pragma unroll
            for (int i = 0; i < 32; i += 4) {
                *((copy_t *)shmem_stream_ptr + shmem_offset) = *((copy_t *)(glmem_stream_ptr + glmem_offset));
                // Move to the right by one tile (4 columns)
                glmem_stream_ptr += 4 * ldw;
                shmem_stream_ptr += SHMEM_TILE_SIZE;
            }
        }
        __syncthreads();
        
        // The current lane needs to access the shared memory tile at this offset location.
        const int lane_bank_a = (lane >> 4) ^ (lane & 7) ^ ((warp_row & 1) << 1);
        const int lane_row_a  = lane >> 4 | (warp_row & 1) << 1;
        const int lane_bank_b = (lane >> 4) ^ (lane & 7) ^ ((warp_col & 1) << 1);
        const int lane_row_b  = lane >> 4 | (warp_col & 1) << 1;
        const int shmem_tile_a_row = warp_row >> 1;
        const int shmem_tile_b_col = warp_col >> 1;

        // STEP 3B: Iterate over shmem tiles and perform the matrix mult
        for (int tile_idx = 0; tile_idx < GLOBAL_TILE_WIDTH / MMA_TILE_K; ++tile_idx) {
            ptr_a = start_a + SHMEM_TILE_SIZE * (shmem_tile_a_row * 4 * 8 + tile_idx) + lane_row_a * 64 + lane_bank_a * 8;
            ptr_b = start_b + SHMEM_TILE_SIZE * (shmem_tile_b_col * 4 * 8 + tile_idx) + lane_row_b * 64 + lane_bank_b * 8;
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

        glmem_tile_a += ldx * GLOBAL_TILE_K;
        glmem_tile_b += ldw * GLOBAL_TILE_K;
    }

    // STEP 4
    // This is very complicated.  See the documentation in the README.  
    // Search for "Storing from registers to Shared Memory".
    const int lane_row_offset       = (lane & 3 | (lane >> 3) << 2) | ((quad_pair & 1) << 4);
    const int shmem_warp_row_offset = (wid >> 2) * 32;
    const int shmem_warp_col_offset = (wid & 3) * 32;

    // This part moves data from local registers into shmem to construct the entire 128x128 matrix.  
    //    - Each warp is responsible for transfering its own 32x32 block into shmem.  
    //    - Each thread performs eight 128-bit loads from its registers to shmem.  
    //    - Each load experiences a 4-way bank conflict, which is the best we can do.
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        // We shift downward (or upward, depending on if we started hi or low, resp.) by four rows after the first four 
        // iterations.  That's what the xor part is doing.
        *((copy_t *)(shmem_ptr_float + shmem_warp_col_offset + 
            GLOBAL_TILE_WIDTH * (shmem_warp_row_offset + lane_row_offset ^ ((i >> 2) << 2))) + (lane + i & 3) + ((quad_pair >> 1) << 2)) =
                mma_acc[ (lane + i) & 1 + (( (lane_row_offset ^ ((i >> 2) << 2)) % 8 ) >> 2) ][ ((lane + i) & 3) >> 1 ];
    }
    __syncthreads(); // Need to sync here, because now we have a block cooperative load from shmem to glmem
    
    // This last part copies the result from shmem to glmem.  Each warp is responsible for transfering eight rows
    // of the 128x128 output matrix.
    for (int i = 0; i < 8; ++i) {
        *((copy_t *)(out + ldo * (global_row_offset + i + wid * 8) + global_col_offset) + lane) = 
            *((copy_t *)(shmem_ptr_float + GLOBAL_TILE_WIDTH * (i + wid * 8)) + lane);
    }

    return;
}

void initRandMatrix(float* mem, int rows, int cols) {
    static std::random_device rd;  
    static std::mt19937 gen(rd()); 
    static std::uniform_int_distribution<int> dis(0, 1);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mem[j + cols * i] = dis(gen);
        }
    }

    return;
}

void colOrderFrom(float* row_ordered, float* col_ordered, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            col_ordered[i + rows * j] = row_ordered[j + cols * i];
        }
    }
}

void printMatrix(float* mem, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << mem[j + cols * i] << " ";
        }
        std::cout << std::endl;
    }
}


int main(void) {

    cudaFuncSetAttribute(forward, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 128 * sizeof(float));

    half* h_A = new half[128 * 128];
    half* h_B = new half[128 * 128];
    float* out = new float[128 * 128];
    float* bias = new float[128];

    half *d_A, *d_B;
    float *d_out, *d_bias;

    cudaError_t err_A = cudaMalloc((void **) &d_A, 128 * 128 * sizeof(half));
    cudaError_t err_B = cudaMalloc((void **) &d_B, 128 * 128 * sizeof(half));
    cudaError_t err_C = cudaMalloc((void **) &d_out, 128 * 128 * sizeof(float));
    cudaError_t err_D = cudaMalloc((void **) &d_bias, 128 * sizeof(float));
    
    for (int i = 0; i < 128; ++i) bias[i] = i;
    
    cudaMemcpy(d_bias, bias, 128 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0.0, 128 * 128 * sizeof(float));
    
    int nblocks = 1;
    int nthreads = 32 * 16;
    int mem = 128 * 128 * sizeof(float);

    forward<<<nblocks, nthreads, mem>>>(d_A, d_B, d_bias, d_out, 128, 128, 128, 128);

    cudaMemcpy(out, d_out, 128 * 128 * sizeof(float), cudaMemcpyDeviceToHost);
    
    printMatrix(out, 128, 128);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    delete [] h_A;
    delete [] h_B;
    delete [] out;
    delete [] bias;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_out);
    cudaFree(d_bias);
}

/*

    if (tid == 256) { 
        printf("%s", "Thread\tWarp\tElements\n");
        printf("%d\t%d\t", tid, wid);
        for (int i = 0; i < 16; ++i) {
            printf("%.2f, ", thread_bias_elts[i]);
        }
        printf("\n");
    }

*/
