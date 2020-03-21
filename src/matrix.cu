#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <random>
#include <iostream>
#include <stdio.h>
#include <vector_types.h>

#define L 256
#define M 512
#define N 1024

using copy4_t = int4;

constexpr int l = L;
constexpr int m = M;
constexpr int n = N;
//constexpr int SHMEM_SIZE = 2 >> 14; // 32kB
constexpr int tbp = 512;
constexpr int nblocks = l * n / tbp;

// This kernel is actually 10x faster than NaiveCOrdered because of memory coalescing.
__global__
void matrixMultiplyNaive(float* A, float* B, float* C, int l, int m, int n) {
    const int row = (blockIdx.x * tbp + threadIdx.x) / n;
    const int col = (blockIdx.x * tbp + threadIdx.x) % n; 
    float tmp = 0;
    for (int k = 0; k < m; ++k)
        tmp += A[k + row * m] * B[col + k * n];

    C[col + n * row] = tmp;
}

__global__
void matrixMultiplyNaiveBColOrdered(float* A, float* B, float* C, int l, int m, int n) {
    const int row = (blockIdx.x * tbp + threadIdx.x) / n;
    const int col = (blockIdx.x * tbp + threadIdx.x) % n; 
    float tmp = 0;
    for (int k = 0; k < m; ++k)
        tmp += A[k + row * m] * B[k + col * m];

    C[col + n * row] = tmp;
}

__global__
void matrixMultiplyNaiveAColOrdered(float* A, float* B, float* C, int l, int m, int n) {
    const int row = (blockIdx.x * tbp + threadIdx.x) / n;
    const int col = (blockIdx.x * tbp + threadIdx.x) % n; 
    float tmp = 0;
    for (int k = 0; k < m; ++k)
        tmp += A[row + k * m] * B[col + k * n];

    C[col + n * row] = tmp;
}

/*
// This is a convenience function for allocating a pair of a host and device pointer
template<typename T>
std::pair<T*, T*> cudaAlloc(size_t nobj) {
    T* host_ptr = new T[nobj * sizeof(T)];
    T* device_ptr;
    
    cudaError_t err;
    if ((err = cudaMalloc((void **) &device_ptr, nobj * sizeof(T))) != cudaSuccess) {
        std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        exit(1):
    }

    return {host_ptr, device_ptr};
}
*/

//constexpr int TILE_WIDTH = 32;
#define TILE_WIDTH 32
// We are going to create a (k x k) tiling - k must divide n and l
__global__
void matrixMultiplyShmem(float* A, float* B, float* C, int l, int m, int n) {
    // Declare some shared memory.  We'll load from global mem into these buffers.
    __shared__ float bufA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float bufB[TILE_WIDTH][TILE_WIDTH];

    const int row = blockDim.y * blockIdx.y + threadIdx.y;
    const int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    float result = 0; // This variable will store the result of matrix mult
    for (int tile_idx = 0; tile_idx < M / TILE_WIDTH; ++tile_idx) { 
        // Load the tile from global memory into shared memory and wait until all 
        // threads have performed the load
        bufA[threadIdx.y][threadIdx.x] = A[row * m + tile_idx * TILE_WIDTH + threadIdx.x];
        bufB[threadIdx.y][threadIdx.x] = B[n * (tile_idx * TILE_WIDTH + threadIdx.y) + col];
        __syncthreads();
        
        // Add the contribution from this tile
#pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) 
            result += bufA[threadIdx.y][k] * bufB[k][threadIdx.x];
        
        // I don't think this barrier is strictly necessary, but if it is not here,
        // we may not be able to coalesce global memory accesses.
        // COMPLETE: See what happens to kernel benchmarks if I remove it.
        // RESULT: Actually, the code breaks and gives incorrect results if I don't include this.
        // Kirk+Hwu, p.93 explains why.  If it's not here, then a thread may modify shared memory
        // while other threads are still using it.
        __syncthreads();
    }
    
    C[col + n * row] = result;
}

/*
* Steps:
* 1. Copy C into shared memory
* 2. Put C from shared memory into an accumulator fragment

*/

constexpr int THREADS_PER_WARP          = 32;
constexpr int WMMA_BIT_ALIGNMENT        = 128;
constexpr int SHMEM_HALF_BANK_OFFSET    = WMMA_BIT_ALIGNMENT / (BITS_PER_BYTE * sizeof(half));
constexpr int SHMEM_FLOAT_BANK_OFFSET   = WMMA_BIT_ALIGNMENT / (BITS_PER_BYTE * sizeof(float));

// How should we create tiles of the output matrix and assign threads/warps to those
// tiles?  What part of the output is each thread/warp responsible for?
//
// Definitions:
// 1) A WMMA tile is the portion of a matrix handled by the wmma::fragment type and its
//    associated operations
// 2) A warp tile is a (WARP_ROWS x WARP_COLS) block of WMMA tiles of the output matrix.
//    Each warp is responsible for computing the matrix elements in its warp tile.
// 3) A block tile is comprised of (BLOCK_WARP_ROWS x BLOCK_ROW_COLS) warp tiles.  This
//    is the portion of the output matrix that each thread block is computing.
// 4) A constant labeled with {A}_TILE_WIDTH or {A}_TILE_HEIGHT is referring to a number
//    of matrix elements along the rows or columns, respectively, of a tile of type A
// 5) A constant labeled with {A}_{B}_ROWS or {A}_{B}_COLS is referring to the number of 
//    tiles of type B along the rows or columns, respectively, of a tile of type A.
constexpr int WARP_WMMA_COLS        = 2;
constexpr int WARP_WMMA_ROWS        = 4;
constexpr int BLOCK_WARP_ROWS       = 2;
constexpr int BLOCK_WARP_COLS       = 4;
constexpr int BLOCK_WMMA_ROWS       = BLOCK_WARP_ROWS * WARP_ROWS; // 8
constexpr int BLOCK_WMMA_COLS       = BLOCK_WARP_COLS * WARP_COLS; // 8
// This is the number of WMMA fragments along the M-dimension for the tiles of A and B
constexpr int BLOCK_WMMA_M_DIM      = 4

constexpr int WMMA_TILE_WIDTH       = 16;
constexpr int WMMA_TILE_HEIGHT      = 16;
constexpr int IRRELEVANT_DIM        = 16;

constexpr int WARP_TILE_WIDTH       = WARP_WMMA_COLS * WMMA_TILE_WIDTH;
constexpr int WARP_TILE_HEIGHT      = WARP_WMMA_ROWS * WMMA_TILE_HEIGHT;
constexpr int BLOCK_TILE_WIDTH      = WMMA_TILE_WIDTH * WARP_WMMA_COLS * BLOCK_WARP_COLS; // 16 * 2 * 4 = 128
constexpr int BLOCK_TILE_HEIGHT     = WMMA_TILE_HEIGHT * WARP_WMMA_ROWS * BLOCK_WARP_ROWS; // 16 * 4 * 2 = 128

constexpr int WARPS_PER_BLOCKS      = BLOCK_WMMA_ROWS * BLOCK_WMMA_COLS / (WARP_WMMA_COLS * WARP_WMMA_ROWS); 
constexpr int THREADS_PER_BLOCKS    = THREADS_PER_WARP * WARPS_PER_BLOCK

// Since we only have 65kB of shared memory available per block, 
// we choose l and m such that:
// l * m * 16 * 16 * 2 * 2 <= 65kB => l * m <= 64
// We choose l = 8 and m = 4.  

template<typename T>
__device__
inline T* offset(T* ptr, int rows, int cols, int lda) {
    return ptr + lda * rows + cols;
}

// Args:
// A: Row major ordered matrix of size l x m
// B: Row major ordered matrix of size m x n
// C: Row major ordered matrix of size l x n
__global__
void matrixMultiplyWmma(float* A, float* B, float* C, int l, int m, int n) {
    extern __shared__ half shmem[][ + SHMEM_BANK_OFFSET];
    const int tid       = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_warp_x = (tid / THREADS_PER_WARP) % BLOCK_WARP_COLS;
    const int block_warp_y = (tid / THREADS_PER_WARP) / BLOCK_WARP_COLS;
    const int lane_id   = tid % THREADS_PER_WARP;
    const int warp_id   = tid / THREADS_PER_WARP;
    const int tile_row_offset = blockIdx.x * BLOCK_TILE_HEIGHT;
    const int tile_col_offset = blockIdx.y * BLOCK_TILE_WIDTH;

    // Load C from glmem into shmem
    float *shmem_float_ptr = offset(
        std::static_cast<float *>(&shmem[0][0]),    // Memory start
        warp_id * WMMA_TILE_HEIGHT,                 // Row offset
        0,                                          // Col offset 
        BLOCK_TILE_WIDTH                            // Stride
    );

    // Each warp copies a 16x128 chunk of the C matrix, one row at a time.
    // Each thread copies 4 entries (32 bytes) at a time from glmem
#pragma unroll
    for (int i = 0; i < WMMA_TILE_HEIGHT; ++i) {
        float* shmem_ptr = offset(shmem_float_ptr, i, 0, BLOCK_TILE_WIDTH);
        float* glmem_ptr = offset(C, i + tile_row_offset + warp_id * WMMA_TILE_HEIGHT, tile_col_offset, n);
        *((copy4_t *)shmem_ptr + lane_id) = *((copy4_t *)glmem_ptr + lane_id);
        // May want to try out a syncthreads in here, rather than after the loop
    }
    __syncthreads();
    
    // This is where we store all the tiles.
    wmma::fragment<wmma::accumulator, WMMA_TILE_HEIGHT, WMMA_TILE_WIDTH, IRRELEVANT_DIM, float> c_frag[WARP_WMMA_COLS][WARP_WMMA_ROWS];

    float* tile_ptr;
#pragma unroll
    for (int i = 0; i < WARP_TILE_ROWS; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_COLS; ++j) {
            tile_ptr = offset(shmem_float_ptr, i + block_warp_y * WARP_TILE_HEIGHT, j * WMMA_TILE_WIDTH + block_warp_x * WARP_TILE_WIDTH, BLOCK_TILE_WIDTH);
            wmma::load_matrix_sync(c[i][j], tile_ptr, BLOCK_TILE_WIDTH, wmma::mem_row_major);
        }
    }
    // Load matrix fragments from shared memory into C
    wmma::fill_fragment(acc_frag, 0.0f);
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag[WARP_WMMA_COLS];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag[WARP_WMMA_ROWS];


    wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
    wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

    /* 
    // Loop for sliding through tiles on A and B.
    for (int tile_idx = 0; tile_idx < m / TILE_WIDTH; ++tile_idx) {
        // Copy A and B from global memory into shared memory.
        
        __syncthreads();

        // (i,j) indexes the tiles of the output matrix.
#pragma unroll
        for (int i = 0; i < WMMA_TILE_ROWS; ++i) {
#pragma unroll
            for (int j = 0; j < WMMA_TILE_COLS; ++j) {

#pragma unroll
                for (int k = 0; k < WMMA_TILE_K; ++k)
                    wmma::mma_sync(c_frag[i][j], a_frag[k], b_frag[k], c_frag[i][j]);
                
            }
        }
    }

    // Memory access patterns for wmma::store_matrix_sync are basically random
    // and will not allow us to coalesce.  In that case, we first load the result
    // into shared memory, then into global memory.

    // Copy tiles into shmem
#pragma unroll
    for (int i = 0; i < WMMA_TILE_ROWS; ++i) {
#pragma unroll
        for (int j = 0; j < WMMA_TILE_COLS; ++j) {
            // TODO: Compute ptr...
            wmma::store_matrix_sync(shmem_ptr, c[i][j], stride, wmma::mem_row_major);
        }
    }
    // Copy from shmem into glmem
*/
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

void initRandMatrixColOrdered(float* mem, int rows, int cols) {
    static std::random_device rd;  
    static std::mt19937 gen(rd()); 
    static std::uniform_int_distribution<int> dis(0, 1);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mem[i + rows * j] = dis(gen);
        }
    }

    return;
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
    float* h_A = new float[l * m];
    float* h_Acol = new float[l * m];
    float* h_B = new float[m * n];
    float* h_Bcol = new float[m * n];
    float* h_C = new float[l * n];
    float* h_C3 = new float[l * n];
    float* h_Ccol = new float[l * n];
    float* h_Ccol2 = new float[l * n];

    float *d_A, *d_Acol, *d_B, *d_Bcol, *d_C, *d_Ccol, *d_Ccol2, *d_C3;

    cudaError_t err_A     = cudaMalloc((void **) &d_A, l * m * sizeof(float));
    cudaError_t err_Acol  = cudaMalloc((void **) &d_Acol, l * m * sizeof(float));
    cudaError_t err_B     = cudaMalloc((void **) &d_B, m * n * sizeof(float));
    cudaError_t err_Bcol  = cudaMalloc((void **) &d_Bcol, m * n * sizeof(float));
    cudaError_t err_C     = cudaMalloc((void **) &d_C, l * n * sizeof(float));
    cudaError_t err_Ccol  = cudaMalloc((void **) &d_Ccol, l * n * sizeof(float));
    cudaError_t err_C3    = cudaMalloc((void **) &d_C3, l * n * sizeof(float));
    
    initRandMatrix(h_A, l, m);
    initRandMatrix(h_B, m, n);
    colOrderFrom(h_B, h_Bcol, m, n);
    colOrderFrom(h_A, h_Acol, l, m);
    //printMatrix(h_A, l, m);
    //printMatrix(h_B, m, n);

    cudaMemcpy(d_A, h_A, l * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Acol, h_Acol, l * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Bcol, h_Bcol, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, l * n * sizeof(float));
    cudaMemset(d_Ccol, 0, l * n * sizeof(float));
    cudaMemset(d_C3, 0, l * n * sizeof(float));

    matrixMultiplyNaive<<<nblocks, tbp>>>(d_A, d_B, d_C, l, m, n);
    matrixMultiplyNaiveBColOrdered<<<nblocks, tbp>>>(d_A, d_Bcol, d_Ccol, l, m, n);
    //matrixMultiplyNaiveAColOrdered<<<nblocks, tbp>>>(d_Acol, d_B, d_Ccol2, l, m, n);
    
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(n / TILE_WIDTH, l / TILE_WIDTH);
    matrixMultiplyShmem<<<dimGrid, dimBlock>>>(d_A, d_B, d_C3, l, m, n);
    
    dim3 dimBlock2(THREADS_PER_WARP * BLOCK_WARP_COLS, BLOCK_WARP_COLS);
    dim3 dimGrid2(l / BLOCK_TILE_WIDTH, n / BLOCK_TILE_HEIGHT);
    matrixMultiplyShmem<<<dimGrid2, dimBlock2, BLOCK_TILE_WIDTH * BLOCK_TILE_HEIGHT * sizeof(float) >>>(d_A, d_B, d_C3, l, m, n);

    cudaMemcpy(h_C, d_C, l * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C3, d_C3, l * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Ccol, d_Ccol, l * n * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(h_Ccol2, d_Ccol2, l * n * sizeof(float), cudaMemcpyDeviceToHost);

    bool agrees = true;
    for (int i = 0; i < l; ++i) {
        for (int j = 0; j < n; ++j) {
            if (h_Ccol[j + n * i] != h_C[j + n * i])
                agrees = false;
        }
    }
    std::cout << agrees << std::endl;
    
    for (int i = 0; i < l; ++i) {
        for (int j = 0; j < n; ++j) {
            if (h_C3[j + n * i] != h_C[j + n * i])
                agrees = false;
        }
    }
    std::cout << agrees << std::endl;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    delete [] h_A;
    delete [] h_B;
    delete [] h_Bcol;
    delete [] h_C;
    delete [] h_C3;
    delete [] h_Ccol;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_Bcol);
    cudaFree(d_C);
    cudaFree(d_C3);
    cudaFree(d_Ccol);
}
