#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include <iostream>
#include <stdio.h>

#define L 256
#define M 512
#define N 1024

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
