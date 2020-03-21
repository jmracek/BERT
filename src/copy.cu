#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector_types.h>
#include <stdio.h>
#include <random>

using copy1_t = int;
using copy2_t = int2;
using copy4_t = int4;

__global__
void copy(float* buffer) {
    extern __shared__ float shmem[];
    const int lane_id = threadIdx.x % 32;
#pragma unroll
    for (int i = 0; i < 15; ++i) {
        *((copy4_t *)(shmem + i * 128) + lane_id) = *((copy4_t *)(buffer + i * 128) + lane_id);
    }
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

int main(void) {
    dim3 dimBlock(32, 1);
    dim3 dimGrid(1, 1);
    int shmem_bytes = 65536;

    float* h = new float[128 * 16];
    float* d;

    cudaError_t err = cudaMalloc((void **) &d, 128 * 16 * sizeof(float));
    initRandMatrix(h, 16, 128);
    cudaMemcpy(d, h, 128 * 16 * sizeof(float), cudaMemcpyHostToDevice);

    cudaFuncSetAttribute(copy, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes);

    copy<<<dimGrid, dimBlock, shmem_bytes>>>(d);

    delete [] h;
    cudaFree(d);
    return 0;
}


/*
    printf("  Total amount of shared memory per block:       %zu bytes\n",
           deviceProp.sharedMemPerBlock);
           */
