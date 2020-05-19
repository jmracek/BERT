#include <random>
#include <iostream>
#include <stdio.h>
#include <cuda_fp16.h>

extern void mmultLauncher(
    half* X, 
    half* W, 
    float* bias, 
    float* out, 
    int ldx, 
    int ldw, 
    int ldo, 
    int m, 
    int n, 
    int k
);

extern void attention_middle_launcher(
    half* Q, 
    half* KT, 
    float* out,
    float* reduce_glmem,
    int max_seq_len, 
    int attention_dim, 
    int ldo
);

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
    int m = 2048;
    int k = m;
    int n = m;

    half* h_A = new half[m * k];
    half* h_B = new half[k * n];
    float* out = new float[m * n];
    float* bias = new float[ n ];

    for (int i = 0; i < n; ++i) bias[i] = 0;
    
    half elt = __float2half(1.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                h_A[i + k  * j] = elt;
                h_B[j + n  * i] = elt;
            }
            else {
                h_A[i + k * j] = __float2half(0.0);
                h_B[j + n * i] = __float2half(0.0);
            }
        }
    }

    mmultLauncher(h_A, h_B, bias, out, k, n, n, m, n, k);
//    printMatrix(out, m, n);
}
