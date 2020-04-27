#include <random>
#include <iostream>
#include <stdio.h>
#include <cuda_fp16.h>

extern void mmultLauncher(half* X, half* W, float* bias, float* out, int ldx, int ldw, int ldo, int k);

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
    half* h_A = new half[128 * 128];
    half* h_B = new half[128 * 128];
    float* out = new float[128 * 128];
    float* bias = new float[128];

    for (int i = 0; i < 128; ++i) bias[i] = i;

    mmultLauncher(h_A, h_B, bias, out, 128, 128, 128, 128);

    printMatrix(out, 128, 128);
}
