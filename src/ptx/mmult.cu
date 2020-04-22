#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector_types.h>

extern "C" __device__
void mmultShmemTiles(
    half*   ptr_a,
    half*   ptr_b,
    float4& qp0_acc_03, 
    float4& qp0_acc_47, 
    float4& qp1_acc_03, 
    float4& qp1_acc_47, 
    float4& qp2_acc_03, 
    float4& qp2_acc_47, 
    float4& qp3_acc_03, 
    float4& qp3_acc_47
);

__global__
void mmult(half* W, half* X, float* b, float* out) {
    return;
}
