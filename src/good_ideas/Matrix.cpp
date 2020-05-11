enum class MemLocation {
    HOST,
    DEVICE
};

enum class MatrixLayout {
    ROW_ORDER,
    COL_ORDER
};

// This class should
// 1) Make space for the matrix on device
// 2) Keep track of row or col ordering
// 3) Embed matrix into a matrix whose dimensions are a multiple of 8 by
//    padding with zeros
// 4) Tiling?
template<int row, int col, typename T = float, MatrixLayout L = ROW_ORDER>
class Matrix {
private:
    T* device_mem;
public:
    Matrix(void) {
         
        cudaMemset(d_C, 0, l * n * sizeof(float));
    };
    
    Matrix(T* host) {
        cudaMemcpy(d_A, h_A, l * m * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_C, 0, l * n * sizeof(float));
    };

    ~Matrix(void) {

    }

    std::shared_ptr<T> toHost(void);
    T& get(int i, int j) {

    }

    T* operator[] (int i) {
        return &device_mem[i * stride]
    }
};



/*

1. Load a global memory tile located at (i, j) to shmem
    - Issues: alignment. swizzling.
2. Load a shared memory tile into registers for mma.sync 
3. Store the result of mma.sync to shmem
4. Store from shmem to glmem

mma.sync performs four 8x8x4 matrix multiplies per warp

*/
