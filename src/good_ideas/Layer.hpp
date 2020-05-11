#ifndef LAYER_HPP
#define LAYER_HPP

#include <cuda.h>
#include <cuda_runtime.h>
/***** BEGIN NEURAL NETWORK *****/

// This is the base case of the recursive class definition
template<typename... Rest>
class NeuralNetwork {
public:
    infer(half* X, int output_dimension);
    inferBatched(half** X, int output_dimension, int batch_size);
    inferStridedBatched(half* X, int stride, int batch_size);
    fit(float label, half* X, int output_dimension);
    fitBatched(float* labels, half** X, int output_dimension, int batch_size);
    fitStridedBatched(float* labels, half* X, int stride, int batch_size);
protected:
    cudaGraph_t  backprop_execution_graph;
    cudaGraph_t inference_execution_graph;
    cudaStream_t stream;
    int device_id;
    std::vector<cudaGraphNode_t> backprop_operations;
    std::vector<cudaGraphNode_t> inference_operations;
    // FIXME: Loss function and its derivative?
};

template <typename L, typename... Rest> 
class NeuralNetwork<L, ...Rest>: public NeuralNetwork<Rest...> {
    L layer;
    NeuralNetwork(void) : NeuralNetwork<Rest...>(void) {
        static_assert(
            CheckCompatibleLayerDimensions<L, Rest...>::value, 
            "Layer dimensions are incompatible! Check that the specified dimensions allow for valid matrix multiplication."
        );
    }
}

/***** END NEURAL NETWORK *****/

/***** BEGIN COMPILE TIME CHECK UTILS ******/

// Template structs for checking whether adjacent Layer dimensions are compatible.
template<typename... Rest>
struct CheckCompatibleLayerDimensions {
    static constexpr bool value = true; 
};

template<typename L, typename Next, typename... Rest>
struct CheckCompatibleLayerDimensions<L, Next, Rest...> {
    static constexpr bool value = (L::col == Next::row) && CheckCompatibleLayerDimensions<Next, Rest...>::value;
};

// These next two structs are used to get the type of the k'th layer in a NN; i.e. relu, sigmoid, etc.
template<size_t, typename> struct GetLayerType {};

template<typename L, typename... Rest>
struct GetLayerType<0, NeuralNetwork<L, Rest...>> {
    typedef L type;
};

template<size_t k, typename L, typename... Rest>
struct GetLayerType<k, NeuralNetwork<L, Rest...>> {
    typedef typename GetLayerType<k - 1, NeuralNetwork<Rest...>>::type type;
};

// These next two structs are used to retrieve the k'th layer from a NN.  Similar to std::tuple.
// Second line after template parameters is the function return type
template<size_t k, Layer... Rest>
typename std::enable_if<k == 0, typename GetLayerType<0, NeuralNetwork<Rest...>>::type&>::type&
getLayer(NeuralNetwork<Rest...>& nn) {
    return nn.layer;
}

template<size_t k, Layer L, Layer... Rest>
typename std::enable_if<k != 0, typename GetLayerType<k, NeuralNetwork<L, Rest...>>::type&>::type&
getLayer(NeuralNetwork<L, Rest...>& nn) {
    NeuralNetwork<Rest...>& peel = nn;
    return getLayer<k - 1>(peel);
}

/***** END COMPILE TIME CHECK UTILS ******/


/***** BEGIN LAYER DEFINITIONS *****/

// This is a container for all the pointers to memory on both the host and device required to do backpropagation.
template<size_t row, size_t col>
struct BaseLayer {
    float* W_h;             // Host storage for weights
    float* W_d;             // Device storage for weights
    float* dW_d;            // Device storage for change in weights
    half*  hW_d;            // For tensor ops requiring weights casted to half precision
    float* b_h;             // Host storage of bias
    float* b_d;             // Device storage of bias
    float* error_d;         // Device storage of change in bias (error term)
    float* X_d;             // Input activation
    float* X_prime_d;       // Derivative at input activation
    float* Z_d;             // Z = weights*X + b
    int stride;
    int batch_size;
    static constexpr size_t input_dim = col;
    static constexpr size_t output_dim = row;
};

template<size_t row, size_t col>
struct Relu: public BaseLayer<row, col> {
    __device__ float activation(float);
}

template<size_t row, size_t col>
struct Sigmoid: public BaseLayer<row, col> {
    __device__ float activation(float);
}

__device__
float relu(const float& x) {
    return x > 0 ? x : 0;
}

__device__
float sigmoid(const float& x) {
    return 1 / (1 + expf(-x));
}

/***** END LAYER DEFINITIONS *****/


/* 

This function template is a little tricky.  Basically the operation graph looks like this:
MemCpy -> X0  ->  X1  ->  ... -> Xn
          |       |              |
          v       v              v
         d0  <-  d1  <-  ... <- dn  
So we solve the problem of constructing the cuda execution graph using a function template.
This is essentially performing a depth first traversal of the execution graph using the
template parameter pack.

*/

// The base case is that we're at the end, Xn -> dn.
template <Layer L>
cudaGraphNode_t initBackPropGraph(NeuralNetwork<L>& nn, cudaGraphNode_t& last) {

}
template <Layer L, Layer Next, Layer... Rest>
cudaGraphNode_t initBackPropGraph(NeuralNetwork<L, Next, ...Rest>& nn, cudaGraphNode_t& last) {

    cudaGraphNode_t     thisLayerForwardProp; 
    cudaKernelNodeParams   forwardPropParams = {0};
    nn.layer
    void* kernelArgs[4] = {
        (void *) W,
        X,

    };
    setKernelParams(
        kernel,


    );
    cudaGraphAddKernelNode(
        thisLayerForwardProp,
        &nn.backprop_execution_graph,
        &last,
        1,
        &kernelOpts
    );
    NeuralNetwork<Next, ...Rest>& peel_layer = nn;
    cudaGraphNode_t        nextLayerBackProp = initBackPropGraph(peel_layer, thisLayerForwardProp);

    cudaGraphNode_t backwardNode; 
}



#endif

/*

What must go into the definition of a neural network?
    Data:
        cudaGraph_t trainGraph; // This cuda graph encodes the order of operations for the backprop algorithm
        cudaStream_t stream;    // This is the stream where this NN submits its operations
        int device_id;  // Which GPU is the NN being trained on?
        Layers.  These could be either part of the type, or make a linked list of layers 
            Data:
                float* W_h;             // Host storage for weights
                float* W_d;             // Device storage for weights
                half* half_W_d;         // For tensor ops requiring weights casted to half precision
                float* b_h;             // Host storage of bias
                float* b_d;             // Device storage of bias
                float* X_d;             // Input activation
                int stride;
                int batch_size;
                float* Z_d;             // Z = weights*X + b
                int input_dim;          // Num cols of W
                int output_dim;         // Num rows of W
                Layer* next;            // next == nullptr Maybe?
            Methods: 
                
    Methods: 
        infer(half* X, int output_dimension);
        inferBatched(half** X, int output_dimension, int batch_size);
        inferStridedBatched(half* X, int stride, int batch_size);
        fit(float label, half* X, int output_dimension);
        fitBatched(float* labels, half** X, int output_dimension, int batch_size);
        fitStridedBatched(float* labels, half* X, int stride, int batch_size);

*/


/* What does the computation graph look like, if we include all memory transfers and conversions to half?

1) Copy X0 from host to device
2) Forward propagate, computing activation and derivative thereof
3) Backward propagate errors and do weight/bias updates all in one step

*/
