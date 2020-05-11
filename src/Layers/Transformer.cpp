
// What goes into a transformer?
// 1) Multi-head self-attention
//  a) (WK, WQ, WV) matrices * num_heads
//      K = XW_K
//      Q = XW_Q
//      V = XW_V
// then S = softmax(KQ^T, axis=1)
//      Z = SV
// 2) Feef-forward layer

class Attention: public Layer {
public:
    
}

class Transformer: public Layer {
public:
    Transformer(void) {}
private:
}

/*
Two different approaches:
1) Design the multi-head part by doing matrix mults as part of one big kernel.  This is the approach I took in my mxnet impl.
2) Execute kernels concurrently by defining a CUDA graph.
*/
