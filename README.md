# BERT
CUDA BERT implementation

I am actively working on this project.  For now, it is just a mish-mash of my attempts to write efficient CUDA code.  When I started this, I knew barely anything about shared memory bank conflicts, coalescing global memory accesses, WMMA operations, etc.  I'm learning this as I go along.  I've been using the CUDA developer blogs/documentation, as well as the textbook, "Programming Massively Parallel Processors".  My plans for the future:

0) Implement multihead self-attention/Transformer kernels.  I am thinking about trying this two different ways.  First, is that I could write a bunch of smaller kernels and glue them together using CUDA graphs.  This is probably the easier approach, but suffers from the problem that it requires more global memory accesses.  The second approach I'm going to try is to see where I can combine some of these kernels and avoid global memory reads/writes.

1) Create a Keras/Gluon style API for creating CUDA graphs.

2) Performance profiling and optimization for inference! In advertising, you typically have less than 30ms to compute predictions for thousands of ads, so this is the benchmark I'm aiming for with BERT - 30ms to complete 1000 forward passes.

3) I'm hoping that eventually I'll be able to deploy this code in a production advertising setting.  The entire Amazon adserver is written in Java, so I'll probably write a JNI wrapper for all this native code eventually.
