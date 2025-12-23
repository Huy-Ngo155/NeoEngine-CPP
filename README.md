# NeoEngine-CPP

NeoEngine-CPP is a high-performance, minimalist Autograd Engine and Neural Network framework built from the ground up in C++. It is engineered for researchers and developers who require deep control over memory management and hardware acceleration.

## Technical Highlights

### 1. Advanced Memory Management
Unlike standard frameworks, NeoEngine uses a **Custom Memory Arena** system. By pre-allocating dedicated pools for Parameters, Activations, and Gradients, the engine minimizes memory fragmentation and avoids the costly overhead of frequent `malloc/free` calls during training loops.

### 2. Hardware-Level Optimizations
The engine is optimized for x86 architectures using **AVX/AVX2 SIMD** intrinsics. This allows for massive parallelism at the instruction level, significantly accelerating matrix-vector operations.



### 3. Graph-Based Autograd
NeoEngine implements a computational graph that tracks operations and dependencies. The backward pass utilizes **Topological Sorting** to ensure correct and efficient gradient flow through the network.

## Core Architecture
* **Tensor Core:** Aligned memory management for SIMD compatibility.
* **Optimization Engine:** Integrated support for SGD/Adam-style updates with gradient tolerance management.
* **Parallel Processing:** Built-in OpenMP support for multi-core performance scaling.



## Build Instructions
To compile the engine with maximum optimizations:
```bash
g++ -O3 -mavx2 -fopenmp src/main.cpp -o neo_engine
