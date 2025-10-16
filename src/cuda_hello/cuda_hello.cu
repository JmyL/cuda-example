#include "cuda_hello.hpp"
#include <cstdio>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    printf("Hello from CUDA kernel (OOP style)!\n");
}

CudaHello::CudaHello() {
    // Constructor logic (if needed)
}

CudaHello::~CudaHello() {
    // Destructor logic (if needed)
}

void CudaHello::say_hello() const {
    hello_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}