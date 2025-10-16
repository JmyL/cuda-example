#include "cuda_sum.hpp"

#include <cuda_util/cuda_util.hpp>

#include <cstdio>
#include <cstring>

using namespace util;

__global__ void sum_kernel(const int *input, int *output, int N) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < N) ? input[i] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

CudaSum::CudaSum(int N, const int *data) : N{N} {
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMemcpy(d_input, data, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, sizeof(int));
}

CudaSum::~CudaSum() {
    cudaFree(d_input);
    cudaFree(d_output);
}

int CudaSum::compute_sum() {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    auto numBlocks = DIV_UP(N, prop.maxThreadsPerBlock);

    if (numBlocks > prop.maxThreadsPerBlock) {
        throw std::runtime_error(
            "Request input is too large: "
            "CudaSum assumes it is less than prop.maxThreadsPerBlock^2");
    }

    {
        auto numThreads =
            (numBlocks == 1) ? ceilPowerOfTwo(N) : prop.maxThreadsPerBlock;
        sum_kernel<<<numBlocks, numThreads, numThreads * sizeof(int)>>>(
            d_input, d_output, N);
        checkError(cudaGetLastError());
    }

    if (numBlocks != 1) {
        auto numThreads = ceilPowerOfTwo(numBlocks);
        sum_kernel<<<1, numThreads, numThreads * sizeof(int)>>>(
            d_output, d_output, numBlocks);
        checkError(cudaGetLastError());
    }

    int result;
    cudaMemcpy(&result, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    checkError(cudaGetLastError());
    return result;
}