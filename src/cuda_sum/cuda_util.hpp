#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

namespace util {

#define DIV_UP(a, b) (((a) + (b) - 1) / (b))

inline void checkError(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

inline unsigned int ceilPowerOfTwo(unsigned int n) {
    if (n == 0)
        return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

int print_device_info();

} // namespace util