#include "cuda_image_morph.hpp"

#include <cuda_util/cuda_util.hpp>

#include <cstdio>
#include <cstring>

using namespace util;

__global__ void erode_kernel(const unsigned char *input, unsigned char *output,
                             int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    unsigned char min_val = 255;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                unsigned char val = input[ny * width + nx];
                if (val < min_val)
                    min_val = val;
            }
        }
    }
    output[y * width + x] = min_val;
}

__global__ void dilate_kernel(const unsigned char *input, unsigned char *output,
                              int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    unsigned char max_val = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                unsigned char val = input[ny * width + nx];
                if (val > max_val)
                    max_val = val;
            }
        }
    }
    output[y * width + x] = max_val;
}

__global__ void efficient_erode_kernel(const unsigned char *input,
                                       unsigned char *output, int width,
                                       int height) {
    extern __shared__ unsigned char sdata[];

    int sx = threadIdx.x;
    int sy = threadIdx.y;
    int x = blockIdx.x * (blockDim.x - 2) + threadIdx.x - 1;
    int y = blockIdx.y * (blockDim.y - 2) + threadIdx.y - 1;

    if (x < 0 || y < 0 || x >= width || y >= height) {
        sdata[sy * blockDim.x + sx] = 255;
    } else {
        sdata[sy * blockDim.x + sx] = input[y * width + x];
    }
    __syncthreads();

    if (sx == 0 || sy == 0 || sx == blockDim.x - 1 || sy == blockDim.y - 1) {
        return;
    }

    if (x >= width || y >= height)
        return;

    unsigned char min_val = 255;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = sx + dx;
            int ny = sy + dy;
            unsigned char val = sdata[ny * blockDim.x + nx];
            if (val < min_val)
                min_val = val;
        }
    }
    output[y * width + x] = min_val;
}

__global__ void efficient_dilate_kernel(const unsigned char *input,
                                        unsigned char *output, int width,
                                        int height) {
    extern __shared__ unsigned char sdata[];

    int sx = threadIdx.x;
    int sy = threadIdx.y;
    int x = blockIdx.x * (blockDim.x - 2) + threadIdx.x - 1;
    int y = blockIdx.y * (blockDim.y - 2) + threadIdx.y - 1;

    if (x < 0 || y < 0 || x >= width || y >= height) {
        sdata[sy * blockDim.x + sx] = 0;
    } else {
        sdata[sy * blockDim.x + sx] = input[y * width + x];
    }
    __syncthreads();

    if (sx == 0 || sy == 0 || sx == blockDim.x - 1 || sy == blockDim.y - 1) {
        return;
    }

    if (x >= width || y >= height)
        return;

    unsigned char max_val = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = sx + dx;
            int ny = sy + dy;
            unsigned char val = sdata[ny * blockDim.x + nx];
            if (val > max_val)
                max_val = val;
        }
    }
    output[y * width + x] = max_val;
}

CudaImageMorph::CudaImageMorph(int width, int height, const unsigned char *data)
    : width(width), height(height) {
    size_t bytes = width * height * sizeof(unsigned char);
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMemcpy(d_input, data, bytes, cudaMemcpyHostToDevice);
}

CudaImageMorph::~CudaImageMorph() {
    cudaFree(d_input);
    cudaFree(d_output);
}

void CudaImageMorph::erode(unsigned char *output_host) {
    dim3 blockDim(16, 16);
    dim3 gridDim(DIV_UP(width, blockDim.x), DIV_UP(height, blockDim.y));

    erode_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    checkError(cudaGetLastError());

    if (output_host == nullptr) {
        cudaDeviceSynchronize();
        return;
    }
    cudaMemcpy(output_host, d_output, width * height * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);
    checkError(cudaGetLastError());
}

void CudaImageMorph::dilate(unsigned char *output_host) {
    dim3 blockDim(16, 16);
    dim3 gridDim(DIV_UP(width, blockDim.x), DIV_UP(height, blockDim.y));

    dilate_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    checkError(cudaGetLastError());

    if (output_host == nullptr) {
        cudaDeviceSynchronize();
        return;
    }
    cudaMemcpy(output_host, d_output, width * height * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);
    checkError(cudaGetLastError());
}

void CudaImageMorph::efficient_erode(unsigned char *output_host) {
    dim3 blockDim(16, 16);
    dim3 gridDim(DIV_UP(width, blockDim.x - 2), DIV_UP(height, blockDim.y - 2));

    efficient_erode_kernel<<<gridDim, blockDim, blockDim.x * blockDim.y>>>(
        d_input, d_output, width, height);
    checkError(cudaGetLastError());

    if (output_host == nullptr) {
        cudaDeviceSynchronize();
        return;
    }
    cudaMemcpy(output_host, d_output, width * height * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);
    checkError(cudaGetLastError());
}

void CudaImageMorph::efficient_dilate(unsigned char *output_host) {
    dim3 blockDim(16, 16);
    dim3 gridDim(DIV_UP(width, blockDim.x - 2), DIV_UP(height, blockDim.y - 2));

    efficient_dilate_kernel<<<gridDim, blockDim, blockDim.x * blockDim.y>>>(
        d_input, d_output, width, height);
    checkError(cudaGetLastError());

    if (output_host == nullptr) {
        cudaDeviceSynchronize();
        return;
    }
    cudaMemcpy(output_host, d_output, width * height * sizeof(unsigned char),
               cudaMemcpyDeviceToHost);
    checkError(cudaGetLastError());
}
