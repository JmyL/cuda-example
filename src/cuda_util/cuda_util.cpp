#include <cuda_runtime.h>
#include <iostream>

namespace util {

int print_device_info() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err)
                  << std::endl;
        return 1;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        std::cout << "\nDevice " << dev << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor
                  << std::endl;
        std::cout << "  Total global memory: " << (prop.totalGlobalMem >> 20)
                  << " MB" << std::endl;
        std::cout << "  Shared memory per block: "
                  << (prop.sharedMemPerBlock >> 10) << " KB" << std::endl;
        std::cout << "  Registers per block: " << prop.regsPerBlock
                  << std::endl;
        std::cout << "  Warp size: " << prop.warpSize << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock
                  << std::endl;
        std::cout << "  Max threads per multiprocessor: "
                  << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Multiprocessor count: " << prop.multiProcessorCount
                  << std::endl;
        std::cout << "  Max grid size: " << prop.maxGridSize[0] << " x "
                  << prop.maxGridSize[1] << " x " << prop.maxGridSize[2]
                  << std::endl;
        std::cout << "  Max threads dim: " << prop.maxThreadsDim[0] << " x "
                  << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2]
                  << std::endl;
        std::cout << "  Unified addressing: "
                  << (prop.unifiedAddressing ? "Yes" : "No") << std::endl;
    }

    return 0;
}
} // namespace util