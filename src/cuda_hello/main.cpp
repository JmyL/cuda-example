#include <iostream>
#include "cuda_hello.hpp"

int main() {
    std::cout << "Creating CudaHello object..." << std::endl;
    CudaHello hello;
    hello.say_hello();
    return 0;
}