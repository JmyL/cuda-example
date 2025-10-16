# CUDA Example

## Prerequisites

- direnv
- Install [nix](https://nixos.org/download/) and create `~/.config/nix/nix.conf` with `experimental-features = nix-command flakes` as it's contents.

All prerequisites will be installed by nix package manager.

## Todos

- [x] Check if how it works when host compiler is clang.
- [ ] Basics
    - [ ] 2D kernel
    - [ ] Atomic variables
    - [ ] Considering Warp
- [ ] Debugging using cuda-gdb
- [ ] Profiling
    - [ ] Nsight Systems
        - [ ] Nsight Compute
- Parallel Execution Management
    - [ ] Stream
    - [ ] Graph
- [ ] HPC lectures
    - [Playlist](https://www.youtube.com/watch?v=_Z0JPlu3d8Y&list=PLmJwSK7qduwVAnNfpueCgQqfchcSIEMg9)
    - [Slide](http://www.morrisriedel.de/wp-content/uploads/2024/03/2024-HPC-Lecture-0-Prologue.pdf)
- Themes
    - General Parallel Primitives
        - [ ] CUB
        - [ ] Thrust
    - Image Processing
        - [ ] OpenCV
    - Linear Algebra
        - [ ] cuBLAS
- Learning Materials
    - [ ] [Youtube](https://www.youtube.com/watch?v=zSCdTOKrnII)
    - [ ] [Container for Deep Learning Frameworks User Guide](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html)
    - [ ] [CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#meta-packages)
    - [ ] [CUDA Toolkit 13.0 Update 2 - Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
    - [ ] [Accelerated Computing - Traning](https://developer.nvidia.com/accelerated-computing-training)
    - [ ] [cuda-samples](https://github.com/NVIDIA/cuda-samples)
    - [ ] [User Forums](https://forums.developer.nvidia.com/c/accelerated-computing/5)
    - [CUDA C Programming Guide (NVIDIA)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
    - [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
    - [CUDA by Example (book)](https://developer.nvidia.com/cuda-example)

## Questions

- [ ] how can I avoid explicit waiting using cudeaDeviceSynchronize() as much as possible?
- [ ] How can I utilize coroutines for waiting GPU execution? Especially on boost.asio.
