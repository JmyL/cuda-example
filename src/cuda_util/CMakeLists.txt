add_library(cuda_util cuda_util.cpp)
target_include_directories(cuda_util PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(cuda_util PUBLIC CUDA::cudart)