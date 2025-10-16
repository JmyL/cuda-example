add_library(cuda_sum cuda_sum.cu cuda_util.cpp)
target_include_directories(cuda_sum PUBLIC ${CMAKE_SOURCE_DIR}/src)