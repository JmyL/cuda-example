add_library(cuda_sum cuda_sum.cu)
target_include_directories(cuda_sum PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(cuda_sum PRIVATE cuda_util)
