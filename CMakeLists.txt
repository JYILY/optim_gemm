cmake_minimum_required(VERSION 3.28)
project(sgemm VERSION 0.1.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit)
set(CXX_STANDARD 21)

# add_definitions(-DCUBLASAPI)
message("CUDA include directories: ${CUDA_INCLUDE_DIRS}")

add_executable(sgemm sgemm.cu)

target_link_libraries(sgemm CUDA::cublas)
set_target_properties(sgemm PROPERTIES CUDA_ARCHITECTURES "61")