cmake_minimum_required(VERSION 3.22.1)
project(cllm C)

set(CMAKE_C_STANDARD 99)

set(CMAKE_CUDA_ARCHITECTURES all)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

find_package(CUDAToolkit REQUIRED)

add_executable(cllm main.c deserialize.c
        tensors.c
        tensors.h)
target_link_libraries(cllm PRIVATE CUDA::cublas CUDA::cudart)

