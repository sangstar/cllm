cmake_minimum_required(VERSION 3.22.1)
project(cllm C)

set(CMAKE_C_STANDARD 99)

set(CMAKE_CUDA_ARCHITECTURES all)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DDEBUG -g ")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
endif()


find_package(CUDAToolkit REQUIRED)

add_executable(cllm src/main.c src/deserialize.c
        src/tensors.c
        src/tensors.h
        src/layers.c
        src/layers.h
        src/tensors.c
        src/debug.h
        src/model.c
        src/model.h
        src/transformer.c
        src/transformer.h
        src/tensor_ops.cu
        src/tensor_ops.cuh
        src/transformer.c
        src/transformer.h
        src/tensor_ops.cu
        src/tensor_ops.cuh)
target_link_libraries(cllm PRIVATE CUDA::cublas CUDA::cudart)

