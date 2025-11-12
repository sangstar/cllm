//
// Created by Sanger Steel on 11/11/25.
//

#include "../tensors.h"
#include <stdlib.h>

void cuda_tensor_memcpy_cpu_to_gpu(struct cuda_tensor* cu_tens)
{
    cudaMemcpy(cu_tens->gpu_data, cu_tens->cpu_data, cu_tens->parent->total_elems * cllm_datatype_to_size(cu_tens->dtype), cudaMemcpyHostToDevice);
    cu_tens->copied = 1;
}

// Performs α(A X B) + βC
float* cuda_tensor_float_multiply_add(cublasHandle_t handle, struct cuda_tensor* a, struct cuda_tensor* b, struct cuda_tensor *c, const float alpha,
    const float beta)
{


    cuda_tensor_memcpy_cpu_to_gpu(a);
    cuda_tensor_memcpy_cpu_to_gpu(b);


    size_t M = a->dims[0];
    size_t K = a->dims[1];
    size_t N = b->dims[1];


    float *d_C;
    if (c)
    {
        cuda_tensor_memcpy_cpu_to_gpu(c);
        d_C = c->gpu_data;
    } else
    {
        // In case c is NULL, where the caller wants to do multiply only
        cudaMalloc((void **)&d_C, M * N * sizeof(float));
        cudaMemset(d_C, 0, M * N * sizeof(float));
    }

    float *C = (float *)safe_malloc(sizeof(float) * M*N);

    if (K != b->dims[0])
    {
        perror("invalid dimensions for matmul");
        exit(1);
    }

    // Note: cuBLAS uses column-major order by default
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,         // (note swapped M/N)
                &alpha,
                b->gpu_data, N,
                a->gpu_data, K,
                &beta,
                d_C, N);

    // Writes result to d_C while using it in the calculation. Copy to host buffer C
    cudaMemcpy(C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    return C;
}


// Performs α(A X B) by calling cuda_tensor_multiply_add, where β=0
float* cuda_tensor_float_gemm(cublasHandle_t handle, struct cuda_tensor* a, struct cuda_tensor* b, const float alpha)
{
    return cuda_tensor_float_multiply_add(handle, a, b, NULL, alpha, 0.0f);
}

struct cuda_tensor* cuda_tensor_from_cllm_tensor_metadata(struct cllm_tensor_metadata* parent)
{
    struct cuda_tensor *cu_tens = (struct cuda_tensor *)safe_malloc(sizeof(struct cuda_tensor));
    cu_tens->parent = parent;
    cu_tens->dims = parent->dims;
    cu_tens->cpu_data = parent->data;
    cu_tens->dtype = parent->dtype;
    cudaMalloc((void **)&cu_tens->gpu_data, parent->total_elems * cllm_datatype_to_size(parent->dtype));
    int copied = 0;
    return cu_tens;
}
