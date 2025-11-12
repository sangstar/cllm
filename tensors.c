//
// Created by Sanger Steel on 11/11/25.
//

#include "tensors.h"

void cuda_tensor_memcpy_cpu_to_gpu(struct cuda_tensor* cu_tens)
{
    cudaMemcpy(cu_tens->gpu_data, cu_tens->cpu_data, cu_tens->parent->total_elems * cllm_datatype_to_size(cu_tens->dtype), cudaMemcpyHostToDevice);
    cu_tens->copied = 1;
}

float* cuda_tensor_float_gemm(cublasHandle_t handle, struct cuda_tensor* a, struct cuda_tensor* b, const float alpha,
    const float beta)
{


    cuda_tensor_memcpy_cpu_to_gpu(a);
    cuda_tensor_memcpy_cpu_to_gpu(b);

    size_t M = a->dims[0];
    size_t K = a->dims[1];
    size_t N = b->dims[1];

    float *d_C;
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    float *C = (float *)xmalloc(sizeof(float) * M*N);

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

    cudaMemcpy(C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    return C;
}

struct cuda_tensor* cuda_tensor_from_cllm_tensor_metadata(struct cllm_tensor_metadata* parent)
{
    struct cuda_tensor *cu_tens = (struct cuda_tensor *)xmalloc(sizeof(struct cuda_tensor));
    cu_tens->parent = parent;
    cu_tens->dims = parent->dims;
    cu_tens->cpu_data = parent->data;
    cu_tens->dtype = parent->dtype;
    cudaMalloc((void **)&cu_tens->gpu_data, parent->total_elems * cllm_datatype_to_size(parent->dtype));
    int copied = 0;
    return cu_tens;
}
