//
// Created by Sanger Steel on 11/11/25.
//

#include "tensors.h"
#include <stdlib.h>
#include "tensor_ops.cuh"
#include "debug.h"

size_t get_n_elems_from_dims(int *dims, int n_dims)
{
    size_t total_elems = 1;
    for (int i = 0; i < n_dims; i++)
    {
        total_elems *= dims[i];
    }
    return (size_t)total_elems;
}


struct cuda_tensor* cuda_tensor_init(void* data, cllm_datatype dtype, int* dims, int n_dims)
{
    struct cuda_tensor *ret = (struct cuda_tensor *)safe_malloc(sizeof(struct cuda_tensor));
    ret->cpu_data = data;
    ret->dtype = dtype;
    ret->gpu_data = NULL;
    ret->n_dims = n_dims;
    ret->dims = dims;
    ret->copied = 0;
    ret->owns_data = 1;
    return ret;
}

void cuda_tensor_memcpy_cpu_to_gpu(struct cuda_tensor* cu_tens)
{
    CUDA_CHECK(cudaMalloc((void **)&cu_tens->gpu_data, get_n_elems_from_dims(cu_tens->dims, cu_tens->n_dims) * cllm_datatype_to_size(cu_tens->dtype)));
    CUDA_CHECK(
        cudaMemcpy(cu_tens->gpu_data, cu_tens->cpu_data, cu_tens->parent->total_elems * cllm_datatype_to_size(cu_tens->
            dtype), cudaMemcpyHostToDevice));
    cu_tens->copied = 1;
}

void cuda_tensor_print(struct cuda_tensor* tensor)
{
    print_tensor(tensor->cpu_data, tensor->dims, tensor->n_dims, tensor->dtype);
}

struct cuda_tensor * cuda_tensor_add(struct cuda_tensor *A, struct cuda_tensor *B)
{
    size_t n = get_n_elems_from_dims(A->dims, A->n_dims);
    struct cuda_tensor *C = cuda_tensor_init(NULL, CLLM_FLOAT32, A->dims, A->n_dims);
    C->cpu_data = NULL;
    C->gpu_data = NULL;
    CUDA_CHECK(cudaMalloc((void **)&C->gpu_data, (n * cllm_datatype_to_size(C->dtype))));
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(A->gpu_data, B->gpu_data, C->gpu_data, n);
    return C;
}

struct cuda_tensor * cuda_tensor_view(struct cuda_tensor *tensor, int start, int stop)
{
    struct cuda_tensor *slice = (struct cuda_tensor *)safe_malloc(sizeof(struct cuda_tensor));
    size_t size = cllm_datatype_to_size(tensor->dtype);
    slice->cpu_data = (char *)tensor->cpu_data + (start * size);
    slice->gpu_data = (char *)tensor->gpu_data + (start * size);
    int *dims = safe_malloc(sizeof(int) * tensor->n_dims);
    memcpy(dims, tensor->dims, sizeof(int) * tensor->n_dims);
    slice->n_dims = tensor->n_dims;
    slice->dims[slice->n_dims - 1] = stop - start;
    slice->copied = 0;
    slice->owns_data = 0;
    return slice;
}

// Performs α(A X B) + βC
struct cuda_tensor *cuda_tensor_float_multiply_add(cublasHandle_t handle, struct cuda_tensor* a, struct cuda_tensor* b, struct cuda_tensor *c, const float alpha,
    const float beta)
{
    DEBUG_GUARD({
        char buf_a[512];
        memset(buf_a, 0, sizeof(buf_a));
        char buf_b[512];
        memset(buf_b, 0, sizeof(buf_b));
        write_tensor_repl_to_buf(buf_a, a->cpu_data, a->dims, a->n_dims, a->dtype);
        write_tensor_repl_to_buf(buf_b, b->cpu_data, b->dims, b->n_dims, b->dtype);
        TRACE("performing cuda_tensor_float_multiply_add on a->%s, b->%s", buf_a, buf_b);
    });

    cuda_tensor_memcpy_cpu_to_gpu(a);
    cuda_tensor_memcpy_cpu_to_gpu(b);


    size_t M = a->dims[0];
    size_t K = a->dims[a->n_dims - 1];
    size_t N = b->dims[b->n_dims - 1];

    float *d_C;
    if (c)
    {
        cuda_tensor_memcpy_cpu_to_gpu(c);
        d_C = c->gpu_data;
    } else
    {
        // In case c is NULL, where the caller wants to do multiply only
        CUDA_CHECK(cudaMalloc((void **)&d_C, M * N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
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
    CUDA_CHECK(cudaMemcpy(C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    int *dims = (int *)safe_malloc(sizeof(int) * a->n_dims);
    dims[0] = M;
    dims[1] = N;
    int ndims = a->n_dims;
    return cuda_tensor_init(C, CLLM_FLOAT32, dims, ndims);
}


// Performs α(A X B) by calling cuda_tensor_multiply_add, where β=0
struct cuda_tensor *cuda_tensor_float_gemm(cublasHandle_t handle, struct cuda_tensor* a, struct cuda_tensor* b, const float alpha)
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
    cu_tens->n_dims = parent->n_dims;
    cu_tens->copied = 0;
    return cu_tens;
}

