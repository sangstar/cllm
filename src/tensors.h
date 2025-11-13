//
// Created by Sanger Steel on 11/11/25.
//

#ifndef CLLM_TENSORS_H
#define CLLM_TENSORS_H
#include <cublas_v2.h>
#include <string.h>

#include "deserialize.h"

size_t get_n_elems_from_dims(int *dims, int n_dims);

struct cuda_tensor
{
    const void *cpu_data;
    void *gpu_data;
    int *dims;
    int n_dims;
    cllm_datatype dtype;
    struct cllm_tensor_metadata *parent;
    int copied;
    int owns_data;
};


struct cuda_tensor *cuda_tensor_init(void *data, cllm_datatype dtype, int *dims, int n_dims);

void cuda_tensor_memcpy_cpu_to_gpu(struct cuda_tensor *cu_tens);


struct cuda_tensor *cuda_tensor_from_cllm_tensor_metadata(struct cllm_tensor_metadata *parent);

struct cuda_tensor *cuda_tensor_float_multiply_add(cublasHandle_t handle, struct cuda_tensor* a, struct cuda_tensor* b, struct cuda_tensor *c, const float alpha,
    const float beta);

struct cuda_tensor *cuda_tensor_float_gemm(cublasHandle_t handle, struct cuda_tensor* a, struct cuda_tensor* b, const float alpha);

void cuda_tensor_print(struct cuda_tensor *tensor);



struct cuda_tensor *cuda_tensor_view(struct cuda_tensor *tensor, int start, int stop);




struct cuda_tensor * cuda_tensor_add(struct cuda_tensor *A, struct cuda_tensor *B);
#endif //CLLM_TENSORS_H
