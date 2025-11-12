//
// Created by Sanger Steel on 11/11/25.
//

#ifndef CLLM_TENSORS_H
#define CLLM_TENSORS_H
#include <cublas_v2.h>

#include "deserialize.h"


struct cuda_tensor
{
    const void *cpu_data;
    void *gpu_data;
    int *dims;
    cllm_datatype dtype;
    struct cllm_tensor_metadata *parent;
    int copied;
};


void cuda_tensor_memcpy_cpu_to_gpu(struct cuda_tensor *cu_tens);


struct cuda_tensor *cuda_tensor_from_cllm_tensor_metadata(struct cllm_tensor_metadata *parent);

float *cuda_tensor_float_gemm(cublasHandle_t handle, struct cuda_tensor *a, struct cuda_tensor *b, const float alpha, const float beta);


#endif //CLLM_TENSORS_H
