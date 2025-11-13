//
// Created by Sanger Steel on 11/13/25.
//

#ifndef CLLM_TENSOR_OPS_CUH
#define CLLM_TENSOR_OPS_CUH

__global__ void add_kernel(const float *A, const float *B, float *C, int n);

struct cuda_tensor *cuda_tensor_add(struct cuda_tensor *A, struct cuda_tensor *B);
#endif //CLLM_TENSOR_OPS_CUH