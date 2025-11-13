//
// Created by Sanger Steel on 11/13/25.
//

#include "tensor_ops.cuh"

#include "debug.h"
#include "deserialize.h"
#include "tensors.h"


__global__ void add_kernel(const float *A, const float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}


__global__ void layernorm_kernel(const float *x, const float *gamma, const float *beta, float *y, int hidden_dim, float eps)
{
    int row = blockIdx.x;
    float mean = 0.f, var = 0.f;
    for (int i = 0; i < hidden_dim; i++)
    {
        mean += x[i];
    }
    mean /= (float)hidden_dim;
    for (int i = 0; i < hidden_dim; i++)
    {
        var += (x[i] - mean) * (x[i] - mean);
    }

    var /= (float)hidden_dim;

    float inv_std = 1.f / sqrtf(var / (hidden_dim + eps));

    for (int i = 0; i < hidden_dim; i++)
    {
        float norm = (x[i] - mean) * inv_std;
        y[i] = norm * gamma[i] + beta[i];
    }
}

void layernorm_forward(struct cuda_tensor *x, struct layer *layernorm_layer);
