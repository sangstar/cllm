//
// Created by Sanger Steel on 11/11/25.
//

#include "layers.h"

struct layer* create_dense_layer(struct cuda_tensor* weight, struct cuda_tensor* bias)
{
    struct layer *layer = (struct layer *)safe_malloc(sizeof(struct layer));
    layer->type = DENSE;
    layer->weight = weight;
    layer->bias = bias;
    layer->forward = dense_forward;
    return layer;
}

struct cuda_tensor* dense_forward(cublasHandle_t handle, struct layer* dense, struct cuda_tensor* x)
{
    return cuda_tensor_float_multiply_add(handle, dense->weight, x, dense->bias, 1.0, 1.0);
}
