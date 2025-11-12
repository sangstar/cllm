//
// Created by Sanger Steel on 11/11/25.
//

#ifndef CLLM_LAYERS_H
#define CLLM_LAYERS_H
#include <stdlib.h>

#include "deserialize.h"
#include "tensors.h"

enum layer_type
{
    EMBEDDING = 0,
    DENSE = 1,
};

struct layer
{
    enum layer_type type;
    struct cuda_tensor *weight;
    struct cuda_tensor *bias;
    struct cuda_tensor *(*forward)(cublasHandle_t, struct layer *, struct cuda_tensor *);
};

struct cuda_tensor *dense_forward(cublasHandle_t handle, struct layer *dense, struct cuda_tensor *x);

struct layer *create_dense_layer(struct cuda_tensor *weight, struct cuda_tensor *bias);

#endif //CLLM_LAYERS_H
