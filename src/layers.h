//
// Created by Sanger Steel on 11/11/25.
//

#ifndef CLLM_LAYERS_H
#define CLLM_LAYERS_H
#include <stdlib.h>

#include "deserialize.h"
#include "tensors.h"
#include "model.h"

enum layer_type
{
    EMBEDDING = 0,
    DENSE = 1,
    LAYERNORM = 2,
};

struct layer
{
    enum layer_type type;
    struct cuda_tensor *weight;
    struct cuda_tensor *bias;
    struct cuda_tensor *(*forward)(cublasHandle_t, struct layer *, struct cuda_tensor *);
    const char *name;
};

struct cuda_tensor *dense_forward(cublasHandle_t handle, struct layer *dense, struct cuda_tensor *x);

struct layer *create_dense_layer(struct cuda_tensor *weight, struct cuda_tensor *bias, const char *name);

struct model *model_from_cllm_data(struct cllm_data *data);

#endif //CLLM_LAYERS_H
