//
// Created by Sanger Steel on 11/12/25.
//

#ifndef CLLM_MODEL_H
#define CLLM_MODEL_H
#include <stddef.h>

#include "layers.h"

struct model
{
    struct layer **layers;
    size_t num_layers;
    const char* name;
    cublasHandle_t handle;
};

struct cuda_tensor *model_forward(struct model *model, struct cuda_tensor *inp);

#endif //CLLM_MODEL_H
