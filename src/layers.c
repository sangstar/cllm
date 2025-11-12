//
// Created by Sanger Steel on 11/11/25.
//

#include "layers.h"

#include <string.h>
#include "debug.h"

#define STRLEN_FOR_WEIGHT 6
#define STRLEN_FOR_BIAS 4

struct layer *create_dense_layer(struct cuda_tensor *weight, struct cuda_tensor *bias, const char *name)
{
    struct layer *layer = (struct layer*)safe_malloc(sizeof(struct layer));
    layer->type = DENSE;
    layer->weight = weight;
    layer->bias = bias;
    layer->forward = dense_forward;
    layer->name = name;
    return layer;
}

struct cuda_tensor *dense_forward(cublasHandle_t handle, struct layer *dense, struct cuda_tensor *x)
{
    return cuda_tensor_float_multiply_add(handle, dense->weight, x, dense->bias, 1.0, 1.0);
}

struct cuda_tensor *layernorm_forward(cublasHandle_t handle, struct layer *dense, struct cuda_tensor *x)
{
    return NULL;
}

struct layer *create_layernorm_layer(struct cuda_tensor *weight, struct cuda_tensor *bias, const char *name)
{
        struct layer *layer = (struct layer*)safe_malloc(sizeof(struct layer));
        layer->type = DENSE;
        layer->weight = weight;
        layer->bias = bias;
        layer->forward = layernorm_forward;
        layer->name = name;
        return layer;
}

struct cuda_tensor *embedding_forward(cublasHandle_t handle, struct layer *dense, struct cuda_tensor *x)
{
    return NULL;
}

struct layer *create_embedding_layer(struct cuda_tensor *weight, const char *name)
{
    struct layer *layer = (struct layer*)safe_malloc(sizeof(struct layer));
    layer->type = EMBEDDING;
    layer->weight = weight;
    layer->bias = NULL;
    layer->forward = layernorm_forward;
    layer->name = name;
    return layer;
}

enum layer_type type_from_cllm_tensor_metadata(struct cllm_tensor_metadata *metadata)
{
    if (strstr(metadata->name, "dense") != NULL)
    {
        return DENSE;
    }
    if (strstr(metadata->name, "layernorm") != NULL || strstr(metadata->name, "layer_norm") != NULL)
    {
        return LAYERNORM;
    }
    if (strstr(metadata->name, "embed") != NULL)
    {
        return EMBEDDING;
    }
    return -1;
}

struct layer *layer_from_cllm_tensor_metadata(struct cllm_tensor_metadata *weight, struct cllm_tensor_metadata *bias, const char *name)
{
    enum layer_type type;
    type = type_from_cllm_tensor_metadata(weight);
    if (bias != NULL && type != type_from_cllm_tensor_metadata(bias))
    {
        return NULL;
    }
    switch (type)
    {
    case DENSE:
        return create_dense_layer(cuda_tensor_from_cllm_tensor_metadata(weight),
                                  cuda_tensor_from_cllm_tensor_metadata(bias), name);
    case LAYERNORM:
        return create_layernorm_layer(cuda_tensor_from_cllm_tensor_metadata(weight),
                                      cuda_tensor_from_cllm_tensor_metadata(bias), name);
    case EMBEDDING:
        if (bias != NULL)
        {
            perror("bias for embedding layer not supported");
            exit(1);
        }
        return create_embedding_layer(cuda_tensor_from_cllm_tensor_metadata(weight), name);
    default:
        return create_dense_layer(cuda_tensor_from_cllm_tensor_metadata(weight),
                          cuda_tensor_from_cllm_tensor_metadata(bias), name);
    }
}

int write_suffix_to_buf(char *buf, char *name, size_t len)
{
    if (strcmp(name + len - STRLEN_FOR_BIAS, "bias") == 0)
    {
        snprintf(buf, len - STRLEN_FOR_BIAS, "%s", name);
        return 1;
    }
    if (strcmp(name + len - STRLEN_FOR_WEIGHT, "weight") == 0)
    {
        snprintf(buf, len - STRLEN_FOR_WEIGHT, "%s", name);
        return 2;
    }
    return 0;
}

int write_model_name_to_buf(char *buf, char *param_name, size_t len)
{
    for (int i = 0; i < len; i++)
    {
        if (param_name[i] == '.') return 1;
        buf += sprintf(buf, "%c", param_name[i]);
    }
    return 0;
}

struct model *model_from_cllm_data(struct cllm_data *data)
{
    struct model *model = (struct model *)safe_malloc(sizeof(struct model));

    // Generally a layer per two tensors. This is therefore more than necessary
    struct layer **layers = (struct layer**)safe_malloc(sizeof(struct layer*) * data->num_tensors);
    size_t layer_idx = 0;
    char suffix_a[256] = {0};
    char suffix_b[256] = {0};
    char prefix[64] = {0};
    int wrote_model_name = 0;
    for (int i = 0; i < data->num_tensors; i++)
    {
        struct cllm_tensor_metadata *candidate_a = (struct cllm_tensor_metadata*)data->tensors[i];
        char *candidate_a_name = candidate_a->name;
        if (!wrote_model_name && !write_model_name_to_buf(prefix, candidate_a_name, strlen(candidate_a_name)))
        {
            perror("could not write model name");
            return NULL;
        }
        wrote_model_name = 1;

        int weight_or_bias_a = write_suffix_to_buf(suffix_a, candidate_a_name, strlen(candidate_a_name));
        TRACE("trying to pair candidate %s..", candidate_a->name);
        if (i + 1 < data->num_tensors)
        {
            struct cllm_tensor_metadata *candidate_b = (struct cllm_tensor_metadata *)data->tensors[i + 1];
            char *candidate_b_name = candidate_b->name;
            int weight_or_bias_b = write_suffix_to_buf(suffix_b, candidate_b_name, strlen(candidate_b_name));
            if (strcmp(suffix_a, suffix_b) == 0)
            {
                TRACE("candidate a=%s paired with candidate b=%s", candidate_a_name, candidate_b_name);
                if (weight_or_bias_a == 2 && weight_or_bias_b == 1) layers[layer_idx++] = layer_from_cllm_tensor_metadata(candidate_a, candidate_b, strdup(suffix_a));
            } else if (strstr(candidate_a_name, "embed") != NULL) layers[layer_idx++] = layer_from_cllm_tensor_metadata(candidate_a, NULL, strdup(suffix_a));
        }
        suffix_a[0] = '\0';
        suffix_b[0] = '\0';
    }
    model->name = prefix;
    model->layers = layers;
    model->num_layers = layer_idx;
    return model;
}
