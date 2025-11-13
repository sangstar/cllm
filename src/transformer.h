//
// Created by Sanger Steel on 11/13/25.
//

#ifndef CLLM_TRANSFORMER_H
#define CLLM_TRANSFORMER_H
#include "layers.h"

struct attn_layer
{
    struct layer *qkv_dense;
    struct layer *rope;
    struct layer *sdpa;
    struct layer *out_proj;
};

struct transformer_layer
{
    struct layer *inp_layernorm;
    struct attn_layer *attention_layer;
    struct layer *outp_layernorm;
    struct layer *mlp;
};

// struct cuda_tensor *transformer_layer_forward(struct transformer_layer *layer, struct cuda_tensor *x)
// {
//     struct cuda_tensor *normed = layer->inp_layernorm->forward(x);
//     struct cuda_tensor *qkv = layer->attention_layer->qkv_dense->forward(normed);
//     struct cuda_tensor *q = cuda_tensor_slice(qkv, Q_SLICE);
//     struct cuda_tensor *k = cuda_tensor_slice(qkv, K_SLICE);
//     struct cuda_tensor *v = cuda_tensor_slice(qkv, V_SLICE);
//     layer->attention_layer->rope->apply(q, k);
//     struct cuda_tensor *sdpa_output = layer->attention_layer->sdpa->forward(q, k, v, CAUSAL_MASK);
//     struct cuda_tensor *attn_proj = layer->attention_layer->out_proj->forward(sdpa_output);
//     struct cuda_tensor *y = cuda_tensor_add(x, attn_proj);
//     struct cuda_tensor *y_normed = layer->outp_layernorm->forward(y);
//     struct cuda_tensor *ff_residual = layer->mlp->forward(y_normed);
//     struct cuda_tensor *output = cuda_tensor_add(y, ff_residual);
//     return output;
// }

#endif //CLLM_TRANSFORMER_H