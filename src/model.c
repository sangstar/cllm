//
// Created by Sanger Steel on 11/12/25.
//

#include "model.h"

// TODO: Obviously too simplistic
//       Need to implement: transformer blocks, forwards for attention layers
struct cuda_tensor * model_forward(struct model *model, struct cuda_tensor *inp)
{
    for (int i = 0; i < model->num_layers; i++)
    {
        struct layer *layer = model->layers[i];
        inp = layer->forward(model->handle, layer, inp);
    }
    return inp;
}
