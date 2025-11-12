#include "deserialize.h"
#include "../tensors.h"



int main(void) {
    FILE *f = fopen("../dummy.cllm", "r");
    cublasHandle_t handle;
    cublasCreate(&handle);

    struct cllm_data *data = cllm_data_init(f);
    for (int i = 0; i < data->header->tensor_count; i++) {
        cllm_data_print_tensor(data, i);
    }

    struct cuda_tensor *a = cuda_tensor_from_cllm_tensor_metadata(data->tensors[0]);
    struct cuda_tensor *b = cuda_tensor_from_cllm_tensor_metadata(data->tensors[1]);

    float *C = cuda_tensor_float_gemm(handle, a, b, 1.0f);

    print_tensor(C, a->dims[0], b->dims[1], a->dtype);
    // Cleanup
    cllm_data_free(data);
    cublasDestroy(handle);
    return 0;
    }