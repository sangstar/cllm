#include <stdlib.h>

#include "deserialize.h"
#include "tensors.h"
#include "layers.h"



int main(void) {
    int count = 0;
    cudaGetDeviceCount(&count);
    printf("Visible GPUs: %d\n", count);


    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA device 0: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    cudaFree(0);  // Forces context creation

    FILE *f = fopen("../pythia-160m.cllm", "r");
    cublasHandle_t handle;
    cublasCreate(&handle);

    struct cllm_data *data = cllm_data_init(f);
    for (int i = 0; i < data->header->tensor_count; i++) {
        cllm_data_print_tensor(data, i);
    }

    struct cuda_tensor *a = cuda_tensor_from_cllm_tensor_metadata(data->tensors[0]);
    struct cuda_tensor *b = cuda_tensor_from_cllm_tensor_metadata(data->tensors[1]);

    struct cuda_tensor *C = cuda_tensor_float_gemm(handle, a, b, 1.0f);

    cuda_tensor_print(C);
    // Cleanup
    cllm_data_free(data);
    cublasDestroy(handle);
    cudaFree(0);
    return 0;
}