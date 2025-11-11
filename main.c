#include "deserialize.h"

int main(void) {
    FILE *f = fopen("../dummy.cllm", "r");

    struct cllm_data *data = cllm_data_init(f);
    for (int i = 0; i < data->header->tensor_count; i++) {
        cllm_data_print_tensor(data, i);
    }
    cllm_data_free(data);
    return 0;
}