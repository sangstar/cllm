//
// Created by Sanger Steel on 11/11/25.
//

#include "../deserialize.h"

#include <stdlib.h>
#include <string.h>

size_t cllm_datatype_to_size(cllm_datatype dtype) {
    switch (dtype) {
        case CLLM_INT8:
            return sizeof(int8_t);
        case CLLM_INT16:
            return sizeof(int16_t);
        case CLLM_INT32:
            return sizeof(int32_t);
        case CLLM_INT64:
            return sizeof(int64_t);
        case CLLM_FLOAT32:
            return sizeof(float);
        case CLLM_FLOAT64:
            return sizeof(double);
        default:
            return 0;
    }
}

void cllm_datatype_to_string(char *buf, cllm_datatype dtype) {
    switch (dtype) {
        case CLLM_INT8:
            buf += sprintf(buf, "%s", "int8");
            break;
        case CLLM_INT16:
            buf += sprintf(buf, "%s", "int16");
            break;
        case CLLM_INT32:
            buf += sprintf(buf, "%s", "int32");
            break;
        case CLLM_INT64:
            buf += sprintf(buf, "%s", "int64");
            break;
        case CLLM_FLOAT32:
            buf += sprintf(buf, "%s", "float32");
            break;
        case CLLM_FLOAT64:
            buf += sprintf(buf, "%s", "float64");
            break;
        default:
            buf += sprintf(buf, "%s", "unknown");
    }
}

void *safe_malloc(size_t size)
{
    void *ptr = malloc(size);
    if (ptr == NULL)
    {
        perror("failed to initialize ptr");
        exit(1);
    }
    memset(ptr, 0, size);
    return ptr;
}

void cllm_header_set(struct cllm_header *header, FILE *f) {
    CLLM_FREAD(header->magic, 4, 1, f);
    if (strncmp(header->magic, "CLLM", 4) != 0) {
        perror("invalid header");
        exit(1);
    }
    CLLM_FREAD(&header->version, sizeof(int), 1, f);
    CLLM_FREAD(&header->tensor_count, sizeof(int), 1, f);
    return;
}

void cllm_tensor_metadata_set_metadata(struct cllm_tensor_metadata *metadata, FILE *f) {
    uint16_t name_bytes;
    CLLM_FREAD(&name_bytes, sizeof(uint16_t), 1, f);
    metadata->name = safe_malloc(name_bytes);
    CLLM_FREAD(metadata->name, 1, name_bytes, f);
    metadata->name[name_bytes] = '\0';
    cllm_datatype dtype;
    CLLM_FREAD(&dtype, sizeof(cllm_datatype), 1, f);
    metadata->dtype = dtype;
    CLLM_FREAD(&metadata->n_dims, sizeof(uint16_t), 1, f);
    metadata->dims = safe_malloc(metadata->n_dims * sizeof(uint16_t));
    for (int i = 0; i < metadata->n_dims; i++) {
        CLLM_FREAD(&metadata->dims[i], sizeof(int), 1, f);
    }
    CLLM_FREAD(&metadata->offset, sizeof(int), 1, f);
    size_t total_elems = 1;
    for (int i = 0; i < metadata->n_dims; i++) {
        total_elems *= metadata->dims[i];
    }
    metadata->total_elems = total_elems;
    return;
}

void cllm_tensor_metadata_set_tensor_data(struct cllm_tensor_metadata *metadata, FILE *f) {
    int pos = ftell(f);
    CLLM_FSEEK(f, metadata->offset, SEEK_SET);
    size_t size = cllm_datatype_to_size(metadata->dtype);
    metadata->data = safe_malloc(metadata->total_elems * size);
    CLLM_FREAD(metadata->data, cllm_datatype_to_size(metadata->dtype), metadata->total_elems, f);
    CLLM_FSEEK(f, pos, SEEK_SET);
}

char *tensor_to_buf(void *data, size_t len, cllm_datatype datatype, char *buf) {
    int *int_buf = NULL;
    float *float_buf = NULL;
    int cutoff_nums = len < 8 ? len: 8;
    if (datatype < 5) {
        int_buf = (int *)data;
        for (int i = 0; i < cutoff_nums; i++) {
            buf += sprintf(buf, "%i, ", int_buf[i]);
        }
    }
    else {
        float_buf = (float *)data;
        for (int i = 0; i < cutoff_nums; i++) {
            buf += sprintf(buf, "%f, ", float_buf[i]);
        }
    }
    if (len > cutoff_nums) {
        buf += sprintf(buf, "... ");
    }
    return buf;
}


void print_tensor(void *data, int n, int m, cllm_datatype datatype)
{
    char buf[256];
    char *p = buf;
    p += sprintf(p, "tensor (%i,%i) = [", n, m);
    p = tensor_to_buf(data, n*m, datatype, p);
    p += sprintf(p, "]");
    printf("%s\n", buf);
}


void cllm_data_print_tensor(struct cllm_data *data, int pos) {
    struct cllm_tensor_metadata *tensor = data->tensors[pos];
    char buf[256];
    memset(buf, 0, sizeof(buf));

    char datatype[64];
    memset(datatype, 0, sizeof(datatype));
    cllm_datatype_to_string(datatype, tensor->dtype);

    char tensor_data[512];
    memset(tensor_data, 0, sizeof(tensor_data));

    tensor_to_buf(tensor->data, tensor->total_elems, tensor->dtype, tensor_data);

    char *p = buf;
    p += sprintf(p, "tensor_idx=%i ", pos);
    p += sprintf(p, "name=%s, dtype=%s, n_dims=%llu", tensor->name, datatype, tensor->n_dims);
    for (int i = 0; i < tensor->n_dims; i++) {
        p += sprintf(p, ", dim_%i=%i", i, tensor->dims[i]);
    }
    p += sprintf(p, " offset=%i", tensor->offset);
    p += sprintf(p, " data=[%s]", tensor_data);
    printf("%s\n", buf);
}

struct cllm_data *cllm_data_free(struct cllm_data *data) {
    size_t count = data->header->tensor_count;
    for (int i = 0; i < count; i++) {
        free(data->tensors[i]->name);
        free(data->tensors[i]->data);
        free(data->tensors[i]);
    }
    free(data->header);
    free(data->tensors);
    free(data);
    return NULL;
}

struct cllm_data *cllm_data_init(FILE *f) {
    CLLM_FSEEK(f, 0, SEEK_SET);
    struct cllm_data *data = safe_malloc(sizeof(struct cllm_data));
    struct cllm_header *header = safe_malloc(sizeof(struct cllm_header));
    cllm_header_set(header, f);
    struct cllm_tensor_metadata **tensors = safe_malloc(sizeof(struct cllm_tensor_metadata *) * header->tensor_count);
    for (int i = 0; i < header->tensor_count; i++) {
        struct cllm_tensor_metadata *metadata = safe_malloc(sizeof(struct cllm_tensor_metadata));
        cllm_tensor_metadata_set_metadata(metadata, f);
        tensors[i] = metadata;
    }
    for (int i = 0; i < header->tensor_count; i++) {
        cllm_tensor_metadata_set_tensor_data(tensors[i], f);
    }
    data->header = header;
    data->tensors = tensors;

    return data;
}
