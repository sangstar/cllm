//
// Created by Sanger Steel on 11/11/25.
//

#ifndef CLLM_DESERIALIZE_H
#define CLLM_DESERIALIZE_H
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>


typedef uint8_t cllm_datatype;

enum {
    CLLM_UNKNOWN = 0,
    CLLM_INT8 = 1,
    CLLM_INT16 = 2,
    CLLM_INT32 = 3,
    CLLM_INT64 = 4,
    CLLM_FLOAT32 = 5,
    CLLM_FLOAT64 = 6,
};

size_t cllm_datatype_to_size(cllm_datatype dtype);

void cllm_datatype_to_string(char *buf, cllm_datatype dtype);

struct cllm_header {
    char magic[4];
    int version;
    int tensor_count;
};

struct cllm_tensor_metadata {
    char *name;
    cllm_datatype dtype;
    uint16_t n_dims;
    int *dims;
    int offset;
    size_t total_elems;
    void *data;
};

struct cllm_data {
    struct cllm_header *header;
    struct cllm_tensor_metadata **tensors;
};


#define CLLM_FREAD(buf, size, n, f) \
do { \
    fread(buf, size, n, f); \
    printf("cursor=%i\n", ftell(f)); \
} while (0)

#define CLLM_FSEEK(f, offset, whence) \
do { \
    fseek(f, offset, whence); \
    printf("cursor=%i\n", ftell(f)); \
} while(0)


void cllm_header_set(struct cllm_header *header, FILE *f);

void cllm_tensor_metadata_set_metadata(struct cllm_tensor_metadata *metadata, FILE *f);

void cllm_tensor_metadata_set_tensor_data(struct cllm_tensor_metadata *metadata, FILE* f);

void cllm_data_tensor_to_buf(void *data, size_t len, cllm_datatype datatype, char *buf);

void cllm_data_print_tensor(struct cllm_data *rose, int pos);

struct cllm_data *cllm_data_init(FILE *f);

struct cllm_data *cllm_data_free(struct cllm_data *rose);

#endif //ROSETTA_DESERIALIZE_H
