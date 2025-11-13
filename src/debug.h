//
// Created by Sanger Steel on 11/11/25.
//

#ifndef CLLM_DEBUG_H
#define CLLM_DEBUG_H

#include "stdlib.h"

#define CUDA_CHECK(err) \
do { \
cudaError_t e = (err); \
if (e != cudaSuccess) { \
fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
exit(EXIT_FAILURE); \
} \
} while(0)


#ifdef DEBUG
#include "inttypes.h"
#include "stdio.h"
#include "pthread.h"

#define TRACE(fmt, ...) \
do { \
char _debug_log_buffer[1024]; \
int len = snprintf(_debug_log_buffer, sizeof(_debug_log_buffer), \
"[ cllm %s tid%-5" PRIuMAX " | %-10s | %-20s:%-3i ] TRACE: " fmt "\n", \
__TIME__, (uintmax_t)pthread_self(), __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__); \
fwrite(_debug_log_buffer, 1, len, stdout); \
fflush(stdout); \
} while(0)
#define DEBUG_GUARD(stmt) do { stmt; } while (0)
#else
#define TRACE(fmt, ...) ((void)0)
#define DEBUG_GUARD(stmt) ((void)0)
#endif
#endif //CLLM_DEBUG_H