#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <cudnn.h>

#include <iostream>
#include <algorithm>

#define CUDA_CHECK(err)                                                                                     \
    if (err != cudaSuccess)                                                                                 \
    {                                                                                                       \
        printf("cuda error: file: %s line: %d details: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        std::terminate();                                                                                   \
    }

#define CUDNN_CHECK(err)                                                                                      \
    if (err != CUDNN_STATUS_SUCCESS)                                                                          \
    {                                                                                                         \
        printf("cudnn error: file: %s line: %d details: %s\n", __FILE__, __LINE__, cudnnGetErrorString(err)); \
        std::terminate();                                                                                     \
    }

#endif