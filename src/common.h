#ifndef COMMON_H
#define COMMON_H

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#ifdef ENABLE_CUDNN
#include <cudnn.h>
#endif
#endif

#include <iostream>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>

static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
static std::default_random_engine ge(42);
static std::normal_distribution<float> dist(0.f, 1.f);

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

static float random_data[6] = {0.757686, 0.453634, 0.412995, 0.558537, 0.116952, 0.557779};

#endif