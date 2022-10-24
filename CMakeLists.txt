cmake_minimum_required(VERSION 3.10)
project(convolution_backward LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

option(ENABLE_LOG "enable log" ON)
option(ENABLE_CUDA "enable cuda" ON)
option(ENABLE_CUDNN "enable cudnn" ON)

if(ENABLE_LOG)
    add_definitions(-DENABLE_LOG)
endif()

if(ENABLE_CUDA)
    add_definitions(-DENABLE_CUDA)
endif()

if(ENABLE_CUDNN)
    add_definitions(-DENABLE_CUDNN)
endif()

set(srcs ./src/cuda/backward_filter.cu
         ./src/backward_filter.cpp
         ./src/gemm.cpp
         ./src/im2col.cpp
         ./src/main.cpp)
add_executable(${PROJECT_NAME} ${srcs})
target_include_directories(${PROJECT_NAME} PRIVATE ./src /usr/local/cuda/include)
target_link_libraries(${PROJECT_NAME} cudnn)