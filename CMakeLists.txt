cmake_minimum_required(VERSION 3.10)
project(convolution_backward LANGUAGES CXX) # CUDA

option(ENABLE_LOG "enable log" ON)
option(ENABLE_CUDA "enable cuda" ON)
option(ENABLE_CUDNN "enable cudnn" ON)

if(ENABLE_LOG)
    add_definitions(-DENABLE_LOG)
endif()

if(ENABLE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    add_definitions(-DENABLE_CUDA)
    if(ENABLE_CUDNN)
        add_definitions(-DENABLE_CUDNN)
    endif()
endif()

set(srcs ./src/backward_filter.cpp
         ./src/gemm.cpp
         ./src/im2col.cpp
         ./src/main.cpp)

if(ENABLE_CUDA)
    set(APPEND srcs ./src/cuda/backward_filter.cu)
endif()
add_executable(${PROJECT_NAME} ${srcs})
target_include_directories(${PROJECT_NAME} PRIVATE ./src /usr/local/cuda/include)
if(ENABLE_CUDNN)
    target_link_libraries(${PROJECT_NAME} PRIVATE cudnn)
endif()