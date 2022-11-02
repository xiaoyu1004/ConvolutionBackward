cmake_minimum_required(VERSION 3.10)
project(convolution_backward LANGUAGES CXX)

if(ENABLE_CUDA)
    enable_language(CUDA)
endif()

option(ENABLE_LOG "enable log" ON)
option(ENABLE_CPU "enable cpu" ON)
option(ENABLE_CUDA "enable cuda" ON)
option(ENABLE_CUDNN "enable cudnn" ON)

if(ENABLE_LOG)
    add_definitions(-DENABLE_LOG)
endif()

if(ENABLE_CPU)
    add_definitions(-DENABLE_CPU)
endif()

if(ENABLE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 70)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    add_definitions(-DENABLE_CUDA)
    if(ENABLE_CUDNN)
        add_definitions(-DENABLE_CUDNN)
    endif()
endif()

set(srcs ./src/backward_filter.cpp
         ./src/backward_data.cpp
         ./src/gemm.cpp
         ./src/im2col.cpp
         ./src/col2im.cpp
         ./src/main.cpp)

if(ENABLE_CUDA)
    list(APPEND srcs ./src/cuda/backward_filter.cu
                     ./src/cuda/backward_data.cu)
endif()
add_executable(${PROJECT_NAME} ${srcs})
target_include_directories(${PROJECT_NAME} PRIVATE ./src /usr/local/cuda/include)
if(ENABLE_CUDNN)
    target_link_libraries(${PROJECT_NAME} PRIVATE cudnn)
endif()