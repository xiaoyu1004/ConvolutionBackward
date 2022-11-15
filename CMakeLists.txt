cmake_minimum_required(VERSION 3.10)
project(convolution_backward LANGUAGES CXX)

option(ENABLE_LOG "enable log" ON)
option(ENABLE_CPU "enable cpu" ON)
option(ENABLE_CUDA "enable cuda" ON)
option(ENABLE_CUDNN "enable cudnn" ON)

option(ENABLE_NVIDIA "enable nvidia" ON)
option(ENABLE_ILUVATAR "enable iluvatar" OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mfma -msse -mavx")

if(ENABLE_NVIDIA)
    if(ENABLE_CUDA)
        enable_language(CUDA)
    endif()
else()
    include(${PROJECT_SOURCE_DIR}/cmake/CUDA.cmake)
endif()

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
    list(APPEND srcs ./src/cuda/backward_filter.cu)
endif()

if(ENABLE_NVIDIA)
    add_executable(${PROJECT_NAME} ${srcs})
    target_include_directories(${PROJECT_NAME} PRIVATE ./src /usr/local/cuda/include)
else()
    cuda_add_executable(${PROJECT_NAME} ${srcs})
    target_include_directories(${PROJECT_NAME} PRIVATE ./src /usr/local/corex/include)
endif()

if(ENABLE_CUDNN)
    target_link_libraries(${PROJECT_NAME} PRIVATE cudnn)
endif()