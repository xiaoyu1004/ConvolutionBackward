cmake_minimum_required(VERSION 3.10)
project(convolution_backward LANGUAGES CXX)

option(ENABLE_LOG "enable log" ON)
option(ENABLE_CPU "enable cpu" ON)
option(ENABLE_CUDA "enable cuda" ON)
option(ENABLE_CUDNN "enable cudnn" ON)

option(ENABLE_NVIDIA "enable nvidia" ON)
option(ENABLE_ILUVATAR "enable iluvatar" OFF)
option(SW_HOME "enable sw_home" OFF)

# cmake .. -DENABLE_LOG=OFF -DENABLE_CPU=ON -DENABLE_CUDA=ON -DENABLE_CUDNN=ON -DENABLE_NVIDIA=OFF -DENABLE_ILUVATAR=ON -DSW_HOME=ON

if(ENABLE_NVIDIA)
    enable_language(CUDA)
endif()

if(ENABLE_ILUVATAR)
    if (SW_HOME)
        set(ROOT_PATH "/opt/sw_home/local/" CACHE PATH "cuda installation root dir")
        set(CUDA_TOOLKIT_ROOT_DIR ${ROOT_PATH}/cuda CACHE PATH "root path to cuda/lib64 cuda/bin etc")
    else()
        set(ROOT_PATH "/usr/local/corex/" CACHE PATH "cuda installation root dir")
        set(CUDA_TOOLKIT_ROOT_DIR ${ROOT_PATH} CACHE PATH "root path to cuda/lib64 cuda/bin etc")
    endif()

    set(COMPILER_PATH ${ROOT_PATH}/bin CACHE PATH "compiler binary path")
    set(CMAKE_C_COMPILER ${COMPILER_PATH}/clang CACHE PATH "C compiler binary path")
    set(CMAKE_CXX_COMPILER ${COMPILER_PATH}/clang++ CACHE PATH "CXX compiler binary path")
    set(CUDA_ARCH "ivcore11" CACHE STRING "CUDA architecture.")

    message(STATUS "ROOT_PATH: ${ROOT_PATH}")
    message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "COMPILER_PATH: ${COMPILER_PATH}")
    message(STATUS "CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
    message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    include(${PROJECT_SOURCE_DIR}/cmake/CUDA.cmake)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mfma -mavx -msse4")

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

if(ENABLE_NVIDIA)
    add_executable(${PROJECT_NAME} ${srcs})
    target_include_directories(${PROJECT_NAME} PRIVATE ./src /usr/local/cuda/include)
else()
    cuda_add_executable(${PROJECT_NAME} ${srcs})
    target_include_directories(${PROJECT_NAME} PRIVATE ./src ${CUDA_TOOLKIT_ROOT_DIR}/include)
endif()

if(ENABLE_CUDNN)
    target_link_libraries(${PROJECT_NAME} PRIVATE cudart cudnn)
endif()
