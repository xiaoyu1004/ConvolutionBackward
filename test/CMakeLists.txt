set(test_srcs ./nv_test.cpp)
set(module_name nvtest)

if(ENABLE_NVIDIA)
    add_executable(${module_name} ${test_srcs})
    target_include_directories(${module_name} PRIVATE ../src /usr/local/cuda/include)
else()
    cuda_add_executable(${module_name} ${test_srcs})
    target_include_directories(${module_name} PRIVATE ../src ${CUDA_TOOLKIT_ROOT_DIR}/include)
endif()

if(ENABLE_CUDNN)
    target_link_libraries(${module_name} PRIVATE cudart cudnn)
endif()