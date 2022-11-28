set(test_srcs ./nv_test.cpp)

if(ENABLE_NVIDIA)
    add_executable(${PROJECT_NAME} ${test_srcs})
    target_include_directories(${PROJECT_NAME} PRIVATE ./src /usr/local/cuda/include)
else()
    cuda_add_executable(${PROJECT_NAME} ${test_srcs})
    target_include_directories(${PROJECT_NAME} PRIVATE ./src ${CUDA_TOOLKIT_ROOT_DIR}/include)
endif()

if(ENABLE_CUDNN)
    target_link_libraries(${PROJECT_NAME} PRIVATE cudart cudnn)
endif()