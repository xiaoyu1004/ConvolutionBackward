#include "common.h"
#include "backward_filter.h"

void ConvolutionBackwardFilterDnn(int input_n, int input_c, int input_h, int input_w,
                                  int output_c, int kernel_h, int kernel_w,
                                  int stride_h, int stride_w,
                                  int pad_h, int pad_w,
                                  int dilation_h, int dilation_w,
                                  int group,
                                  cudnnConvolutionBwdFilterAlgo_t algo,
                                  float *d_x, float *d_y, float *d_w)
{
    float alpha = 1.f;
    float beta = 0.f;
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnFilterDescriptor_t wDesc;
    cudnnTensorDescriptor_t xDesc, yDesc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&wDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, output_c, input_c, kernel_h, kernel_w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, input_c, input_h, input_w));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_n, output_c, output_h, output_w));

    cudnnConvolutionDescriptor_t convDesc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1; // CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED
    size_t workspace_size;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, xDesc, yDesc, convDesc, wDesc, algo, &workspace_size));
    float *workSpace;
    CUDA_CHECK(cudaMalloc(&workSpace, workspace_size));
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle, &alpha, xDesc, d_x, yDesc, d_y, convDesc, algo, workSpace, workspace_size, &beta, wDesc, d_w));

    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(wDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));

    CUDA_CHECK(cudaDeviceSynchronize());
}

void ConvolutionBackwardFilter(int input_n, int input_c, int input_h, int input_w,
                               int output_c, int kernel_h, int kernel_w,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dilation_h, int dilation_w,
                               int group)
{
    int x_size = input_n * input_c * input_h * input_w;
#ifdef ENABLE_CUDA
    // malloc device
    float *d_x;
    CUDA_CHECK(cudaMalloc(&d_x, x_size * sizeof(float)));
#endif

    int w_size = output_c * input_c * kernel_h * kernel_w;
#ifdef ENABLE_CUDA
    float *d_w;
    CUDA_CHECK(cudaMalloc(&d_w, w_size * sizeof(float)));
#endif

    int khd = (kernel_h - 1) * dilation_h + 1;
    int kwd = (kernel_w - 1) * dilation_w + 1;
    int output_h = (input_h - khd + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kwd + 2 * pad_w) / stride_w + 1;

    int y_size = input_n * output_c * output_h * output_w;
#ifdef ENABLE_CUDA
    float *d_y;
    CUDA_CHECK(cudaMalloc(&d_y, y_size * sizeof(float)));
#endif

    // malloc host
    float *h_x = new float[x_size]{0};
#ifdef ENABLE_CUDA
    float *h_w = new float[w_size]{0};
#endif
    float *h_ref_w = new float[w_size]{0};
    float *h_y = new float[y_size]{0};

    // init x
    for (int i = 0; i < x_size; ++i)
    {
        // h_x[i] = static_cast<float>(i % 19 - 11); // x_data[i % 280]
        // h_x[i] = dist(ge);
        h_x[i] = h_x_demo[i % 1000];
    }

    // init y
    for (int i = 0; i < y_size; ++i)
    {
        // h_y[i] = static_cast<float>(i % 9 - 5); // 1.f
        h_y[i] = h_y_demo[i % 1000];
    }

#ifdef ENABLE_CUDA
    // memcpy host -> device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, x_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, y_size * sizeof(float), cudaMemcpyHostToDevice));
#endif

#ifdef ENABLE_CUDA
#ifdef ENABLE_CUDNN
    cudnnConvolutionBwdFilterAlgo_t algo1 = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    cudnnConvolutionBwdFilterAlgo_t algo2 = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;

    ConvolutionBackwardFilterDnn(input_n, input_c, input_h, input_w,
                                 output_c, kernel_h, kernel_w,
                                 stride_h, stride_w,
                                 pad_h, pad_w,
                                 dilation_h, dilation_w,
                                 group,
                                 algo,
                                 d_x, d_y, d_w);
    CUDA_CHECK(cudaMemcpy(h_ref_w, d_w, w_size * sizeof(float), cudaMemcpyDeviceToHost));

    ConvolutionBackwardFilterDnn(input_n, input_c, input_h, input_w,
                                 output_c, kernel_h, kernel_w,
                                 stride_h, stride_w,
                                 pad_h, pad_w,
                                 dilation_h, dilation_w,
                                 group,
                                 algo,
                                 d_x, d_y, d_w);
    CUDA_CHECK(cudaMemcpy(h_w, d_w, w_size * sizeof(float), cudaMemcpyDeviceToHost));
#endif
#endif

#ifdef ENABLE_CPU
#ifdef ENABLE_CUDA
    // compare
    float L2 = 0.F;
    float max_diff = 0.f;
    for (int i = 0; i < w_size; ++i)
    {
        L2 += std::pow(h_w[i] - h_ref_w[i], 2);
        max_diff = std::max(std::abs(h_w[i] - h_ref_w[i]), max_diff);

        // if (err > 1e-1f)
        // {
        //     std::cout << std::setprecision(10) << "ERROR: h_w[" << i << "]=" << h_w[i] << " != h_ref_w[" << i << "]=" << h_ref_w[i] << std::endl;
        //     std::terminate();
        // }
    }

    L2 = std::sqrt(L2);
    bool bres = L2 > 1e-2f;
    if (bres)
    {
        std::cout << std::setprecision(10) << "ERROR! L2: " << L2 << " max diff: " << max_diff << std::endl;
    }

    std::cout << "compare " << (bres ? "failed" : "pass") << "!" << std::endl;
#endif
#endif

#ifdef ENABLE_CUDA
    // free
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_y));
#endif

#ifdef ENABLE_CUDA
#ifdef ENABLE_CUDNN
    CUDA_CHECK(cudaFree(workSpace));
#endif
#endif

    delete[] h_x;
#ifdef ENABLE_CUDA
    delete[] h_w;
#endif
    delete[] h_ref_w;
    delete[] h_y;
}

int main()
{
    int input_n = 32;
    int input_c = 64;
    int input_h = 224;
    int input_w = 224;
    int output_c = 64;
    int kernel_h = 3;
    int kernel_w = 3;
    int stride_h = 1;
    int stride_w = 1;
    int pad_h = 1;
    int pad_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    int group = 1;

    ConvolutionBackwardFilter(input_n, input_c, input_h, input_w,
                              output_c, kernel_h, kernel_w,
                              stride_h, stride_w,
                              pad_h, pad_w,
                              dilation_h, dilation_w,
                              group);
}