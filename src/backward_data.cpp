#include "common.h"
#include "backward_data.h"
#include "col2im.h"
#include "gemm.h"

void ConvolutionBackwardDataCpu(int input_n, int input_c, int input_h, int input_w,
                                int stride_h, int stride_w,
                                int pad_h, int pad_w,
                                int dilation_h, int dilation_w,
                                int group,
                                int output_c, int output_h, int output_w,
                                int kernel_h, int kernel_w,
                                float *w, float *y, float *x)
{
    int M = kernel_h * kernel_w * input_c;
    int N = output_h * output_w;
    int K = output_c;

    int lda = M;
    int ldb = N;
    int ldc = N;

    float alpha = 1.f;
    float beta = 0.f;

    float *workspace = new float[M * N]{0};

    for (int n = 0; n < input_n; ++n)
    {
        int y_offset = n * output_c * output_h * output_w;
        int x_offset = n * input_c * input_h * input_w;
        memset(workspace, 0, M * N * sizeof(float));
        cpu_gemm(true, false, M, N, K, alpha, w, lda, y + y_offset, ldb, beta, workspace, ldc);
        col2im_cpu(input_c, input_h, input_w,
                   stride_h, stride_w,
                   pad_h, pad_w,
                   dilation_h, dilation_w,
                   group,
                   kernel_h, kernel_w,
                   output_h, output_w,
                   workspace, x + x_offset);
    }

    delete[] workspace;
}

void ConvolutionBackwardData(int input_n, int input_c, int input_h, int input_w,
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
    CUDA_CHECK(cudaMemset(d_x, 0, x_size * sizeof(float)));
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
    float *h_w = new float[w_size]{0};
    float *h_y = new float[y_size]{0};
#ifdef ENABLE_CUDA
    float *h_x = new float[x_size]{0};
#endif
#ifdef ENABLE_CUDNN
    float *h_dnn_x = new float[x_size]{0};
#endif
    float *h_ref_x = new float[x_size]{0};

    // init w
    for (int i = 0; i < w_size; ++i)
    {
        h_w[i] = static_cast<float>(i % 3); // x_data[i % 280]
    }

    // init y
    for (int i = 0; i < y_size; ++i)
    {
        h_y[i] = static_cast<float>(i % 5); // 1.f
    }

#ifdef ENABLE_CUDA
    // memcpy host -> device
    CUDA_CHECK(cudaMemcpy(d_w, h_w, w_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, y_size * sizeof(float), cudaMemcpyHostToDevice));
#endif

#ifdef ENABLE_CPU
    // cpu
    ConvolutionBackwardDataCpu(input_n, input_c, input_h, input_w,
                               stride_h, stride_w,
                               pad_h, pad_w,
                               dilation_h, dilation_w,
                               group,
                               output_c, output_h, output_w,
                               kernel_h, kernel_w,
                               h_w, h_y, h_ref_x);
#endif

#ifdef ENABLE_CUDA
    ConvolutionBackwardDataGpu(input_n, input_c, input_h, input_w,
                               stride_h, stride_w,
                               pad_h, pad_w,
                               dilation_h, dilation_w,
                               group,
                               output_c, output_h, output_w,
                               kernel_h, kernel_w,
                               d_w, d_y, d_x);
    CUDA_CHECK(cudaMemcpy(h_x, d_x, x_size * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef ENABLE_CUDNN
    CUDA_CHECK(cudaMemset(d_x, 0, x_size * sizeof(float)));
    // gpu
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

    cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    size_t workspace_size;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle, wDesc, yDesc, convDesc, xDesc, algo, &workspace_size));
    float *workSpace;
    CUDA_CHECK(cudaMalloc(&workSpace, workspace_size));
    CUDNN_CHECK(cudnnConvolutionBackwardData(handle, &alpha, wDesc, d_w, yDesc, d_y, convDesc, algo, workSpace, workspace_size, &beta, xDesc, d_x));

    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(wDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));

    CUDA_CHECK(cudaMemcpy(h_dnn_x, d_x, x_size * sizeof(float), cudaMemcpyDeviceToHost));
#endif
#endif

#ifdef ENABLE_LOG
#define PRINT_OUTPUT(device, ptr)                                                                                        \
    std::cout << #device << ":" << std::endl;                                                                            \
    for (int n = 0; n < input_n; ++n)                                                                                    \
    {                                                                                                                    \
        for (int i = 0; i < input_h; ++i)                                                                                \
        {                                                                                                                \
            for (int j = 0; j < input_c; ++j)                                                                            \
            {                                                                                                            \
                for (int k = 0; k < input_w; ++k)                                                                        \
                {                                                                                                        \
                    std::cout << ptr[n * input_h * input_w * input_c + j * input_h * input_w + i * input_w + k] << "\t"; \
                }                                                                                                        \
                std::cout << "\t\t";                                                                                     \
            }                                                                                                            \
            std::cout << std::endl;                                                                                      \
        }                                                                                                                \
        std::cout << "\n";                                                                                               \
    }

#ifdef ENABLE_CPU
    PRINT_OUTPUT(cpu, h_ref_x);
#endif

#ifdef ENABLE_CUDA
    PRINT_OUTPUT(gpu, h_x);
#ifdef ENABLE_CUDNN
    PRINT_OUTPUT(cudnn, h_dnn_x);
#endif // ENABLE_CUDNN
#endif // ENABLE_CUDA
#undef PRINT_OUTPUT
#endif // ENABLE_LOG

#ifdef ENABLE_CPU
#ifdef ENABLE_CUDA
    // compare
    for (int i = 0; i < x_size; ++i)
    {
        float err = std::abs(h_x[i] - h_ref_x[i]);
        if (err > 1e-4f)
        {
            std::cout << "ERROR: h_x[" << i << "]=" << h_x[i] << " != h_ref_x[" << i << "]=" << h_ref_x[i] << std::endl;
            std::terminate();
        }
    }
    std::cout << "compare pass!" << std::endl;
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
    delete[] h_dnn_x;
#endif
#endif

#ifdef ENABLE_CUDA
    delete[] h_x;
#endif
    delete[] h_w;
    delete[] h_y;
    delete[] h_ref_x;
}