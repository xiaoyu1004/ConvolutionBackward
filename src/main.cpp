#include "common.h"
#include "backward_filter.h"

#include <map>
#include <string>
#include <vector>

void TestBackward(int input_n, int input_c, int input_h, int input_w,
                  int output_c, int kernel_h, int kernel_w,
                  int stride_h, int stride_w,
                  int pad_h, int pad_w,
                  int dilation_h, int dilation_w,
                  int group)
{
    // malloc device
    int x_size = input_n * input_c * input_h * input_w;
    float *d_x;
    CUDA_CHECK(cudaMalloc(&d_x, x_size * sizeof(float)));

    int w_size = output_c * input_c * kernel_h * kernel_w;
    float *d_w;
    CUDA_CHECK(cudaMalloc(&d_w, w_size * sizeof(float)));

    int khd = (kernel_h - 1) * dilation_h + 1;
    int kwd = (kernel_w - 1) * dilation_w + 1;
    int output_h = (input_h - khd + 2 * pad_h) / stride_h + 1;
    int output_w = (input_w - kwd + 2 * pad_w) / stride_w + 1;

    int y_size = input_n * output_c * output_h * output_w;
    float *d_y;
    CUDA_CHECK(cudaMalloc(&d_y, y_size * sizeof(float)));

    // malloc host
    float *h_x = new float[x_size]{0};
    float *h_w = new float[w_size]{0};
    float *h_ref_w = new float[w_size]{0};
    float *h_y = new float[y_size]{0};

    // init x
    for (int i = 0; i < x_size; ++i)
    {
        h_x[i] = static_cast<float>(i % 10);
    }

    // init y
    for (int i = 0; i < y_size; ++i)
    {
        h_y[i] = static_cast<float>(i % 5);
    }

    // memcpy host -> device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, x_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, y_size * sizeof(float), cudaMemcpyHostToDevice));

    // cpu
    ConvolutionBackwardFilterCpu(input_n, input_c, input_h, input_w,
                                 stride_h, stride_w,
                                 pad_h, pad_w,
                                 dilation_h, dilation_w,
                                 group,
                                 output_c, output_h, output_w,
                                 kernel_h, kernel_w,
                                 h_x, h_y, h_ref_w);

#ifdef ENABLE_CUDNN
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

    cudnnConvolutionBwdFilterAlgo_t algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    size_t workspace_size;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle, xDesc, yDesc, convDesc, wDesc, algo, &workspace_size));
    float *workSpace;
    CUDA_CHECK(cudaMalloc(&workSpace, workspace_size * sizeof(float)));
    CUDNN_CHECK(cudnnConvolutionBackwardFilter(handle, &alpha, xDesc, d_x, yDesc, d_y, convDesc, algo, workSpace, workspace_size, &beta, wDesc, d_w));

    CUDNN_CHECK(cudnnDestroy(handle));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(wDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(convDesc));

    CUDA_CHECK(cudaMemcpy(h_w, d_w, w_size * sizeof(float), cudaMemcpyDeviceToHost));
#endif

    // compare
    for (int i = 0; i < w_size; ++i)
    {
        float err = std::abs(h_w[i] - h_ref_w[i]);
        if (err > 1e-4f)
        {
            std::cout << "ERROR: h_w[" << i << "]=" << h_w[i] << " != h_ref_w[" << i << "]=" << h_ref_w[i] << std::endl;
            std::terminate();
        }
    }
    std::cout << "compare pass!" << std::endl;

#ifdef ENABLE_LOG
    std::cout << "cpu:" << std::endl;
    for (int n = 0; n < output_c; ++n)
    {
        for (int i = 0; i < kernel_h; ++i)
        {
            for (int j = 0; j < input_c; ++j)
            {
                for (int k = 0; k < kernel_w; ++k)
                {
                    std::cout << h_ref_w[n * kernel_h * kernel_w * input_c + j * kernel_h * kernel_w + i * kernel_w + k] << "\t";
                }
                std::cout << "\t\t";
            }
            std::cout << std::endl;
        }
        std::cout << "\n";
    }

#ifdef ENABLE_CUDNN
    std::cout << "gpu(cudnn):" << std::endl;
    for (int n = 0; n < output_c; ++n)
    {
        for (int i = 0; i < kernel_h; ++i)
        {
            for (int j = 0; j < input_c; ++j)
            {
                for (int k = 0; k < kernel_w; ++k)
                {
                    std::cout << h_w[n * kernel_h * kernel_w * input_c + j * kernel_h * kernel_w + i * kernel_w + k] << "\t";
                }
                std::cout << "\t\t";
            }
            std::cout << std::endl;
        }
        std::cout << "\n";
    }
#endif // ENABLE_CUDNN
#endif // ENABLE_LOG

    // free
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_y));

#ifdef ENABLE_CUDNN
    CUDA_CHECK(cudaFree(workSpace));
#endif

    delete[] h_x;
    delete[] h_w;
    delete[] h_ref_w;
    delete[] h_y;
}

int main()
{
    std::vector<std::map<std::string, int>> conv_data =
        {
            {{"n", 2}, {"c", 3}, {"h", 4}, {"w", 4}, {"oc", 2}, {"kh", 3}, {"kw", 3}, {"sh", 1}, {"sw", 1}, {"ph", 0}, {"pw", 0}, {"dh", 1}, {"dw", 1}, {"g", 1}}};

    for (std::map<std::string, int> &m : conv_data)
    {
        TestBackward(m["n"], m["c"], m["h"], m["w"], m["oc"], m["kh"], m["kw"], m["sh"], m["sw"], m["ph"], m["pw"], m["dh"], m["dw"], m["g"]);
    }
}