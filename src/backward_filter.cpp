#include "backward_filter.h"
#include "im2col.h"
#include "gemm.h"

void ConvolutionBackwardFilterCpu(int input_n, int input_c, int input_h, int input_w,
                                  int stride_h, int stride_w,
                                  int pad_h, int pad_w,
                                  int dilation_h, int dilation_w,
                                  int group,
                                  int output_c, int output_h, int output_w,
                                  int kernel_h, int kernel_w,
                                  float *x, float *y, float *w)
{
    int M = output_c;
    int N = kernel_h * kernel_w * input_c;
    int K = output_h * output_w;

    float *workspace = new float[K * N];
    for (int n = 0; n < input_n; ++n)
    {
        int x_offset = n * input_c * input_h * input_w;
        im2col_cpu(input_c, input_h, input_w,
                   stride_h, stride_w,
                   pad_h, pad_w,
                   dilation_h, dilation_w,
                   group,
                   kernel_h, kernel_w,
                   output_h, output_w,
                   x + x_offset, workspace);
        int y_offset = n * output_c * output_h * output_w;
        cpu_gemm(false, true, M, N, K, 1.f, y + y_offset, K, workspace, K, 0.f, w, N);
    }

    delete[] workspace;
}