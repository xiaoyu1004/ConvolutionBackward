#include "backward_data.h"
#include "common.h"

template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_N, const int BLOCK_SIZE_K, const int THREAD_SIZE_X, const int THREAD_SIZE_Y>
__global__ void implicitConvolutionBackwardData(int input_n, int input_c, int input_h, int input_w,
                                                int stride_h, int stride_w,
                                                int pad_h, int pad_w,
                                                int dilation_h, int dilation_w,
                                                int group,
                                                int output_c, int output_h, int output_w,
                                                int kernel_h, int kernel_w,
                                                float *w, float *y, float *x)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    __shared__ float SW[BLOCK_SIZE_M][BLOCK_SIZE_K];
    __shared__ float SY[BLOCK_SIZE_K][BLOCK_SIZE_N];

    int M = input_c;
    int N = input_n * input_h * input_w;
    int K = output_c * kernel_h * kernel_w;

    int output_h_dilation = (output_h - 1) * stride_h + 1;
    int output_w_dilation = (output_w - 1) * stride_w + 1;

    int n = (bx * BLOCK_SIZE_N + tx) / (input_h * input_w);
    int n_rest = (bx * BLOCK_SIZE_N + tx) % (input_h * input_w);
    int ic = by * BLOCK_SIZE_M + ty;
    int ih = n_rest / input_w;
    int iw = n_rest % input_w;

    float sum = 0.f;

    for (int idx = 0; idx < K; idx += BLOCK_SIZE_K)
    {
        // load weight from gmem to smem
        float w_val = 0.f;
        if (ic < input_c && (idx + tx) < K)
        {
            int oc = (idx + tx) / (kernel_h * kernel_w);
            int w_rest_offset = (idx + tx) % (kernel_h * kernel_w);
            int w_idx = (oc * input_c + ic + 1) * kernel_h * kernel_w - w_rest_offset - 1;
            w_val = w[w_idx];
        }
        SW[ty][tx] = w_val;

        // load y from gmem to smem
        float y_val = 0.f;
        if ((idx + ty) < K && (bx * BLOCK_SIZE_N + tx) < N)
        {
            int oc = (idx + ty) / (kernel_h * kernel_w);
            int y_rest_offset = (idx + ty) % (kernel_h * kernel_w);
            int kh = y_rest_offset / kernel_w;
            int kw = y_rest_offset % kernel_w;

            int oh_stride = ih + kh - (kernel_h - pad_h - 1);
            int ow_stride = iw + kw - (kernel_w - pad_w - 1);

            int oh = oh_stride / stride_h;
            int ow = ow_stride / stride_w;

            if (oh >= 0 && oh < output_h_dilation && ow >= 0 && ow < output_w_dilation && oh_stride % stride_h == 0 && ow_stride % stride_w == 0)
            {
                int y_idx = ((n * output_c + oc) * output_h + oh) * output_w + ow;
                y_val = y[y_idx];
            }
        }
        SY[ty][tx] = y_val;

        __syncthreads();

        // compute
        for (int p = 0; p < BLOCK_SIZE_K; ++p)
        {
            sum += SW[ty][p] * SY[p][tx];
        }

        __syncthreads();
    }

    // store
    if (ic < M && (bx * BLOCK_SIZE_N + tx) < N)
    {
        int x_idx = ((n * input_c + ic) * input_h + ih) * input_w + iw;
        x[x_idx] = sum;
    }
}

void ConvolutionBackwardDataGpu(int input_n, int input_c, int input_h, int input_w,
                                int stride_h, int stride_w,
                                int pad_h, int pad_w,
                                int dilation_h, int dilation_w,
                                int group,
                                int output_c, int output_h, int output_w,
                                int kernel_h, int kernel_w,
                                float *w, float *y, float *x)
{
    constexpr int TILE_THREADS_X = 32;
    constexpr int TILE_THREADS_Y = 32;

    constexpr int THREAD_SIZE_X = 1;
    constexpr int THREAD_SIZE_Y = 1;

    constexpr int BLOCK_SIZE_M = THREAD_SIZE_Y * TILE_THREADS_Y;
    constexpr int BLOCK_SIZE_N = THREAD_SIZE_X * TILE_THREADS_X;
    constexpr int BLOCK_SIZE_K = 32;

    dim3 dimBlock(TILE_THREADS_X, TILE_THREADS_Y);
    dim3 dimGrid((input_n * input_h * input_w + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N, (output_c * kernel_h * kernel_w + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);
    implicitConvolutionBackwardData<BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, THREAD_SIZE_X, THREAD_SIZE_Y><<<dimGrid, dimBlock>>>(input_n, input_c, input_h, input_w,
                                                                                                                                   stride_h, stride_w,
                                                                                                                                   pad_h, pad_w,
                                                                                                                                   dilation_h, dilation_w,
                                                                                                                                   group,
                                                                                                                                   output_c, output_h, output_w,
                                                                                                                                   kernel_h, kernel_w,
                                                                                                                                   w, y, x);
    CUDA_CHECK(cudaDeviceSynchronize());
}