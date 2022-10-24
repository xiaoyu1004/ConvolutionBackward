#include "im2col.h"

void im2col_cpu(int input_c, int input_h, int input_w,
                int stride_h, int stride_w,
                int pad_h, int pad_w,
                int dilation_d, int dilation_w,
                int group,
                int kernel_h, int kernel_w,
                int output_h, int output_w,
                const float *x, float *y)
{
    int col_h = kernel_h * kernel_w * input_c;
    int col_w = output_h * output_w;

    for (int h = 0; h < col_h; ++h)
    {
        int ic = h / (kernel_h * kernel_w);
        int kh = h % (kernel_h * kernel_w) / kernel_w;
        int kw = h % (kernel_h * kernel_w) % kernel_w;
        for (int w = 0; w < col_w; ++w)
        {
            int oh = w / output_w;
            int ow = w % output_w;
            
            float val = 0;
            int ih = oh * stride_h + kh - pad_h;
            int iw = ow * stride_w + kw - pad_w;
            if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w)
            {
                int x_idx = (ic * input_h + ih) * input_w + iw;
                val = x[x_idx];
            }
            int y_idx = h * col_w + w;
            y[y_idx] = val;
        }
    }
}