#include "col2im.h"

void col2im_cpu(int input_c, int input_h, int input_w,
                int stride_h, int stride_w,
                int pad_h, int pad_w,
                int dilation_h, int dilation_w,
                int group,
                int kernel_h, int kernel_w,
                int output_h, int output_w,
                const float *x, float *y)
{
    int h_col = kernel_h * kernel_w * input_c;
    int w_col = output_h * output_w;

    for (int h = 0; h < h_col; ++h)
    {
        int ic = h / (kernel_h * kernel_w);
        int kh = h % (kernel_h * kernel_w) / kernel_w;
        int kw = h % (kernel_h * kernel_w) % kernel_w;
        for (int w = 0; w < w_col; ++w)
        {
            int oh = w / output_w;
            int ow = w % output_w;
            int ih = oh * stride_h + kh - pad_h;
            int iw = ow * stride_w + kw - pad_w;
            if (ih < 0 || ih >= input_h || iw < 0 || iw >= input_w)
            {
                continue;
            }
            int y_idx = (ic * input_h + ih) * input_w + iw;
            y[y_idx] += x[h * w_col + w];
        }
    }
}