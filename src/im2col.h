#ifndef IM2COL_H
#define IM2COL_H

void im2col_cpu(int input_c, int input_h, int input_w,
                int stride_h, int stride_w,
                int pad_h, int pad_w,
                int dilation_h, int dilation_w,
                int group,
                int kernel_h, int kernel_w,
                int output_h, int output_w,
                const float *x, float *y);

#endif