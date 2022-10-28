#ifndef BACKWARD_DATA_H
#define BACKWARD_DATA_H

void ConvolutionBackwardDataCpu(int input_n, int input_c, int input_h, int input_w,
                                int stride_h, int stride_w,
                                int pad_h, int pad_w,
                                int dilation_h, int dilation_w,
                                int group,
                                int output_c, int output_h, int output_w,
                                int kernel_h, int kernel_w,
                                float *w, float *y, float *x);

#ifdef ENABLE_CUDA
void ConvolutionBackwardDataGpu(int input_n, int input_c, int input_h, int input_w,
                                int stride_h, int stride_w,
                                int pad_h, int pad_w,
                                int dilation_h, int dilation_w,
                                int group,
                                int output_c, int output_h, int output_w,
                                int kernel_h, int kernel_w,
                                float *w, float *y, float *x);
#endif

void ConvolutionBackwardData(int input_n, int input_c, int input_h, int input_w,
                             int output_c, int kernel_h, int kernel_w,
                             int stride_h, int stride_w,
                             int pad_h, int pad_w,
                             int dilation_h, int dilation_w,
                             int group);

#endif