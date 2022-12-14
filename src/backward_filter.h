#ifndef BACKWARD_FILTER_H
#define BACKWARD_FILTER_H

void ConvolutionBackwardFilterCpu(int input_n, int input_c, int input_h, int input_w,
                                  int stride_h, int stride_w,
                                  int pad_h, int pad_w,
                                  int dilation_h, int dilation_w,
                                  int group,
                                  int output_c, int output_h, int output_w,
                                  int kernel_h, int kernel_w,
                                  float *x, float *y, float *w);

#ifdef ENABLE_CUDA
void ConvolutionBackwardFilterGpu(int input_n, int input_c, int input_h, int input_w,
                                  int stride_h, int stride_w,
                                  int pad_h, int pad_w,
                                  int dilation_h, int dilation_w,
                                  int group,
                                  int output_c, int output_h, int output_w,
                                  int kernel_h, int kernel_w,
                                  float *x, float *y, float *w);
#endif

void ConvolutionBackwardFilter(int input_n, int input_c, int input_h, int input_w,
                               int output_c, int kernel_h, int kernel_w,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w,
                               int dilation_h, int dilation_w,
                               int group);

#endif