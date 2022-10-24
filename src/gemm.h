#ifndef GEMM_H
#define GEMM_H

void cpu_gemm(bool ATrans, bool BTrans,
              const int M, const int N, const int K, 
              const float alpha,
              const float *A, const int lda,
              const float *B, const int ldb, 
              const float beta,
              float *C, const int ldc);

#endif