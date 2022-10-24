#include "gemm.h"

void cpu_gemm(bool ATrans, bool BTrans,
              const int M, const int N, const int K, 
              const float alpha,
              const float *A, const int lda,
              const float *B, const int ldb, 
              const float beta,
              float *C, const int ldc)
{
    for (int p = 0; p < K; ++p)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                C[i * ldc + j] += (ATrans ? A[p * lda + i] : A[i * lda + p]) * (BTrans ? B[j * ldb + p] : B[p * ldb + j]);
            }
        }
    }
}