#ifndef NN_BLAS_H
#define NN_BLAS_H

#include "nn_core.h"

/* y = alpha * A * x + beta * y   (A is rows x cols, row-major) */
void nn_blas_gemv(const nn_real *A, const nn_real *x, nn_real *y,
                  size_t rows, size_t cols,
                  nn_real alpha, nn_real beta);

/* C = alpha * A * B + beta * C   (A is m x k, B is k x n, all row-major) */
void nn_blas_gemm(const nn_real *A, const nn_real *B, nn_real *C,
                  size_t m, size_t k, size_t n,
                  nn_real alpha, nn_real beta);

#endif
