/* nn_blas.c — portable BLAS wrappers for dense layer operations.
 *
 * Dispatch to Apple Accelerate, OpenBLAS/CBLAS, or scalar fallback
 * depending on build-time NN_BLAS setting (see Makefile).
 *
 * This file is safe to compile standalone; it only needs <cblas.h> or
 * <Accelerate/Accelerate.h> when the corresponding mode is active.
 */

#include "nn_core.h"

#if defined(NN_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(NN_USE_CBLAS)
#include <cblas.h>
#endif

void nn_blas_gemv(const nn_real *A, const nn_real *x, nn_real *y,
                  size_t rows, size_t cols,
                  nn_real alpha, nn_real beta)
{
#if defined(NN_USE_ACCELERATE) || defined(NN_USE_CBLAS)
#if NN_REAL_IS_FLOAT
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                (int)rows, (int)cols,
                (float)alpha, (const float *)A, (int)cols,
                (const float *)x, 1, (float)beta, (float *)y, 1);
#else
    cblas_dgemv(CblasRowMajor, CblasNoTrans,
                (int)rows, (int)cols,
                (double)alpha, (const double *)A, (int)cols,
                (const double *)x, 1, (double)beta, (double *)y, 1);
#endif
#else
    /* Scalar fallback: y = alpha * A * x + beta * y */
    size_t r;
    for (r = 0; r < rows; ++r) {
        nn_real dot = (nn_real)0;
        size_t c;
        for (c = 0; c < cols; ++c)
            dot += A[r * cols + c] * x[c];
        y[r] = alpha * dot + beta * y[r];
    }
#endif
}

void nn_blas_gemm(const nn_real *A, const nn_real *B, nn_real *C,
                  size_t m, size_t k, size_t n,
                  nn_real alpha, nn_real beta)
{
    /* C(m x n) = alpha * A(m x k) * B(k x n) + beta * C.
     * For dense_forward_layer_batch: C = X * W^T where X is batch x prev,
     * W is current x prev, so A=X(m=batch, k=prev), B=W^T (k=prev, n=current).
     * Storage: A row-major, B row-major needs CblasTrans to treat it as W^T. */
#if defined(NN_USE_ACCELERATE) || defined(NN_USE_CBLAS)
#if NN_REAL_IS_FLOAT
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)m, (int)n, (int)k,
                (float)alpha, (const float *)A, (int)k,
                (const float *)B, (int)k,
                (float)beta, (float *)C, (int)n);
#else
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                (int)m, (int)n, (int)k,
                (double)alpha, (const double *)A, (int)k,
                (const double *)B, (int)k,
                (double)beta, (double *)C, (int)n);
#endif
#else
    /* Scalar fallback: C = alpha * A * B + beta * C.
     * B is row-major with shape k x n; B[j * k + p] indexes row j, column p. */
    size_t i, j, p;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            nn_real s = (nn_real)0;
            for (p = 0; p < k; ++p)
                s += A[i * k + p] * B[j * k + p];
            C[i * n + j] = alpha * s + beta * C[i * n + j];
        }
    }
#endif
}
