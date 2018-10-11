#include <iostream>

#include "blas_calls.h"

#ifndef OFFSET
#define OFFSET(N, incX) ((incX) > 0 ? 0 : ((N)-1) * (-(incX)))
#endif

typedef int INDEX;


void my_cblas_dgemm_sym(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                        const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                        const int K, const double alpha, const double *A, const int lda,
                        const double *B, const int ldb, const double beta, double *C,
                        const int ldc) {
  {
    INDEX i, j, k;
    INDEX n1, n2;
    INDEX ldf, ldg;
    int TransF, TransG;
    const double *F, *G;
    if (alpha == 0.0 && beta == 1.0)
      return;
    if (Order == CblasRowMajor) {
      n1 = M;
      n2 = N;
      F = A;
      ldf = lda;
      TransF = (TransA == CblasConjTrans) ? CblasTrans : TransA;
      G = B;
      ldg = ldb;
      TransG = (TransB == CblasConjTrans) ? CblasTrans : TransB;
    } else {
      n1 = N;
      n2 = M;
      F = B;
      ldf = ldb;
      TransF = (TransB == CblasConjTrans) ? CblasTrans : TransB;
      G = A;
      ldg = lda;
      TransG = (TransA == CblasConjTrans) ? CblasTrans : TransA;
    }
    if (beta == 0.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          C[ldc * i + j] = 0.0;
        }
      }
    } else if (beta != 1.0) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          C[ldc * i + j] *= beta;
        }
      }
    }
    if (alpha == 0.0)
      return;
    if (TransF == CblasNoTrans && TransG == CblasNoTrans) {
      for (k = 0; k < K; k++) {
        for (i = 0; i < n1; i++) {
          const double temp = alpha * F[ldf * i + k];
          if (temp != 0.0) {
            for (j = 0; j < n2; j++) {
              C[ldc * i + j] += temp * G[ldg * k + j];
            }
          }
        }
      }
    } else if (TransF == CblasNoTrans && TransG == CblasTrans) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          double temp = 0.0;
          for (k = 0; k < K; k++) {
            temp += F[ldf * i + k] * G[ldg * j + k];
          }
          C[ldc * i + j] += alpha * temp;
        }
      }
    } else if (TransF == CblasTrans && TransG == CblasNoTrans) {
      for (k = 0; k < K; k++) {
        for (i = 0; i < n1; i++) {
          const double temp = alpha * F[ldf * k + i];
          if (temp != 0.0) {
            for (j = i; j < n2; j++) {
              C[ldc * i + j] += temp * G[ldg * k + j];
            }
          }
        }
      }
      for (i = 0; i < n1; i++) {
        for (j = i + 1; j < n2; j++) {
          C[ldc * j + i] += C[ldc *i + j];
        }
      }
    } else if (TransF == CblasTrans && TransG == CblasTrans) {
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n2; j++) {
          double temp = 0.0;
          for (k = 0; k < K; k++) {
            temp += F[ldf * k + i] * G[ldg * j + k];
          }
          C[ldc * i + j] += alpha * temp;
        }
      }
    } else {
      std::cerr << "unrecognized operation in dgemm call" << std::endl;
      exit(0);
    }
  }
}
void Ddgmm(const enum DDGMM_SIDE Side,
           int m, int n,
           const double *A, int lda,
           const double *x, int incx,
           double *C, int ldc) {
  if (Side == DDGMM_SIDE_LEFT) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        C[i * ldc + j] = x[j * incx] * A[i * lda + j];
      }
    }
  } else if (Side == DDGMM_SIDE_RIGHT) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        C[i * ldc + j] = x[i * incx] * A[i * lda + j];
      }
    }
  }
}

void Transpose(const double*A, int m, double *B) {
  for (int row = 0; row < m; row++) {
    for (int col = 0; col < m; col++) {
      B[col * m + row] = A[row * m + col];
    }
  }
}
