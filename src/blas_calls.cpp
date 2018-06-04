#include <iostream>

#include "blas_calls.h"

#ifndef OFFSET
#define OFFSET(N, incX) ((incX) > 0 ? 0 : ((N)-1) * (-(incX)))
#endif

typedef int INDEX;

void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
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
            for (j = 0; j < n2; j++) {
              C[ldc * i + j] += temp * G[ldg * k + j];
            }
          }
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

void cblas_dger(const enum CBLAS_ORDER order, const int M, const int N,
                const double alpha, const double *X, const int incX,
                const double *Y, const int incY, double *A, const int lda) {
  {
    INDEX i, j;

    if (order == CblasRowMajor) {
      INDEX ix = OFFSET(M, incX);
      for (i = 0; i < M; i++) {
        const double tmp = alpha * X[ix];
        INDEX jy = OFFSET(N, incY);
        for (j = 0; j < N; j++) {
          A[lda * i + j] += Y[jy] * tmp;
          jy += incY;
        }
        ix += incX;
      }
    } else if (order == CblasColMajor) {
      INDEX jy = OFFSET(N, incY);
      for (j = 0; j < N; j++) {
        const double tmp = alpha * Y[jy];
        INDEX ix = OFFSET(M, incX);
        for (i = 0; i < M; i++) {
          A[i + lda * j] += X[ix] * tmp;
          ix += incX;
        }
        jy += incY;
      }
    } else {
      std::cerr << "unrecognized operation in dgemm call" << std::endl;
      exit(0);
    }
  }
}

double
cblas_ddot(const int N, const double *X, const int incX, const double *Y,
           const int incY) {
  {
    double r = 0.0;
    INDEX i;
    INDEX ix = OFFSET(N, incX);
    INDEX iy = OFFSET(N, incY);
    for (i = 0; i < N; i++) {
      r += X[ix] * Y[iy];
      ix += incX;
      iy += incY;
    }
    return r;
  }
}

void cblas_dgemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
                 const int M, const int N, const double alpha, const double *A,
                 const int lda, const double *X, const int incX,
                 const double beta, double *Y, const int incY) {
  {
    INDEX i, j;
    INDEX lenX, lenY;
    const int Trans = (TransA != CblasConjTrans) ? TransA : CblasTrans;
    if (M == 0 || N == 0)
      return;
    if (alpha == 0.0 && beta == 1.0)
      return;
    if (Trans == CblasNoTrans) {
      lenX = N;
      lenY = M;
    } else {
      lenX = M;
      lenY = N;
    }
    if (beta == 0.0) {
      INDEX iy = OFFSET(lenY, incY);
      for (i = 0; i < lenY; i++) {
        Y[iy] = 0.0;
        iy += incY;
      }
    } else if (beta != 1.0) {
      INDEX iy = OFFSET(lenY, incY);
      for (i = 0; i < lenY; i++) {
        Y[iy] *= beta;
        iy += incY;
      }
    }
    if (alpha == 0.0)
      return;
    if ((order == CblasRowMajor && Trans == CblasNoTrans) || (order == CblasColMajor && Trans == CblasTrans)) {
      INDEX iy = OFFSET(lenY, incY);
      for (i = 0; i < lenY; i++) {
        double temp = 0.0;
        INDEX ix = OFFSET(lenX, incX);
        for (j = 0; j < lenX; j++) {
          temp += X[ix] * A[lda * i + j];
          ix += incX;
        }
        Y[iy] += alpha * temp;
        iy += incY;
      }
    } else if ((order == CblasRowMajor && Trans == CblasTrans) || (order == CblasColMajor && Trans == CblasNoTrans)) {
      INDEX ix = OFFSET(lenX, incX);
      for (j = 0; j < lenX; j++) {
        const double temp = alpha * X[ix];
        if (temp != 0.0) {
          INDEX iy = OFFSET(lenY, incY);
          for (i = 0; i < lenY; i++) {
            Y[iy] += temp * A[lda * j + i];
            iy += incY;
          }
        }
        ix += incX;
      }
    } else {
      std::cerr << "unrecognized operation" << std::endl;
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
