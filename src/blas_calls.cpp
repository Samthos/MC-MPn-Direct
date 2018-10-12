#include <iostream>

#include "blas_calls.h"

#ifndef OFFSET
#define OFFSET(N, incX) ((incX) > 0 ? 0 : ((N)-1) * (-(incX)))
#endif

typedef int INDEX;

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

void set_Upper_from_Lower(double *A, int m) {
  for (int col = 0; col < m; ++col) {
    for (int row = col+1; row < m; ++row) {
      A[row * m + col] = A[col * m + row];
    }
  }
}
