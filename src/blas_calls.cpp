#include <functional>
#include "blas_calls.h"

void Ddgmm(const enum DDGMM_SIDE Side,
    int m, int n,
    const double *A, int lda,
    const double *x, int incx,
    double *C, int ldc) {
  Ddgmm(Side,
      m, n,
      A, lda,
      x, incx,
      C, ldc, std::multiplies<>());
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
