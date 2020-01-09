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

void dspr_batched(
    int rows, int cols,
    double alpha,
    const double* x,
    double* ap) {

  for (auto c1 = 0; c1 < cols; c1++) {
    for (auto c2 = 0; c2 <= c1; c2++) {
      for (auto row = 0; row < rows; row++) {
        ap[(c1 * (c1 + 1) / 2 + c2) * rows + row] += alpha * x[c1 * rows + row] * x[c2 * rows + row];
      }
    }
  }
}

void dspr(int mode, int uplo, int n,
    double alpha,
    const double* x, int incx,
    double* ap){
  for (auto col = 0; col < n; col++) {
    for (auto row = 0; row <= col; row++) {
      ap[row + col * (col+1) / 2] += alpha * x[row * incx] * x[col * incx];
    }
  }
}


