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

void tiled_transpose(const double* A, int lda, double* B) {
  constexpr int tile_size = 8;
  for (int col = 0; col < tile_size; col++) {
    for (int row = 0; row < tile_size; row++) {
      B[row * lda + col] = A[col * lda + row];
    }
  }
}

void tiled_transpose(const double* A, int lda, double* B, int tile_size) {
  for (int col = 0; col < tile_size; col++) {
    for (int row = 0; row < tile_size; row++) {
      B[row * lda + col] = A[col * lda + row];
    }
  }
}

void tiled_transpose_col(const double* A, int lda, double* B, int col_tile_size) {
  constexpr int tile_size = 8;
  for (int col = 0; col < col_tile_size; col++) {
    for (int row = 0; row < tile_size; row++) {
      B[row * lda + col] = A[col * lda + row];
    }
  }
}

void tiled_transpose_row(const double* A, int lda, double* B, int row_tile_size) {
  constexpr int tile_size = 8;
  for (int col = 0; col < tile_size; col++) {
    for (int row = 0; row < row_tile_size; row++) {
      B[row * lda + col] = A[col * lda + row];
    }
  }
}

void Transpose(const double*A, int lda, double *B) {
  constexpr int tile_size = 8;
  int n = lda % 8;
  int m = lda - n;

  for (int col = 0; col < m; col+=tile_size) {
    for (int row = 0; row < m; row+=tile_size) {
      tiled_transpose(A + col * lda + row, lda, B + row * lda + col);
    }
    tiled_transpose_row(A + col * lda + m, lda, B + m * lda + col, n);
  }
  for (int row = 0; row < m; row+=tile_size) {
    tiled_transpose_col(A + m * lda + row, lda, B + row * lda + m, n);
  }
  tiled_transpose(A + m * lda + m, lda, B + m * lda + m, n);
}

void tiled_set_upper_from_lower(double* A, int lda) {
  constexpr int tile_size = 8;
  for (int col = 0; col < tile_size; ++col) {
    for (int row = col+1; row < tile_size; ++row) {
      A[row * lda + col] = A[col * lda + row];
    }
  }
}

void tiled_set_upper_from_lower(double* A, int lda, int tile_size) {
  for (int col = 0; col < tile_size; ++col) {
    for (int row = col+1; row < tile_size; ++row) {
      A[row * lda + col] = A[col * lda + row];
    }
  }
}

void set_Upper_from_Lower(double *A, int lda) {
  constexpr int tile_size = 8;
  int n = lda % 8;
  int m = lda - n;

  for (int col = 0; col < m; col += tile_size) {
    tiled_set_upper_from_lower(A + col * lda + col, lda);
    for (int row = col + tile_size; row < m; row += tile_size) {
      tiled_transpose(A + col * lda + row, lda, A + row * lda + col);
    }
    tiled_transpose_row(A + col * lda + m, lda, A + m * lda + col, n);
  }
    tiled_set_upper_from_lower(A + m * lda + m, lda, n);
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


