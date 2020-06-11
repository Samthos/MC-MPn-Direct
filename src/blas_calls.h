#ifndef BLAS_CALLS_H_
#define BLAS_CALLS_H_

enum DDGMM_SIDE { DDGMM_SIDE_LEFT = 101,
  DDGMM_SIDE_RIGHT = 102 };

template <class BinaryOperator>
void Ddgmm(const enum DDGMM_SIDE Side,
    int m, int n,
    const double *A, int lda,
    const double *x, int incx,
    double *C, int ldc, BinaryOperator op) {
  if (Side == DDGMM_SIDE_LEFT) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        C[i * ldc + j] = op(A[i * lda + j], x[j * incx]);
      }
    }
  } else if (Side == DDGMM_SIDE_RIGHT) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        C[i * ldc + j] = op(A[i * lda + j], x[i * incx]);
      }
    }
  }
}
void Ddgmm(const enum DDGMM_SIDE Side,
    int m, int n,
    const double *A, int lda,
    const double *x, int incx,
    double *C, int ldc);
void Transpose(const double*A, int m, double *B);
void set_Upper_from_Lower(double *A, int m);
void dspr(int mode, int uplo, int n,
          double alpha,
          const double* x, int incx,
          double* ap);

void dspr_batched(int rows, int cols,
          double alpha,
          const double* x,
          double* ap);
#endif  // BLAS_CALLS_H_
