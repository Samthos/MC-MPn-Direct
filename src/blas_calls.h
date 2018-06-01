// Copyright 2017

#ifndef BLAS_CALLS_H_
#define BLAS_CALLS_H_

enum CBLAS_ORDER {CblasRowMajor = 101, CblasColMajor = 102};
enum CBLAS_TRANSPOSE {CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113};
enum DDGMM_SIDE {DDGMM_SIDE_LEFT = 101, DDGMM_SIDE_RIGHT = 102};

// from gls-2.1 cblas
void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
             const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
             const int K, const double alpha, const double *A, const int lda,
             const double *B, const int ldb, const double beta, double *C,
             const int ldc);

// from gls-2.1 cblas
void cblas_dger(const enum CBLAS_ORDER order, const int M, const int N,
            const double alpha, const double *X, const int incX,
            const double *Y, const int incY, double *A, const int lda);

// from gls-2.1 cblas
void cblas_dgemv(const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
             const int M, const int N, const double alpha, const double *A,
             const int lda, const double *X, const int incX,
             const double beta, double *Y, const int incY);

// from gls-2.1 cblas
double cblas_ddot(const int N, const double *X, const int incX, const double *Y,
            const int incY);

void Ddgmm(const enum DDGMM_SIDE Side,
    int m, int n,
    const double *A, int lda,
    const double *x, int incx,
    double *C, int ldc);

#endif  // BLAS_CALLS_H_
