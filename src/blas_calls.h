// Copyright 2017

#include "cblas.h"

#ifndef BLAS_CALLS_H_
#define BLAS_CALLS_H_

enum DDGMM_SIDE { DDGMM_SIDE_LEFT = 101,
  DDGMM_SIDE_RIGHT = 102 };

void Ddgmm(const enum DDGMM_SIDE Side,
           int m, int n,
           const double *A, int lda,
           const double *x, int incx,
           double *C, int ldc);

void Transpose(const double*A, int m, double *B);

#endif  // BLAS_CALLS_H_
