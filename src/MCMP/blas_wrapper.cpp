#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include "cublas_v2.h"
#endif

#include <algorithm>
#include <functional>


#include "cblas.h"
#include "../blas_calls.h"

#include "blas_wrapper.h"

template <> 
Blas_Wrapper<std::vector, std::allocator>::Blas_Wrapper() {}

template <> 
Blas_Wrapper<std::vector, std::allocator>::~Blas_Wrapper() {}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::dgemm(bool TransA, bool TransB, 
      size_t m, size_t n, size_t k, 
      double alpha,
      const vector_double& A, size_t lda,
      const vector_double& B, size_t ldb,
      double beta,
      vector_double& C, size_t ldc) {
  auto TA = (TransA ? CblasTrans : CblasNoTrans);
  auto TB = (TransB ? CblasTrans : CblasNoTrans);
  cblas_dgemm(CblasColMajor,
      TA, TB,
      m, n, k,
      alpha,
      A.data(), lda,
      B.data(), ldb,
      beta,
      C.data(), ldc);
}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::dgemv(bool Trans, 
    size_t m, size_t n,
    double alpha,
    const vector_double& A, size_t lda,
    const vector_double& x, size_t incx,
    double beta,
    vector_double& y, size_t incy) {
  auto T = (Trans ? CblasTrans : CblasNoTrans);
  cblas_dgemv(CblasColMajor,
      T,
      m, n,
      alpha,
      A.data(), lda,
      x.data(), incx,
      beta,
      y.data(), incy);
}

template <>
void Blas_Wrapper<std::vector, std::allocator>::Ddgmm(bool right_side,
      size_t m, size_t n,
      const vector_double& A, size_t lda,
      const vector_double& x, size_t incx,
      vector_double& B, size_t ldb) {
  auto side = (right_side ? DDGMM_SIDE_RIGHT : DDGMM_SIDE_LEFT);
// Ddgmm(side,
//    m, n,
//    A.data(), lda,
//    x.data(), incx,
//    B.data(), ldb);
}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::ddot(size_t N, 
    const vector_double& X, size_t incx,
    const vector_double& Y, size_t incy, 
    double *result) {
  *result = cblas_ddot(N,
      X.data(), incx,
      Y.data(), incy);
};

template <> 
void Blas_Wrapper<std::vector, std::allocator>::transform_multiplies(const vector_double& A, 
      const vector_double& B, 
      vector_double& C) {
  std::transform(A.begin(), A.end(), B.begin(), C.begin(), std::multiplies<>());
}


#ifdef HAVE_CUDA
/*
template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dgemm(bool TransA, bool TransB, 
      size_t m, size_t n, size_t k, 
      double alpha,
      const vector_double& A, size_t lda,
      const vector_double& B, size_t ldb,
      double beta,
      vector_double& C, size_t ldc) {
  auto TA = (TransA ? CblasTrans : CblasNoTrans);
  auto TB = (TransB ? CblasTrans : CblasNoTrans);
  cblas_dgemm(CblasColMajor,
      TA, TB,
      m, n, k,
      alpha,
      A.data(), lda,
      B.data(), ldb,
      beta,
      C.data(), ldc);
}

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dgemv(bool Trans, 
    size_t m, size_t n,
    double alpha,
    const vector_double& A, size_t lda,
    const vector_double& x, size_t incx,
    double beta,
    vector_double& y, size_t incy) {
  auto T = (Trans ? CblasTrans : CblasNoTrans);
  cblas_dgemv(CblasColMajor,
      T,
      m, n,
      alpha,
      A.data(), lda,
      x.data(), incx,
      beta,
      y.data(), incy);
}

template <> void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::Ddgmm(bool right_side,
      size_t m, size_t n,
      const vector_double& A, size_t lda,
      const vector_double& x, size_t incx,
      vector_double& B, size_t ldb) {
  auto side = (right_side ? DDGMM_SIDE_RIGHT : DDGMM_SIDE_LEFT);
  Ddgmm(side,
      m, n,
      A.data(), lda,
      x.data(), incx,
      B.data(), ldb);
}

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::ddot(size_t N, 
    const vector_double& X, size_t incx,
    const vector_double& Y, size_t incy, 
    double *result) {
// cublas_Ddot(N,
//     X.data(), incx,
//     Y.data(), incy,
//     result);
};

template <> void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::transform_multiplies(const vector_double& A, 
      const vector_double& B, 
      vector_double& C) {
  thrust::transform(A.begin(), A.end(), B.begin(), C.begin(), thrust::multiplies<double>());
}
*/
#endif

