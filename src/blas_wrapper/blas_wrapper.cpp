#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include "cublas_v2.h"
#endif

#include <algorithm>
#include <functional>


#include "cblas.h"
#include "blas_calls.h"

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
void Blas_Wrapper<std::vector, std::allocator>::dgemm(bool TransA, bool TransB, 
      size_t m, size_t n, size_t k, 
      double alpha,
      const vector_double& A, size_t offset_a, size_t lda,
      const vector_double& B, size_t offset_b, size_t ldb,
      double beta,
      vector_double& C, size_t offset_c, size_t ldc) {
  auto TA = (TransA ? CblasTrans : CblasNoTrans);
  auto TB = (TransB ? CblasTrans : CblasNoTrans);
  cblas_dgemm(CblasColMajor,
      TA, TB,
      m, n, k,
      alpha,
      A.data() + offset_a, lda,
      B.data() + offset_b, ldb,
      beta,
      C.data() + offset_c, ldc);
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
void Blas_Wrapper<std::vector, std::allocator>::ddgmm(bool right_side,
    size_t m, size_t n,
    const vector_double& A, size_t lda,
    const vector_double& x, size_t incx,
    vector_double& B, size_t ldb) {
  DDGMM_SIDE side = (right_side ? DDGMM_SIDE_RIGHT : DDGMM_SIDE_LEFT);
  Ddgmm(side,
     m, n,
     A.data(), lda,
     x.data(), incx,
     B.data(), ldb);
}

template <> 
double Blas_Wrapper<std::vector, std::allocator>::ddot(size_t N, 
    const vector_double& X, size_t incx,
    const vector_double& Y, size_t incy) { 
  double result;
  result = cblas_ddot(N,
      X.data(), incx,
      Y.data(), incy);
  return result;
};

template <> 
void Blas_Wrapper<std::vector, std::allocator>::dscal(size_t N, 
    double alpha,
    vector_double& X, size_t incx) { 
  cblas_dscal(N,
      alpha,
      X.data(), incx);
};

template <> 
void Blas_Wrapper<std::vector, std::allocator>::multiplies(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result) {
  std::transform(first1, last1, first2, result, std::multiplies<>());
}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::minus(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result) {
  std::transform(first1, last1, first2, result, std::minus<>());
}


#ifdef HAVE_CUDA
template <> 
Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::Blas_Wrapper() {
  cublasCreate(&handle);
}

template <> 
Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::~Blas_Wrapper() {
  cublasDestroy(handle);
}

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dgemm(bool TransA, bool TransB, 
      size_t m, size_t n, size_t k, 
      double alpha,
      const vector_double& A, size_t lda,
      const vector_double& B, size_t ldb,
      double beta,
      vector_double& C, size_t ldc) {
  auto TA = (TransA ? CUBLAS_OP_T : CUBLAS_OP_N);
  auto TB = (TransB ? CUBLAS_OP_T : CUBLAS_OP_N);
  cublasDgemm(handle,
      TA, TB,
      m, n, k,
      &alpha,
      A.data().get(), lda,
      B.data().get(), ldb,
      &beta,
      C.data().get(), ldc);
}

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dgemm(bool TransA, bool TransB, 
      size_t m, size_t n, size_t k, 
      double alpha,
      const vector_double& A, size_t offset_a, size_t lda,
      const vector_double& B, size_t offset_b, size_t ldb,
      double beta,
      vector_double& C, size_t offset_c, size_t ldc) {
  auto TA = (TransA ? CUBLAS_OP_T : CUBLAS_OP_N);
  auto TB = (TransB ? CUBLAS_OP_T : CUBLAS_OP_N);
  cublasDgemm(handle,
      TA, TB,
      m, n, k,
      &alpha,
      A.data().get() + offset_a, lda,
      B.data().get() + offset_b, ldb,
      &beta,
      C.data().get() + offset_c, ldc);
}

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dgemv(bool Trans, 
    size_t m, size_t n,
    double alpha,
    const vector_double& A, size_t lda,
    const vector_double& x, size_t incx,
    double beta,
    vector_double& y, size_t incy) {
  auto T = (Trans ? CUBLAS_OP_T : CUBLAS_OP_N);
  cublasDgemv(handle,
      T,
      m, n,
      &alpha,
      A.data().get(), lda,
      x.data().get(), incx,
      &beta,
      y.data().get(), incy);
}

template <> void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::ddgmm(bool right_side,
      size_t m, size_t n,
      const vector_double& A, size_t lda,
      const vector_double& x, size_t incx,
      vector_double& B, size_t ldb) {
  auto side = (right_side ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT);
  cublasDdgmm(handle,
      side,
      m, n,
      A.data().get(), lda,
      x.data().get(), incx,
      B.data().get(), ldb);
}

template <> 
double Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::ddot(size_t N, 
    const vector_double& X, size_t incx,
    const vector_double& Y, size_t incy) { 
  double result;
  cublasDdot(handle, N,
      X.data().get(), incx,
      Y.data().get(), incy,
      &result);
  return result;
};

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::multiplies(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result) {
  thrust::transform(first1, last1, first2, result, thrust::multiplies<double>());
}

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::minus(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result) {
  thrust::transform(first1, last1, first2, result, thrust::minus<double>());
}
#endif

