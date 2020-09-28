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

//
// Template Level 3 Blas
//
template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::dgemm(bool TransA, bool TransB, 
      size_t m, size_t n, size_t k, 
      double alpha,
      const vector_double& A, size_t lda,
      const vector_double& B, size_t ldb,
      double beta,
      vector_double& C, size_t ldc) {
  this->dgemm(
      TransA, TransB,
      m, n, k,
      alpha,
      A, 0, lda,
      B, 0, ldb,
      beta,
      C, 0, ldc);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::dsyrk(BLAS_WRAPPER::Fill_Mode_t fill_mode, bool Trans, 
      size_t m, size_t k, 
      double alpha,
      const vector_double& A, size_t lda,
      double beta,
      vector_double& B, size_t ldb) {
  this->dsyrk(
      fill_mode, Trans,
      m, k,
      alpha,
      A, 0, lda,
      beta,
      B, 0, ldb);
}

//
// Template Level 2 Blas
//
template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::dgemv(bool Trans, 
    size_t m, size_t n,
    double alpha,
    const vector_double& A, size_t lda,
    const vector_double& x, size_t incx,
    double beta,
    vector_double& y, size_t incy) {
  this->dgemv(
      Trans,
      m, n,
      alpha,
      A, 0, lda,
      x, 0, incx,
      beta,
      y, 0, incy);
}

//
// Template Level 1 Blas
//
template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::dcopy(size_t N,
    const vector_double& X, size_t incx,
    vector_double& Y, size_t incy) {
  this->dcopy(N, 
      X, 0, incx,
      Y, 0, incy);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
double Blas_Wrapper<Container, Allocator>::ddot(size_t N, 
    const vector_double& X, size_t incx,
    const vector_double& Y, size_t incy) {
  return this->ddot(N,
      X, 0, incx,
      Y, 0, incy);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::ddot(size_t N, 
    const vector_double& X, size_t incx,
    const vector_double& Y, size_t incy, 
    double* result) {
  this->ddot(N,
      X, 0, incx,
      Y, 0, incy,
      result);
}

//
// CPU Level 3 Blas
//
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
void Blas_Wrapper<std::vector, std::allocator>::dsyrk(BLAS_WRAPPER::Fill_Mode_t fill_mode, bool Trans, 
      size_t m, size_t k, 
      double alpha,
      const vector_double& A, size_t offset_a, size_t lda,
      double beta,
      vector_double& B, size_t offset_b, size_t ldb) {
  auto F = (fill_mode == BLAS_WRAPPER::FILL_UPPER ? CblasUpper : CblasLower);
  auto T = (Trans ? CblasTrans : CblasNoTrans);
  cblas_dsyrk(CblasColMajor,
      F, T,
      m, k,
      alpha,
      A.data() + offset_a, lda,
      beta,
      B.data() + offset_b, ldb);
  if (fill_mode == BLAS_WRAPPER::FILL_FULL) {
    set_Upper_from_Lower(B.data(), ldb);
  }
}

//
// CPU Level 2 Blas
//
template <>
void Blas_Wrapper<std::vector, std::allocator>::batched_ddot(size_t N, size_t K,
    const vector_double& A, size_t offset_a, size_t lda,
    const vector_double& B, size_t offset_b, size_t ldb,
    vector_double& X, size_t incx) {
  for (size_t i = 0; i < N; ++i) {
    this->ddot(K, 
        A, offset_a + lda * i, 1,
        B, offset_b + ldb * i, 1,
        X.data() + i * incx);
  }
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
void Blas_Wrapper<std::vector, std::allocator>::dgemv(bool Trans, 
    size_t m, size_t n,
    double alpha,
    const vector_double& A, size_t offset_a, size_t lda,
    const vector_double& x, size_t offset_x, size_t incx,
    double beta,
    vector_double& y, size_t offset_y, size_t incy) {
  auto T = (Trans ? CblasTrans : CblasNoTrans);
  cblas_dgemv(CblasColMajor,
      T,
      m, n,
      alpha,
      A.data() + offset_a, lda,
      x.data() + offset_x, incx,
      beta,
      y.data() + offset_y, incy);
}

//
// CPU Level 1 Blas
//
template <> 
void Blas_Wrapper<std::vector, std::allocator>::dcopy(size_t N, 
    const vector_double& X, size_t offset_x, size_t incx,
    vector_double& Y, size_t offset_y, size_t incy) { 
  cblas_dcopy(N,
      X.data() + offset_x, incx,
      Y.data() + offset_y, incy);
};

template <> 
double Blas_Wrapper<std::vector, std::allocator>::ddot(size_t N, 
    const vector_double& X, size_t offset_x, size_t incx,
    const vector_double& Y, size_t offset_y, size_t incy) { 
  double result;
  result = cblas_ddot(N,
      X.data() + offset_x, incx,
      Y.data() + offset_y, incy);
  return result;
};

template <> 
void Blas_Wrapper<std::vector, std::allocator>::ddot(size_t N, 
    const vector_double& X, size_t offset_x, size_t incx,
    const vector_double& Y, size_t offset_y, size_t incy,
    double* result) { 
  *result = cblas_ddot(N,
      X.data() + offset_x, incx,
      Y.data() + offset_y, incy);
};

template <> 
void Blas_Wrapper<std::vector, std::allocator>::dscal(size_t N, 
    double alpha,
    vector_double& X, size_t incx) { 
  cblas_dscal(N,
      alpha,
      X.data(), incx);
};

//
// CPU Iterator
//
template <> 
void Blas_Wrapper<std::vector, std::allocator>::fill(
    iterator first1, iterator last1, double value) {
  std::fill(first1, last1, value);
}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::minus(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result) {
  std::transform(first1, last1, first2, result, std::minus<>());
}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::multiplies(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result) {
  std::transform(first1, last1, first2, result, std::multiplies<>());
}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::plus(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result) {
  std::transform(first1, last1, first2, result, std::plus<>());
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

//
// GPU Level 3 Blas
// 
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
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dsyrk(BLAS_WRAPPER::Fill_Mode_t fill_mode, bool Trans, 
      size_t m, size_t k, 
      double alpha,
      const vector_double& A, size_t offset_a, size_t lda,
      double beta,
      vector_double& B, size_t offset_b, size_t ldb) {
  auto F = (fill_mode == BLAS_WRAPPER::FILL_UPPER ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER );
  auto T = (Trans ? CUBLAS_OP_T : CUBLAS_OP_N);
  cublasDsyrk(handle,
      F, T,
      m, k,
      &alpha,
      A.data().get() + offset_a, lda,
      &beta,
      B.data().get() + offset_b, ldb);
  if (fill_mode == BLAS_WRAPPER::FILL_FULL) {
    printf("FILL MODE FULL in dysrk not supported for cuda\n");
    exit(0);
  }
}

//
// GPU Level 2 Blas
// 
template <>
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::batched_ddot(size_t N, size_t K,
    const vector_double& A, size_t offset_a, size_t lda,
    const vector_double& B, size_t offset_b, size_t ldb,
    vector_double& X, size_t incx) {
  for (size_t i = 0; i < N; ++i) {
    this->ddot(K, 
        A, offset_a + lda * i, 1,
        B, offset_b + ldb * i, 1,
        X.data().get() + i * incx);
  }
}

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::ddgmm(bool right_side,
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
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dgemv(bool Trans, 
    size_t m, size_t n,
    double alpha,
    const vector_double& A, size_t offset_a, size_t lda,
    const vector_double& x, size_t offset_x, size_t incx,
    double beta,
    vector_double& y, size_t offset_y, size_t incy) {
  auto T = (Trans ? CUBLAS_OP_T : CUBLAS_OP_N);
  cublasDgemv(handle,
      T,
      m, n,
      &alpha,
      A.data().get() + offset_a, lda,
      x.data().get() + offset_x, incx,
      &beta,
      y.data().get() + offset_y, incy);
}

//
// GPU Level 1 Blas
// 
template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dcopy(size_t N, 
    const vector_double& X, size_t offset_x, size_t incx,
    vector_double& Y, size_t offset_y, size_t incy) { 
  cublasDcopy(handle, N,
      X.data().get() + offset_x, incx,
      Y.data().get() + offset_y, incy);
};

template <> 
double Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::ddot(size_t N, 
    const vector_double& X, size_t offset_x, size_t incx,
    const vector_double& Y, size_t offset_y, size_t incy) { 
  double result;
  cublasDdot(handle, N,
      X.data().get() + offset_x, incx,
      Y.data().get() + offset_y, incy,
      &result);
  return result;
};

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::ddot(size_t N, 
    const vector_double& X, size_t offset_x, size_t incx,
    const vector_double& Y, size_t offset_y, size_t incy,
    double* result) { 
  cublasDdot(handle, N,
      X.data().get() + offset_x, incx,
      Y.data().get() + offset_y, incy,
      result);
};

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dscal(size_t N, 
    double alpha,
    vector_double& X, size_t incx) { 
  cublasDscal(handle,
      N,
      &alpha,
      X.data().get(), incx);
};

//
// GPU Iterator
// 
template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::fill(
    iterator first1, iterator last1, double value) {
  thrust::fill(first1, last1, value);
}

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::minus(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result) {
  thrust::transform(first1, last1, first2, result, thrust::minus<double>());
}

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::multiplies(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result) {
  thrust::transform(first1, last1, first2, result, thrust::multiplies<double>());
}

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::plus(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result) {
  thrust::transform(first1, last1, first2, result, thrust::plus<double>());
}
#endif

