#include <iostream>

#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include "cublas_v2.h"
#endif

#include "blas_wrapper.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::dgeam(bool TransA, bool TransB,
    size_t m, size_t n,
    double alpha,
    const vector_type& A, size_t lda,
    double beta,
    const vector_type& B, size_t ldb,
    vector_type& C, size_t ldc) {
  this->dgeam(
      TransA, TransB,
      m, n,
      alpha, 
      A, 0, lda,
      beta,
      B, 0, ldb,
      C, 0, ldc);
}

template <>
void Blas_Wrapper<std::vector, std::allocator>::dgeam(bool TransA, bool TransB,
    size_t m, size_t n,
    double alpha,
    const vector_type& A, size_t offset_a, size_t lda,
    double beta,
    const vector_type& B, size_t offset_b, size_t ldb,
    vector_type& C, size_t offset_c, size_t ldc) {
  const double* a_ptr = A.data() + offset_a;
  const double* b_ptr = B.data() + offset_b;
  double* c_ptr = C.data() + offset_c;
  if (TransA == false && TransB == false) {
    if (a_ptr == c_ptr) {
      for (size_t col = 0; col < n; col++) {
        for (size_t row = 0; row < m; row++) {
          c_ptr[col * ldc + row] = alpha * c_ptr[col * ldc + row] + beta * b_ptr[col * ldb + row];
        }
      }
    } else if (b_ptr == c_ptr) {
      for (size_t col = 0; col < n; col++) {
        for (size_t row = 0; row < m; row++) {
          c_ptr[col * ldc + row] = alpha * a_ptr[col * lda + row] + beta * c_ptr[col * ldb + row];
        }
      }
    } else {
      for (size_t col = 0; col < n; col++) {
        for (size_t row = 0; row < m; row++) {
          c_ptr[col * ldc + row] = alpha * a_ptr[col * lda + row] + beta * b_ptr[col * ldb + row];
        }
      }
    }
  } else {
    std::cerr << "vector<double> implation of blas_wrapper.dgeam is unable to transpose matrices\n";
    exit(0);
  }
}

#ifdef HAVE_CUDA
template <>
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dgeam(bool TransA, bool TransB,
    size_t m, size_t n,
    double alpha,
    const vector_type& A, size_t offset_a, size_t lda,
    double beta,
    const vector_type& B, size_t offset_b, size_t ldb,
    vector_type& C, size_t offset_c, size_t ldc) {
  auto TA = (TransA ? CUBLAS_OP_T : CUBLAS_OP_N);
  auto TB = (TransB ? CUBLAS_OP_T : CUBLAS_OP_N);
  auto status = cublasDgeam(handle,
      TA, TB,
      m, n,
      &alpha,
      A.data().get() + offset_a, lda,
      &beta,
      B.data().get() + offset_b, ldb,
      C.data().get() + offset_c, ldc);
}
#endif
