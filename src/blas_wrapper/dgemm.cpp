#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include "cublas_v2.h"
#endif

#include "cblas.h"
#include "blas_wrapper.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::dgemm(bool TransA, bool TransB, 
      size_t m, size_t n, size_t k, 
      double alpha,
      const vector_type& A, size_t lda,
      const vector_type& B, size_t ldb,
      double beta,
      vector_type& C, size_t ldc) {
  this->dgemm(
      TransA, TransB,
      m, n, k,
      alpha,
      A, 0, lda,
      B, 0, ldb,
      beta,
      C, 0, ldc);
}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::dgemm(bool TransA, bool TransB, 
      size_t m, size_t n, size_t k, 
      double alpha,
      const vector_type& A, size_t offset_a, size_t lda,
      const vector_type& B, size_t offset_b, size_t ldb,
      double beta,
      vector_type& C, size_t offset_c, size_t ldc) {
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

#ifdef HAVE_CUDA
template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dgemm(bool TransA, bool TransB, 
      size_t m, size_t n, size_t k, 
      double alpha,
      const vector_type& A, size_t offset_a, size_t lda,
      const vector_type& B, size_t offset_b, size_t ldb,
      double beta,
      vector_type& C, size_t offset_c, size_t ldc) {
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
#endif
