#ifdef HAVE_CUDA
#include "cublas_v2.h"
#endif

#include "blas_wrapper.h"
#include "blas_calls.h"

template <>
void Blas_Wrapper<std::vector, std::allocator>::transpose(
    size_t m, size_t n,
    const vector_type& A, size_t lda,
    vector_type& B, size_t ldb) {
  const double* a_ptr = A.data();
  double* b_ptr = B.data();
  if (m == n) {
    Transpose(a_ptr, m, b_ptr);
  } else {
    std::cerr << "In blas_wrapper.transpose: CPU transpose only for dense square matrix\n";
    exit(0);
  }
}

#ifdef HAVE_CUDA
template <>
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::transpose(
    size_t m, size_t n,
    const vector_type& A, size_t lda,
    vector_type& B, size_t ldb) {
  double alpha = 1.0;
  double beta = 0.0;
  this->dgeam(true, false,
      m, n,
      alpha,
      A, 0, lda,
      beta,
      A, 0, lda,
      B, 0, ldb);
}
#endif
