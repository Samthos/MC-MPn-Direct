#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include "cublas_v2.h"
#endif

#include "cblas.h"
#include "blas_wrapper.h"

template <>
void Blas_Wrapper<std::vector, std::allocator>::batched_ddot(size_t N, size_t K,
    const vector_type& A, size_t offset_a, size_t lda,
    const vector_type& B, size_t offset_b, size_t ldb,
    vector_type& X, size_t incx) {
  for (size_t i = 0; i < N; ++i) {
    this->ddot(K, 
        A, offset_a + lda * i, 1,
        B, offset_b + ldb * i, 1,
        X.data() + i * incx);
  }
}

#ifdef HAVE_CUDA
template <>
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::batched_ddot(size_t N, size_t K,
    const vector_type& A, size_t offset_a, size_t lda,
    const vector_type& B, size_t offset_b, size_t ldb,
    vector_type& X, size_t incx) {
  for (size_t i = 0; i < N; ++i) {
    this->ddot(K, 
        A, offset_a + lda * i, 1,
        B, offset_b + ldb * i, 1,
        X.data().get() + i * incx);
  }
}
#endif
