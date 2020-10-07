#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include "cublas_v2.h"
#endif

#include "cblas.h"
#include "blas_wrapper.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::batched_ddot(size_t N, size_t K,
    const vector_type& A, size_t offset_a, size_t lda,
    const vector_type& B, size_t offset_b, size_t ldb,
    vector_type& X, size_t incx) {
  vector_type C(N * K, 0.0);
  vector_type o(K, 1.0);
  this->dgekm(
      false, false,
      K, N,
      1.0,
      A, offset_a, lda,
      B, offset_b, ldb,
      0.0,
      C, 0, K);
  this->dgemv(
      true,
      K, N,
      1.0, 
      C, 0, K,
      o, 0, 1,
      0.0,
      X, 0, incx);
}
