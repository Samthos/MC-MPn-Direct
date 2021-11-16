#ifdef HAVE_CUDA
#include "cublas_v2.h"
#endif

#include "cblas.h"
#include "blas_wrapper.h"
#include "blas_calls.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::ddgmm(BLAS_WRAPPER::Side_t side,
    size_t m, size_t n,
    const vector_type& A, size_t lda,
    const vector_type& x, size_t incx,
    vector_type& B, size_t ldb) {
  this->ddgmm(side,
     m, n,
     A, 0, lda,
     x, 0, incx,
     B, 0, ldb);
}

template <>
void Blas_Wrapper<std::vector, std::allocator>::ddgmm(
    BLAS_WRAPPER::Side_t side,
    size_t m, size_t n,
    const vector_type& A, size_t offset_a, size_t lda,
    const vector_type& x, size_t offset_x, size_t incx,
    vector_type& B, size_t offset_b, size_t ldb) {
  DDGMM_SIDE S = (side == BLAS_WRAPPER::RIGHT_SIDE ? DDGMM_SIDE_RIGHT : DDGMM_SIDE_LEFT);
  Ddgmm(S,
     m, n,
     A.data() + offset_a, lda,
     x.data() + offset_x, incx,
     B.data() + offset_b, ldb);
}

#ifdef HAVE_CUDA
template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::ddgmm(BLAS_WRAPPER::Side_t side,
    size_t m, size_t n,
    const vector_type& A, size_t offset_a, size_t lda,
    const vector_type& x, size_t offset_x, size_t incx,
    vector_type& B, size_t offset_b, size_t ldb) {
  auto  S = (side == BLAS_WRAPPER::RIGHT_SIDE ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT);
  cublasDdgmm(handle,
      S,
      m, n,
      A.data().get() + offset_a, lda,
      x.data().get() + offset_x, incx,
      B.data().get() + offset_b, ldb);
}
#endif

