#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include "cublas_v2.h"
#endif

#include "cblas.h"
#include "blas_wrapper.h"
#include "blas_calls.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::dherk(BLAS_WRAPPER::Fill_Mode_t fill_mode, bool Trans, 
      size_t m, size_t k, 
      double alpha,
      const vector_type& A, size_t lda,
      double beta,
      vector_type& B, size_t ldb) {
  this->dherk(
      fill_mode, Trans,
      m, k,
      alpha,
      A, 0, lda,
      beta,
      B, 0, ldb);
}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::dherk(BLAS_WRAPPER::Fill_Mode_t fill_mode, bool Trans, 
      size_t m, size_t k, 
      double alpha,
      const vector_type& A, size_t offset_a, size_t lda,
      double beta,
      vector_type& B, size_t offset_b, size_t ldb) {
  this->dsyrk(
      fill_mode, Trans,
      m, k,
      alpha,
      A, offset_a, lda,
      beta,
      B, offset_b, ldb);
}

#ifdef HAVE_CUDA
template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dherk(BLAS_WRAPPER::Fill_Mode_t fill_mode, bool Trans, 
      size_t m, size_t k, 
      double alpha,
      const vector_type& A, size_t offset_a, size_t lda,
      double beta,
      vector_type& B, size_t offset_b, size_t ldb) {
  this->dsyrk(
      fill_mode, Trans,
      m, k,
      alpha,
      A, offset_a, lda,
      beta,
      B, offset_b, ldb);
}
#endif
