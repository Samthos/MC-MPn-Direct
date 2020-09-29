#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include "cublas_v2.h"
#endif

#include "cblas.h"
#include "blas_wrapper.h"
#include "blas_calls.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::dsyrk(BLAS_WRAPPER::Fill_Mode_t fill_mode, bool Trans, 
      size_t m, size_t k, 
      double alpha,
      const vector_type& A, size_t lda,
      double beta,
      vector_type& B, size_t ldb) {
  this->dsyrk(
      fill_mode, Trans,
      m, k,
      alpha,
      A, 0, lda,
      beta,
      B, 0, ldb);
}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::dsyrk(BLAS_WRAPPER::Fill_Mode_t fill_mode, bool Trans, 
      size_t m, size_t k, 
      double alpha,
      const vector_type& A, size_t offset_a, size_t lda,
      double beta,
      vector_type& B, size_t offset_b, size_t ldb) {
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

#ifdef HAVE_CUDA
template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dsyrk(BLAS_WRAPPER::Fill_Mode_t fill_mode, bool Trans, 
      size_t m, size_t k, 
      double alpha,
      const vector_type& A, size_t offset_a, size_t lda,
      double beta,
      vector_type& B, size_t offset_b, size_t ldb) {
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
#endif
