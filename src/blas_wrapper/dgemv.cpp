#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include "cublas_v2.h"
#endif

#include "cblas.h"
#include "blas_wrapper.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::dgemv(bool Trans, 
    size_t m, size_t n,
    double alpha,
    const vector_type& A, size_t lda,
    const vector_type& x, size_t incx,
    double beta,
    vector_type& y, size_t incy) {
  this->dgemv(
      Trans,
      m, n,
      alpha,
      A, 0, lda,
      x, 0, incx,
      beta,
      y, 0, incy);
}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::dgemv(bool Trans, 
    size_t m, size_t n,
    double alpha,
    const vector_type& A, size_t offset_a, size_t lda,
    const vector_type& x, size_t offset_x, size_t incx,
    double beta,
    vector_type& y, size_t offset_y, size_t incy) {
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

#ifdef HAVE_CUDA
template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dgemv(bool Trans, 
    size_t m, size_t n,
    double alpha,
    const vector_type& A, size_t offset_a, size_t lda,
    const vector_type& x, size_t offset_x, size_t incx,
    double beta,
    vector_type& y, size_t offset_y, size_t incy) {
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
#endif
