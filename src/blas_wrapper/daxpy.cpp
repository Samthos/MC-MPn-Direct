#ifdef HAVE_CUDA
#include "cublas_v2.h"
#endif

#include "cblas.h"
#include "blas_wrapper.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::daxpy(size_t N,
    double alpha,
    const vector_type& X, size_t incx,
    vector_type& Y, size_t incy) {
  this->daxpy(N, 
      alpha,
      X, 0, incx,
      Y, 0, incy);
}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::daxpy(size_t N, 
    double alpha, 
    const vector_type& X, size_t offset_x, size_t incx,
    vector_type& Y, size_t offset_y, size_t incy) { 
  cblas_daxpy(N,
      alpha,
      X.data() + offset_x, incx,
      Y.data() + offset_y, incy);
};

#ifdef HAVE_CUDA
template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::daxpy(size_t N, 
    double alpha,
    const vector_type& X, size_t offset_x, size_t incx,
    vector_type& Y, size_t offset_y, size_t incy) { 
  cublasDaxpy(handle, N,
      &alpha,
      X.data().get() + offset_x, incx,
      Y.data().get() + offset_y, incy);
};
#endif

