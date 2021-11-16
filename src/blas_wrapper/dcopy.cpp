#ifdef HAVE_CUDA
#include "cublas_v2.h"
#endif

#include "cblas.h"
#include "blas_wrapper.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::dcopy(size_t N,
    const vector_type& X, size_t incx,
    vector_type& Y, size_t incy) {
  this->dcopy(N, 
      X, 0, incx,
      Y, 0, incy);
}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::dcopy(size_t N, 
    const vector_type& X, size_t offset_x, size_t incx,
    vector_type& Y, size_t offset_y, size_t incy) { 
  cblas_dcopy(N,
      X.data() + offset_x, incx,
      Y.data() + offset_y, incy);
};

#ifdef HAVE_CUDA
template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dcopy(size_t N, 
    const vector_type& X, size_t offset_x, size_t incx,
    vector_type& Y, size_t offset_y, size_t incy) { 
  cublasDcopy(handle, N,
      X.data().get() + offset_x, incx,
      Y.data().get() + offset_y, incy);
};
#endif

