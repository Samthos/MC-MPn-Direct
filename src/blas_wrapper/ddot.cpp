#ifdef HAVE_CUDA
#include "cublas_v2.h"
#endif

#include "cblas.h"
#include "blas_wrapper.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
double Blas_Wrapper<Container, Allocator>::ddot(size_t N, 
    const vector_type& X, size_t incx,
    const vector_type& Y, size_t incy) {
  return this->ddot(N,
      X, 0, incx,
      Y, 0, incy);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::ddot(size_t N, 
    const vector_type& X, size_t incx,
    const vector_type& Y, size_t incy, 
    double* result) {
  this->ddot(N,
      X, 0, incx,
      Y, 0, incy,
      result);
}

template <> 
double Blas_Wrapper<std::vector, std::allocator>::ddot(size_t N, 
    const vector_type& X, size_t offset_x, size_t incx,
    const vector_type& Y, size_t offset_y, size_t incy) { 
  double result;
  result = cblas_ddot(N,
      X.data() + offset_x, incx,
      Y.data() + offset_y, incy);
  return result;
};

template <> 
void Blas_Wrapper<std::vector, std::allocator>::ddot(size_t N, 
    const vector_type& X, size_t offset_x, size_t incx,
    const vector_type& Y, size_t offset_y, size_t incy,
    double* result) { 
  *result = cblas_ddot(N,
      X.data() + offset_x, incx,
      Y.data() + offset_y, incy);
};

#ifdef HAVE_CUDA
template <> 
double Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::ddot(size_t N, 
    const vector_type& X, size_t offset_x, size_t incx,
    const vector_type& Y, size_t offset_y, size_t incy) { 
  double result;
  cublasDdot(handle, N,
      X.data().get() + offset_x, incx,
      Y.data().get() + offset_y, incy,
      &result);
  return result;
};

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::ddot(size_t N, 
    const vector_type& X, size_t offset_x, size_t incx,
    const vector_type& Y, size_t offset_y, size_t incy,
    double* result) { 
  cublasDdot(handle, N,
      X.data().get() + offset_x, incx,
      Y.data().get() + offset_y, incy,
      result);
};
#endif

