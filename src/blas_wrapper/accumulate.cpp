#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#endif

#include <numeric>
#include "blas_wrapper.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
double Blas_Wrapper<Container, Allocator>::accumulate(
    size_t N,
    const vector_type& X, size_t incx) {
  return this->ddot(N, X, 0, incx, one, 0, 0);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
double Blas_Wrapper<Container, Allocator>::accumulate(
    size_t N,
    const vector_type& X, size_t offset_x, size_t incx) {
  return this->ddot(N,
      X, 0, incx,
      one, 0, 0);
}
