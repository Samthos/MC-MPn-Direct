#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#endif

#include <numeric>
#include "blas_wrapper.h"

template <> 
double Blas_Wrapper<std::vector, std::allocator>::accumulate(
    iterator first1, iterator last1, double value) {
  return std::accumulate(first1, last1, value);
}

#ifdef HAVE_CUDA
template <> 
double Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::accumulate(
    iterator first1, iterator last1, double value) {
//  return thrust::reduce(first1, last1, value);
  return 0.0;
}
#endif

