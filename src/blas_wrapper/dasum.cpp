#ifdef HAVE_CUDA
#include "cublas_v2.h"
#endif

#include "cblas.h"
#include "blas_wrapper.h"

template <> 
double Blas_Wrapper<std::vector, std::allocator>::dasum(size_t N, 
    const vector_type& X, size_t incx) { 
  return cblas_dasum(N, X.data(), incx);
};

#ifdef HAVE_CUDA
template <> 
double Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dasum(size_t N, 
    const vector_type& X, size_t incx) { 
  double result;
  cublasDasum(handle, N, X.data().get(), incx, &result);
  return result;
};
#endif

