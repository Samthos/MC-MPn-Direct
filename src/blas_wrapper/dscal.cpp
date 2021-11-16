#ifdef HAVE_CUDA
#include "cublas_v2.h"
#endif

#include "cblas.h"
#include "blas_wrapper.h"

template <> 
void Blas_Wrapper<std::vector, std::allocator>::dscal(size_t N, 
    double alpha,
    vector_type& X, size_t incx) { 
  cblas_dscal(N,
      alpha,
      X.data(), incx);
};

#ifdef HAVE_CUDA
template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dscal(size_t N, 
    double alpha,
    vector_type& X, size_t incx) { 
  cublasDscal(handle,
      N,
      &alpha,
      X.data().get(), incx);
};
#endif

