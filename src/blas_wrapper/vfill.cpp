#ifdef HAVE_CUDA
#include "cublas_v2.h"
#endif

#include "cblas.h"
#include "blas_wrapper.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::vfill(
    size_t n,
    double alpha,
    vector_type& X, size_t inc_x) {
  this->vfill(n,
      alpha,
      X, 0, inc_x);
}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::vfill(
    size_t n,
    double alpha,
    vector_type& X, size_t offset_x, size_t inc_x) {
  double* x_ptr = X.data() + offset_x;
  for (int row = 0; row < n; row++) {
    x_ptr[row * inc_x] = alpha;
  }
}

#ifdef HAVE_CUDA
__global__ 
void vfill_kernel(
    size_t n,
    double alpha,
    double* x_ptr, size_t inc_x) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n) {
    x_ptr[row * inc_x] = alpha;
  }
}

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::vfill(
    size_t n,
    double alpha,
    vector_type& X, size_t offset_x, size_t inc_x) {
  dim3 blockSize(128, 1, 1);
  dim3 gridSize(
      (n + blockSize.x - 1) / blockSize.x,
      1,
      1);
  double* x_ptr = X.data().get() + offset_x;
  vfill_kernel<<<gridSize, blockSize>>>(n, alpha, x_ptr, inc_x);
};
#endif

