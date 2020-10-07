#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#endif

#include "blas_wrapper.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Blas_Wrapper<Container, Allocator>::shift(
    size_t m,
    double alpha,
    const vector_type& X, size_t inc_x,
    vector_type& Y, size_t inc_y) {
  this->shift(
      m,
      alpha,
      X, 0, inc_x,
      Y, 0, inc_y);
}

template <>
void Blas_Wrapper<std::vector, std::allocator>::shift(size_t m,
    double alpha,
    const vector_type& X, size_t offset_x, size_t inc_x,
    vector_type& Y, size_t offset_y, size_t inc_y) {
  const double* x_ptr = X.data() + offset_x;
  double* y_ptr = Y.data() + offset_y;
  for (int idx = 0; idx < m; idx++) {
    Y[idx * inc_y] = X[idx * inc_x] + alpha;
  }
}

#ifdef HAVE_CUDA
__global__ void shift_kernel(size_t m,
    double alpha,
    const double* X, size_t inc_x,
    double* Y, size_t inc_y) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    Y[idx * inc_y] = X[idx * inc_x] + alpha;
  }
}

template <>
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::shift(size_t m,
    double alpha,
    const vector_type& X, size_t offset_x, size_t inc_x,
    vector_type& Y, size_t offset_y, size_t inc_y) {
  const double* x_ptr = X.data().get() + offset_x;
  double* y_ptr = Y.data().get() + offset_y;
  dim3 blockSize(128, 1, 1);
  dim3 gridSize((m + blockSize.x - 1) / blockSize.x, 1, 1);
  shift_kernel<<<gridSize, blockSize>>>(m, alpha, x_ptr, inc_x, y_ptr, inc_y);
  cudaDeviceSynchronize();
}
#endif
