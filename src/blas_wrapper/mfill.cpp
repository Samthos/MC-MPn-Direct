#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include "cublas_v2.h"
#endif

#include "cblas.h"
#include "blas_wrapper.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::mfill(
    size_t m, size_t n,
    double alpha,
    vector_type& A, size_t lda) {
  this->mfill(m, n,
      alpha,
      A, 0, lda);
}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::mfill(
    size_t m, size_t n,
    double alpha,
    vector_type& A, size_t offset_a, size_t lda) {
  double* a_ptr = A.data() + offset_a;
  for (int col = 0; col < n; col++) {
    for (int row = 0; row < m; row++) {
      a_ptr[col * lda + row] = alpha;
    }
  }
}

#ifdef HAVE_CUDA
__global__ 
void mfill_kernel(
    size_t m, size_t n,
    double alpha,
    double* a_ptr, size_t lda) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < n) {
    a_ptr[col * lda + row] = alpha;
  }
}

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::mfill(
    size_t m, size_t n,
    double alpha,
    vector_type& A, size_t offset_a, size_t lda) {
  dim3 blockSize(32, 32, 1);
  dim3 gridSize(
      (m + blockSize.x - 1) / blockSize.x,
      (n + blockSize.y - 1) / blockSize.y,
      1);
  double* a_ptr = A.data().get() + offset_a;
  mfill_kernel<<<gridSize, blockSize>>>(m, n, alpha, a_ptr, lda);
};
#endif

