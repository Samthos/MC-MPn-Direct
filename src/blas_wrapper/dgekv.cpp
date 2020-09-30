#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#endif

#include "blas_wrapper.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
void Blas_Wrapper<Container, Allocator>::dgekv(
    size_t m,
    double alpha,
    const vector_type& A, size_t inc_a,
    const vector_type& B, size_t inc_b,
    double beta,
    vector_type& C, size_t inc_c) {
  this->dgekv(
      m,
      alpha, 
      A, 0, inc_a,
      B, 0, inc_b,
      beta,
      C, 0, inc_c);
}

template <>
void Blas_Wrapper<std::vector, std::allocator>::dgekv(
    size_t m,
    double alpha,
    const vector_type& A, size_t offset_a, size_t inc_a,
    const vector_type& B, size_t offset_b, size_t inc_b,
    double beta,
    vector_type& C, size_t offset_c, size_t inc_c) {
  const double* a_ptr = A.data() + offset_a;
  const double* b_ptr = B.data() + offset_b;
  double* c_ptr = C.data() + offset_c;
  if (0.0 == beta) {
    if (alpha == 1.0) {
      for (size_t idx = 0; idx < m; idx++) {
        c_ptr[idx * inc_c] = a_ptr[idx * inc_c] * b_ptr[idx * inc_b];
      }
    } else {
      for (size_t idx = 0; idx < m; idx++) {
        c_ptr[idx * inc_c] = alpha * a_ptr[idx * inc_c] * b_ptr[idx * inc_b];
      }
    }
  } else if (1.0 == beta) {
    if (alpha == 1.0) {
      for (size_t idx = 0; idx < m; idx++) {
        c_ptr[idx * inc_c] = a_ptr[idx * inc_c] * b_ptr[idx * inc_b] + c_ptr[idx * inc_c];
      }
    } else {
      for (size_t idx = 0; idx < m; idx++) {
        c_ptr[idx * inc_c] = alpha * a_ptr[idx * inc_c] * b_ptr[idx * inc_b] + c_ptr[idx * inc_c];
      }
    }
  } else {
    if (alpha == 1.0) {
      for (size_t idx = 0; idx < m; idx++) {
        c_ptr[idx * inc_c] = a_ptr[idx * inc_c] * b_ptr[idx * inc_b] + beta * c_ptr[idx * inc_c];
      }
    } else {
      for (size_t idx = 0; idx < m; idx++) {
        c_ptr[idx * inc_c] = alpha * a_ptr[idx * inc_c] * b_ptr[idx * inc_b] + beta * c_ptr[idx * inc_c];
      }
    }
  }
}

#ifdef HAVE_CUDA
__global__
void dgekv_kernel(size_t m,
    double alpha,
    const double* A, size_t inc_a,
    const double* B, size_t inc_b,
    double beta,
    double* C, size_t inc_c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    C[idx * inc_c] = A[idx * inc_c] * B[idx * inc_b];
  }
}

__global__
void dgekv_kernel_0b(size_t m,
    double alpha,
    const double* A, size_t inc_a,
    const double* B, size_t inc_b,
    double* C, size_t inc_c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    C[idx * inc_c] = alpha * A[idx * inc_c] * B[idx * inc_b];
  }
}

__global__
void dgekv_kernel_1b(size_t m,
    double alpha,
    const double* A, size_t inc_a,
    const double* B, size_t inc_b,
    double* C, size_t inc_c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    C[idx * inc_c] = alpha * A[idx * inc_c] * B[idx * inc_b] + C[idx * inc_c];
  }
}

__global__
void dgekv_kernel_1a(size_t m,
    const double* A, size_t inc_a,
    const double* B, size_t inc_b,
    double beta,
    double* C, size_t inc_c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    C[idx * inc_c] = A[idx * inc_c] * B[idx * inc_b] + beta * C[idx * inc_c];
  }
}

__global__
void dgekv_kernel_1a_0b(size_t m,
    const double* A, size_t inc_a,
    const double* B, size_t inc_b,
    double* C, size_t inc_c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    C[idx * inc_c] = A[idx * inc_c] * B[idx * inc_b];
  }
}

__global__
void dgekv_kernel_1a_1b(size_t m,
    const double* A, size_t inc_a,
    const double* B, size_t inc_b,
    double* C, size_t inc_c) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m) {
    C[idx * inc_c] = A[idx * inc_c] * B[idx * inc_b] + C[idx * inc_c];
  }
}

template <>
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dgekv(size_t m,
    double alpha,
    const vector_type& A, size_t offset_a, size_t inc_a,
    const vector_type& B, size_t offset_b, size_t inc_b,
    double beta,
    vector_type& C, size_t offset_c, size_t inc_c) {
  const double* a_ptr = A.data().get() + offset_a;
  const double* b_ptr = B.data().get() + offset_b;
  double* c_ptr = C.data().get() + offset_c;
  dim3 blockSize(128, 1, 1);
  dim3 gridSize((m + blockSize.x - 1) / blockSize.x, 1, 1);
  if (0.0 == beta) {
    if (alpha == 1.0) {
      dgekv_kernel_1a_0b<<<gridSize, blockSize>>>(m, a_ptr, inc_a, b_ptr, inc_b, c_ptr, inc_c);
    } else {
      dgekv_kernel_0b<<<gridSize, blockSize>>>(m, alpha, a_ptr, inc_a, b_ptr, inc_b, c_ptr, inc_c);
    }
  } else if (beta == 1.0) {
    if (alpha == 1.0) {
      dgekv_kernel_1a_1b<<<gridSize, blockSize>>>(m, a_ptr, inc_a, b_ptr, inc_b, c_ptr, inc_c);
    } else {
      dgekv_kernel_1b<<<gridSize, blockSize>>>(m, alpha, a_ptr, inc_a, b_ptr, inc_b, c_ptr, inc_c);
    }
  } else {
    if (alpha == 1.0) {
      dgekv_kernel_1a<<<gridSize, blockSize>>>(m, a_ptr, inc_a, b_ptr, inc_b, beta, c_ptr, inc_c);
    } else {
      dgekv_kernel<<<gridSize, blockSize>>>(m, alpha, a_ptr, inc_a, b_ptr, inc_b, beta, c_ptr, inc_c);
    }
  }
}
#endif
