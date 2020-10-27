#include "blas_wrapper.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Blas_Wrapper<Container, Allocator>::dgekm(bool TransA, bool TransB,
    size_t m, size_t n,
    double alpha,
    const vector_type& A, size_t lda,
    const vector_type& B, size_t ldb,
    double beta,
    vector_type& C, size_t ldc) {
  this->dgekm(TransA, TransB,
      m, n,
      alpha,
      A, 0, lda,
      B, 0, ldb,
      beta,
      C, 0, ldc);
}

template <>
void Blas_Wrapper<std::vector, std::allocator>::dgekm(bool TransA, bool TransB,
    size_t m, size_t n,
    double alpha,
    const vector_type& A, size_t offset_a, size_t lda,
    const vector_type& B, size_t offset_b, size_t ldb,
    double beta,
    vector_type& C, size_t offset_c, size_t ldc) {
  const double* a_ptr = A.data() + offset_a;
  const double* b_ptr = B.data() + offset_b;
  double* c_ptr = C.data() + offset_c;

  vector_type A_T;
  vector_type B_T;
  if (TransA) {
    if (offset_a != 0) {
      std::cerr << "BAD\n";
    }
    A_T.resize(A.size());
    this->transpose(
        m, n,
        A, lda,
        A_T, n);
    a_ptr = A_T.data() + offset_a;
  }
  if (TransB) {
    if (offset_b != 0) {
      std::cerr << "BAD\n";
    }
    B_T.resize(B.size());
    this->transpose(
        m, n,
        B, lda,
        B_T, n);
    b_ptr = B_T.data() + offset_b;
  }
  if (0.0 == beta) {
    if (alpha == 1.0) {
      for (size_t col = 0; col < n; col++) {
        for (size_t row = 0; row < m; row++) {
          c_ptr[col * ldc + row] = a_ptr[col * lda + row] * b_ptr[col * ldb + row];
        }
      }
    } else {
      for (size_t col = 0; col < n; col++) {
        for (size_t row = 0; row < m; row++) {
          c_ptr[col * ldc + row] = alpha * a_ptr[col * lda + row] * b_ptr[col * ldb + row];
        }
      }
    }
  } else if (1.0 == beta) {
    if (alpha == 1.0) {
      for (size_t col = 0; col < n; col++) {
        for (size_t row = 0; row < m; row++) {
          c_ptr[col * ldc + row] = a_ptr[col * lda + row] * b_ptr[col * ldb + row] + c_ptr[col * ldc + row];
        }
      }
    } else {
      for (size_t col = 0; col < n; col++) {
        for (size_t row = 0; row < m; row++) {
          c_ptr[col * ldc + row] = alpha * a_ptr[col * lda + row] * b_ptr[col * ldb + row] + c_ptr[col * ldc + row];
        }
      }
    }
  } else {
    if (alpha == 1.0) {
      for (size_t col = 0; col < n; col++) {
        for (size_t row = 0; row < m; row++) {
          c_ptr[col * ldc + row] = a_ptr[col * lda + row] * b_ptr[col * ldb + row] + beta * c_ptr[col * ldc + row];
        }
      }
    } else {
      for (size_t col = 0; col < n; col++) {
        for (size_t row = 0; row < m; row++) {
          c_ptr[col * ldc + row] = alpha * a_ptr[col * lda + row] * b_ptr[col * ldb + row] + beta * c_ptr[col * ldc + row];
        }
      }
    }
  }
}

#ifdef HAVE_CUDA
__global__ void dgekm_kernel(size_t m, size_t n,
    double alpha,
    const double* A, size_t lda,
    const double* B, size_t ldb,
    double beta,
    double* C, size_t ldc) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < n) {
    C[col * ldc + row] = alpha * A[col * lda + row] * B[col * ldb + row] + beta * C[col * ldc + row];
  }
}

__global__ void dgekm_kernel_0b(size_t m, size_t n,
    double alpha,
    const double* A, size_t lda,
    const double* B, size_t ldb,
    double* C, size_t ldc) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < n) {
    C[col * ldc + row] = alpha * A[col * lda + row] * B[col * ldb + row];
  }
}

__global__ void dgekm_kernel_1b(size_t m, size_t n,
    double alpha,
    const double* A, size_t lda,
    const double* B, size_t ldb,
    double* C, size_t ldc) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < n) {
    C[col * ldc + row] = alpha * A[col * lda + row] * B[col * ldb + row] + C[col * ldc + row];
  }
}

__global__ void dgekm_kernel_1a(size_t m, size_t n,
    const double* A, size_t lda,
    const double* B, size_t ldb,
    double beta,
    double* C, size_t ldc) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < n) {
    C[col * ldc + row] = A[col * lda + row] * B[col * ldb + row] + beta * C[col * ldc + row];
  }
}

__global__ void dgekm_kernel_1a_0b(size_t m, size_t n,
    const double* A, size_t lda,
    const double* B, size_t ldb,
    double* C, size_t ldc) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < n) {
    C[col * ldc + row] = A[col * lda + row] * B[col * ldb + row];
  }
}

__global__ void dgekm_kernel_1a_0b_fast(size_t m,
    const double* A,
    const double* B,
    double* C) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m) {
    C[row] = A[row] * B[row];
  }
}

__global__ void dgekm_kernel_1a_1b(size_t m, size_t n,
    const double* A, size_t lda,
    const double* B, size_t ldb,
    double* C, size_t ldc) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < m && col < n) {
    C[col * ldc + row] = A[col * lda + row] * B[col * ldb + row] + C[col * ldc + row];
  }
}

template <>
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dgekm(bool TransA, bool TransB,
    size_t m, size_t n,
    double alpha,
    const vector_type& A, size_t offset_a, size_t lda,
    const vector_type& B, size_t offset_b, size_t ldb,
    double beta,
    vector_type& C, size_t offset_c, size_t ldc) {
  const double* a_ptr = A.data().get() + offset_a;
  const double* b_ptr = B.data().get() + offset_b;
  double* c_ptr = C.data().get() + offset_c;
  dim3 blockSize(16, 16, 1);
  dim3 gridSize(
      (m + blockSize.x - 1) / blockSize.x,
      (n + blockSize.y - 1) / blockSize.y,
      1);
  vector_type A_T;
  vector_type B_T;
  if (TransA) {
    if (offset_a != 0) {
      std::cerr << "BAD\n";
    }
    A_T.resize(A.size());
    this->transpose(
        m, n,
        A, lda,
        A_T, n);
    a_ptr = A_T.data().get() + offset_a;
  }
  if (TransB) {
    if (offset_b != 0) {
      std::cerr << "BAD\n";
    }
    B_T.resize(B.size());
    this->transpose(
        m, n,
        B, lda,
        B_T, n);
    b_ptr = B_T.data().get() + offset_b;
  }
 if (0.0 == beta) {
   if (alpha == 1.0) {
     if (lda == ldb && lda == ldc && lda == m) {
      blockSize.x = 128;
      blockSize.y = 1;
      gridSize.x = (m*n + blockSize.x - 1) / blockSize.x;
      gridSize.y = 1;
      dgekm_kernel_1a_0b_fast<<<gridSize, blockSize>>>(m*n, a_ptr, b_ptr, c_ptr);
     } else {
      dgekm_kernel_1a_0b<<<gridSize, blockSize>>>(m, n, a_ptr, lda, b_ptr, ldb, c_ptr, ldc);
     }
   } else {
     dgekm_kernel_0b<<<gridSize, blockSize>>>(m, n, alpha, a_ptr, lda, b_ptr, ldb, c_ptr, ldc);
   }
 } else if (beta == 1.0) {
   if (alpha == 1.0) {
     dgekm_kernel_1a_1b<<<gridSize, blockSize>>>(m, n, a_ptr, lda, b_ptr, ldb, c_ptr, ldc);
   } else {
     dgekm_kernel_1b<<<gridSize, blockSize>>>(m, n, alpha, a_ptr, lda, b_ptr, ldb, c_ptr, ldc);
   }
 } else {
   if (alpha == 1.0) {
     dgekm_kernel_1a<<<gridSize, blockSize>>>(m, n, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc);
   } else {
      dgekm_kernel<<<gridSize, blockSize>>>(m, n, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc);
   }
 }
  cudaDeviceSynchronize();
}
#endif
