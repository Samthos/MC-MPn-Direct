#ifndef BLAS_WRAPPER_H_
#define BLAS_WRAPPER_H_

#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include "cublas_v2.h"
#endif

#include <vector>

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
class Blas_Wrapper {
  typedef Container<double, Allocator<double>> vector_double;

 public:
  Blas_Wrapper();
  ~Blas_Wrapper();
  void dgemm(bool TransA, bool TransB, 
      size_t m, size_t n, size_t k, 
      double alpha,
      const vector_double& A, size_t lda,
      const vector_double& B, size_t ldb,
      double beta,
      vector_double& C, size_t ldc);
  void Ddgmm(bool right_side,
      size_t m, size_t n,
      const vector_double& A, size_t lda,
      const vector_double& x, size_t incx,
      vector_double& B, size_t ldb);
  void dgemv(bool Trans, 
      size_t m, size_t n,
      double alpha,
      const vector_double& A, size_t lda,
      const vector_double& x, size_t incx,
      double beta,
      vector_double& y, size_t);
  void ddot(size_t N, 
      const vector_double& X, size_t incx,
      const vector_double& Y, size_t incy,
      double* result);
  void transform_multiplies(const vector_double& A, 
      const vector_double& B, 
      vector_double& C);

 private:
#ifdef HAVE_CUDA
  cublasHandle_t handle;
#endif
};

template <>
void Blas_Wrapper<std::vector, std::allocator>::dgemm(
    bool TransA, bool TransB, 
    size_t m, size_t n, size_t k, 
    double alpha,
    const vector_double& A, size_t lda,
    const vector_double& B, size_t ldb,
    double beta,
    vector_double& C, size_t ldc);

template <>
void Blas_Wrapper<std::vector, std::allocator>::Ddgmm(
    bool right_side,
    size_t m, size_t n,
    const vector_double& A, size_t lda,
    const vector_double& x, size_t incx,
    vector_double& B, size_t ldb);

template <>
void Blas_Wrapper<std::vector, std::allocator>::dgemv(
    bool Trans, 
    size_t m, size_t n,
    double alpha,
    const vector_double& A, size_t lda,
    const vector_double& x, size_t incx,
    double beta,
    vector_double& y, size_t);

template <>
void Blas_Wrapper<std::vector, std::allocator>::ddot(
    size_t N, 
    const vector_double& X, size_t incx,
    const vector_double& Y, size_t incy,
    double* result);

template <>
void Blas_Wrapper<std::vector, std::allocator>::transform_multiplies(
    const vector_double& A, 
    const vector_double& B, 
    vector_double& C);

template <> Blas_Wrapper<std::vector, std::allocator>::Blas_Wrapper();
template <> Blas_Wrapper<std::vector, std::allocator>::~Blas_Wrapper();
template class Blas_Wrapper<std::vector, std::allocator>;
typedef Blas_Wrapper<std::vector, std::allocator> Blas_Wrapper_Host;

#ifdef HAVE_CUDA
template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dgemm(
    bool TransA, bool TransB, 
    size_t m, size_t n, size_t k, 
    double alpha,
    const vector_double& A, size_t lda,
    const vector_double& B, size_t ldb,
    double beta,
    vector_double& C, size_t ldc);

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::Ddgmm(
    bool right_side,
    size_t m, size_t n,
    const vector_double& A, size_t lda,
    const vector_double& x, size_t incx,
    vector_double& B, size_t ldb);

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::dgemv(
    bool Trans, 
    size_t m, size_t n,
    double alpha,
    const vector_double& A, size_t lda,
    const vector_double& x, size_t incx,
    double beta,
    vector_double& y, size_t);

template <> void 
Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::ddot(
    size_t N, 
    const vector_double& X, size_t incx,
    const vector_double& Y, size_t incy,
    double* result);

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::transform_multiplies(
    const vector_double& A, 
    const vector_double& B, 
    vector_double& C);
template <> Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::Blas_Wrapper();
template <> Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::~Blas_Wrapper();
template class Blas_Wrapper<thrust::device_vector, thrust::device_allocator>;
typedef Blas_Wrapper<thrust::device_vector, thrust::device_allocator> Blas_Wrapper_Device;
#endif
#endif  // BLAS_WRAPPER_H_
