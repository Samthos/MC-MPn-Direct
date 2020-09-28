#ifndef BLAS_WRAPPER_H_
#define BLAS_WRAPPER_H_

#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include "cublas_v2.h"
#endif

#include <vector>

namespace BLAS_WRAPPER {
  enum Fill_Mode_t {
    FILL_FULL, 
    FILL_LOWER,
    FILL_UPPER,
  };
}

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
class Blas_Wrapper {
  typedef Container<double, Allocator<double>> vector_double;
  typedef typename vector_double::iterator iterator;
  typedef typename vector_double::const_iterator const_iterator;

 public:
  Blas_Wrapper();
  ~Blas_Wrapper();

  // 
  // Template Level 3 Blas 
  //
  void dgemm(bool TransA, bool TransB, 
      size_t m, size_t n, size_t k, 
      double alpha,
      const vector_double& A, size_t lda,
      const vector_double& B, size_t ldb,
      double beta,
      vector_double& C, size_t ldc);

  void dsyrk(BLAS_WRAPPER::Fill_Mode_t fill_mode, bool Trans, 
      size_t m, size_t k, 
      double alpha,
      const vector_double& A, size_t lda,
      double beta,
      vector_double& B, size_t ldb);

  // 
  // Instantiated Level 3 Blas 
  //
  void dgemm(bool TransA, bool TransB, 
      size_t m, size_t n, size_t k, 
      double alpha,
      const vector_double& A, size_t offset_a, size_t lda,
      const vector_double& B, size_t offset_b, size_t ldb,
      double beta,
      vector_double& C, size_t offset_c, size_t ldc);


  void dsyrk(BLAS_WRAPPER::Fill_Mode_t fill_mode, bool Trans, 
      size_t m, size_t k, 
      double alpha,
      const vector_double& A, size_t offset_a, size_t lda,
      double beta,
      vector_double& B, size_t offset_b, size_t ldb);

  // 
  // Template Level 2 Blas 
  //
  void dgemv(bool Trans, 
      size_t m, size_t n,
      double alpha,
      const vector_double& A, size_t lda,
      const vector_double& x, size_t incx,
      double beta,
      vector_double& y, size_t);


  // 
  // Instantiated Level 2 Blas 
  //
  void batched_ddot(size_t N, size_t K,
      const vector_double& A, size_t offset_a, size_t lda,
      const vector_double& B, size_t offset_b, size_t ldb,
      vector_double& X, size_t incx);

  void ddgmm(bool right_side,
      size_t m, size_t n,
      const vector_double& A, size_t lda,
      const vector_double& x, size_t incx,
      vector_double& B, size_t ldb);

  void dgemv(bool Trans, 
      size_t m, size_t n,
      double alpha,
      const vector_double& A, size_t offset_a, size_t lda,
      const vector_double& x, size_t offset_x, size_t incx,
      double beta,
      vector_double& y, size_t offset_y, size_t);

  // 
  // Template Level 1 Blas 
  //
  void dcopy(size_t N,
      const vector_double& X, size_t incx,
      vector_double& Y, size_t incy);

  double ddot(size_t N, 
      const vector_double& X, size_t incx,
      const vector_double& Y, size_t incy);

  void ddot(size_t N, 
      const vector_double& X, size_t incx,
      const vector_double& Y, size_t incy, 
      double* result);

  // 
  // Instantiated Level 1 Blas 
  //
  void dcopy(size_t N,
      const vector_double& X, size_t offset_x, size_t incx,
      vector_double& Y, size_t offset_y, size_t incy);

  double ddot(size_t N, 
      const vector_double& X, size_t offset_x, size_t incx,
      const vector_double& Y, size_t offset_y, size_t incy);

  void ddot(size_t N, 
      const vector_double& X, size_t offset_x, size_t incx,
      const vector_double& Y, size_t offset_y, size_t incy, 
      double* result);

  void dscal(size_t N,
      double alpha,
      vector_double& X, size_t incx);

  //
  // Iterator 
  //
  void fill(iterator first1, iterator last1, double value);

  void minus(const_iterator first1, const_iterator last1,
      const_iterator first2, iterator result);

  void multiplies(const_iterator first1, const_iterator last1,
      const_iterator first2, iterator result);

  void plus(const_iterator first1, const_iterator last1,
      const_iterator first2, iterator result);

 private:
#ifdef HAVE_CUDA
  cublasHandle_t handle;
#endif
};

#define VECTOR_TYPE std::vector
#define ALLOCATOR_TYPE std::allocator
#include "blas_wrapper_declarations.h"
#undef VECTOR_TYPE
#undef ALLOCATOR_TYPE 
typedef Blas_Wrapper<std::vector, std::allocator> Blas_Wrapper_Host;

#ifdef HAVE_CUDA
#define VECTOR_TYPE thrust::device_vector
#define ALLOCATOR_TYPE thrust::device_allocator
#include "blas_wrapper_declarations.h"
#undef VECTOR_TYPE
#undef ALLOCATOR_TYPE 
typedef Blas_Wrapper<thrust::device_vector, thrust::device_allocator> Blas_Wrapper_Device;
#endif
#endif  // BLAS_WRAPPER_H_
