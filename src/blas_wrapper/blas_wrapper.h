#ifndef BLAS_WRAPPER_H_
#define BLAS_WRAPPER_H_

#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#include <thrust/device_allocator.h>
#include "cublas_v2.h"
#endif

#include <vector>

namespace BLAS_WRAPPER {
  enum Fill_Mode_t {
    FILL_FULL, 
    FILL_LOWER,
    FILL_UPPER,
  };

  enum Transpose{
    NoTrans, 
    Trans,
    Adjoint,
  };

  enum Side_t {
    RIGHT_SIDE,
    LEFT_SIDE
  };
}

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
class Blas_Wrapper {
  typedef double value_type;
  typedef Container<value_type, Allocator<value_type>> vector_type;
  typedef typename vector_type::iterator iterator;
  typedef typename vector_type::const_iterator const_iterator;

 public:
  Blas_Wrapper();
  Blas_Wrapper(const Blas_Wrapper&);
  Blas_Wrapper operator = (const Blas_Wrapper&);
  ~Blas_Wrapper();

  // 
  // Template Level 3 Blas 
  //
  void dgemm(bool TransA, bool TransB, 
      size_t m, size_t n, size_t k, 
      value_type alpha,
      const vector_type& A, size_t lda,
      const vector_type& B, size_t ldb,
      value_type beta,
      vector_type& C, size_t ldc);

  void dsyrk(BLAS_WRAPPER::Fill_Mode_t fill_mode, bool Trans, 
      size_t m, size_t k, 
      value_type alpha,
      const vector_type& A, size_t lda,
      value_type beta,
      vector_type& B, size_t ldb);

  void dherk(BLAS_WRAPPER::Fill_Mode_t fill_mode, bool Trans, 
      size_t m, size_t k, 
      value_type alpha,
      const vector_type& A, size_t lda,
      value_type beta,
      vector_type& B, size_t ldb);

  // 
  // Instantiated Level 3 Blas 
  //
  void dgemm(bool TransA, bool TransB, 
      size_t m, size_t n, size_t k, 
      value_type alpha,
      const vector_type& A, size_t offset_a, size_t lda,
      const vector_type& B, size_t offset_b, size_t ldb,
      value_type beta,
      vector_type& C, size_t offset_c, size_t ldc);


  void dsyrk(BLAS_WRAPPER::Fill_Mode_t fill_mode, bool Trans, 
      size_t m, size_t k, 
      value_type alpha,
      const vector_type& A, size_t offset_a, size_t lda,
      value_type beta,
      vector_type& B, size_t offset_b, size_t ldb);

  void dherk(BLAS_WRAPPER::Fill_Mode_t fill_mode, bool Trans, 
      size_t m, size_t k, 
      value_type alpha,
      const vector_type& A, size_t offset_a, size_t lda,
      value_type beta,
      vector_type& B, size_t offset_b, size_t ldb);

  // 
  // Template Level 2 Blas 
  //
  void batched_ddot(size_t N, size_t K,
      const vector_type& A, size_t offset_a, size_t lda,
      const vector_type& B, size_t offset_b, size_t ldb,
      vector_type& X, size_t incx);

  void dgeam(bool TransA, bool TransB,
      size_t m, size_t n,
      value_type alpha,
      const vector_type& A, size_t lda,
      value_type beta,
      const vector_type& B, size_t ldb,
      vector_type& C, size_t ldc);

  void dgekm(bool TransA, bool TransB, 
      size_t m, size_t n,
      value_type alpha,
      const vector_type& A, size_t lda,
      const vector_type& B, size_t ldb,
      value_type beta,
      vector_type& C, size_t ldc);

  void dgemv(bool Trans, 
      size_t m, size_t n,
      value_type alpha,
      const vector_type& A, size_t lda,
      const vector_type& x, size_t incx,
      value_type beta,
      vector_type& y, size_t);

  void ddgmm(BLAS_WRAPPER::Side_t side,
      size_t m, size_t n,
      const vector_type& A, size_t lda,
      const vector_type& x, size_t incx,
      vector_type& B, size_t ldb);

  void mfill(size_t m, size_t n,
      double alpha,
      vector_type& A, size_t lda);

  // 
  // Instantiated Level 2 Blas 
  //
  void ddgmm(BLAS_WRAPPER::Side_t side,
      size_t m, size_t n,
      const vector_type& A, size_t offset_a, size_t lda,
      const vector_type& x, size_t offset_x, size_t incx,
      vector_type& B, size_t offset_b, size_t ldb);

  void dgeam(bool TransA, bool TransB,
      size_t m, size_t n,
      value_type alpha,
      const vector_type& A, size_t offset_a, size_t lda,
      value_type beta,
      const vector_type& B, size_t offset_b, size_t ldb,
      vector_type& C, size_t offset_c, size_t ldc);

  void dgekm(bool TransA, bool TransB, 
      size_t m, size_t n,
      value_type alpha,
      const vector_type& A, size_t offset_a, size_t lda,
      const vector_type& B, size_t offset_b, size_t ldb,
      value_type beta,
      vector_type& C, size_t offset_c, size_t ldc);

  void dgemv(bool Trans, 
      size_t m, size_t n,
      value_type alpha,
      const vector_type& A, size_t offset_a, size_t lda,
      const vector_type& x, size_t offset_x, size_t incx,
      value_type beta,
      vector_type& y, size_t offset_y, size_t);

  void mfill(size_t m, size_t n,
      double alpha,
      vector_type& A, size_t offset_a, size_t lda);

  void transpose(size_t m, size_t n,
      const vector_type& A, size_t lda,
      vector_type& B, size_t ldb);

  // 
  // Template Level 1 Blas 
  //
  double dasum(size_t N, const vector_type& X, size_t incx);

  void daxpy(size_t N,
      value_type alpha,
      const vector_type& X, size_t incx,
      vector_type& Y, size_t incy);

  void dcopy(size_t N,
      const vector_type& X, size_t incx,
      vector_type& Y, size_t incy);

  value_type ddot(size_t N, 
      const vector_type& X, size_t incx,
      const vector_type& Y, size_t incy);

  void ddot(size_t N, 
      const vector_type& X, size_t incx,
      const vector_type& Y, size_t incy, 
      value_type* result);

  void dgekv(size_t m,
      value_type alpha,
      const vector_type& A, size_t inc_a,
      const vector_type& B, size_t inc_b,
      value_type beta,
      vector_type& C, size_t inc_c);

  void shift(size_t m,
      value_type alpha,
      const vector_type& X, size_t incx,
      vector_type& Y, size_t incy);

  void vfill(size_t n,
      double alpha,
      vector_type& X, size_t inc_x);


  // 
  // Instantiated Level 1 Blas 
  //
  void daxpy(size_t N,
      value_type alpha,
      const vector_type& X, size_t offset_x, size_t incx,
      vector_type& Y, size_t offset_y, size_t incy);

  void dcopy(size_t N,
      const vector_type& X, size_t offset_x, size_t incx,
      vector_type& Y, size_t offset_y, size_t incy);

  value_type ddot(size_t N, 
      const vector_type& X, size_t offset_x, size_t incx,
      const vector_type& Y, size_t offset_y, size_t incy);

  void ddot(size_t N, 
      const vector_type& X, size_t offset_x, size_t incx,
      const vector_type& Y, size_t offset_y, size_t incy, 
      value_type* result);

  void dgekv(size_t m,
      value_type alpha,
      const vector_type& A, size_t offset_a, size_t inc_a,
      const vector_type& B, size_t offset_b, size_t inc_b,
      value_type beta,
      vector_type& C, size_t offset_c, size_t inc_c);

  void dscal(size_t N,
      value_type alpha,
      vector_type& X, size_t incx);

  void shift(size_t m,
      value_type alpha,
      const vector_type& X, size_t offset_x, size_t incx,
      vector_type& Y, size_t offset_y, size_t incy);

  void vfill(size_t n,
      double alpha,
      vector_type& X, size_t offset_x, size_t inc_x);

  //
  // Iterator 
  //
  double accumulate(size_t N, const vector_type& X, size_t incx);
  double accumulate(size_t N, const vector_type& X, size_t offset_x, size_t incx);

  void fill(iterator first1, iterator last1, value_type value);

  void minus(const_iterator first1, const_iterator last1,
      const_iterator first2, iterator result);

  void multiplies(const_iterator first1, const_iterator last1,
      const_iterator first2, iterator result);

  void plus(const_iterator first1, const_iterator last1,
      const_iterator first2, iterator result);

 private:
  vector_type one;
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
