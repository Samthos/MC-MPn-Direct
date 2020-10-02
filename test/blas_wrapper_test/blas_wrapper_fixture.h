#ifndef BLAS_WRAPPER_FIXTURE_H_
#define BLAS_WRAPPER_FIXTURE_H_

#include "blas_wrapper.h"

template <template <class, class> class Container, template <class> class Allocator>
class Blas_Wrapper_Fixture {
 public:
  typedef Container<double, Allocator<double>> vector_double;
  typedef Blas_Wrapper<Container, Allocator> Blas_Wrapper_Type;

  Blas_Wrapper_Fixture();

  size_t m;
  size_t n;
  size_t lda;
  size_t ldb;
  size_t ldc;
  size_t offset_a;
  size_t offset_b;
  size_t offset_c;
  vector_double A;
  vector_double B;
  vector_double C;
  Blas_Wrapper_Type blas_wrapper;
};

template class Blas_Wrapper_Fixture<std::vector, std::allocator>;
#ifdef HAVE_CUDA
template class Blas_Wrapper_Fixture<thrust::device_vector, thrust::device_allocator>;
#endif
#endif
