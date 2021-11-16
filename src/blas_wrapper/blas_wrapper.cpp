#ifdef HAVE_CUDA
#include "cublas_v2.h"
#endif

#include <algorithm>
#include <iostream>
#include <functional>

#include "cblas.h"
#include "blas_calls.h"

#include "blas_wrapper.h"

template <> 
Blas_Wrapper<std::vector, std::allocator>::Blas_Wrapper() : one(1, 1.0) {}

template <> 
Blas_Wrapper<std::vector, std::allocator>::Blas_Wrapper(const Blas_Wrapper&) : one(1, 1.0) {}

template <> 
Blas_Wrapper<std::vector, std::allocator> Blas_Wrapper<std::vector, std::allocator>::operator = (const Blas_Wrapper<std::vector, std::allocator>& other) {
  if (this != &other) {
    one.push_back(1.0);
  }
  return *this;
}

template <> 
Blas_Wrapper<std::vector, std::allocator>::~Blas_Wrapper() {}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::fill(
    iterator first1, iterator last1, double value) {
  std::fill(first1, last1, value);
}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::minus(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result) {
  std::transform(first1, last1, first2, result, std::minus<>());
}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::multiplies(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result) {
  std::transform(first1, last1, first2, result, std::multiplies<>());
}

template <> 
void Blas_Wrapper<std::vector, std::allocator>::plus(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result) {
  std::transform(first1, last1, first2, result, std::plus<>());
}


#ifdef HAVE_CUDA
template <> 
Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::Blas_Wrapper() : one(1, 1.0) {
  cublasCreate(&handle);
}

template <> 
Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::Blas_Wrapper(const Blas_Wrapper&) : one(1, 1.0) {
  cublasCreate(&handle);
}

template <> 
Blas_Wrapper<thrust::device_vector, thrust::device_allocator> Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::operator = (const Blas_Wrapper<thrust::device_vector, thrust::device_allocator>& other) {
  if (this != &other) {
    cublasCreate(&handle);
    one = other.one;
  }
  return *this;
}

template <> 
Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::~Blas_Wrapper() {
  cublasDestroy(handle);
}

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::fill(
    iterator first1, iterator last1, double value) {
  thrust::fill(first1, last1, value);
}

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::minus(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result) {
  thrust::transform(first1, last1, first2, result, thrust::minus<double>());
}

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::multiplies(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result) {
  thrust::transform(first1, last1, first2, result, thrust::multiplies<double>());
}

template <> 
void Blas_Wrapper<thrust::device_vector, thrust::device_allocator>::plus(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result) {
  thrust::transform(first1, last1, first2, result, thrust::plus<double>());
}
#endif

