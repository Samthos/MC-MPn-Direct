#ifndef CREATE_MP2_FUNCTIONAL_H_
#define CREATE_MP2_FUNCTIONAL_H_

#include <vector>
#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#endif

#include "mp_functional.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator> 
MP_Functional* create_MP2_Functional(int, int) { std::cerr << "Default create mp2 function. Returns nullptr\n"; exit(0); return nullptr; }

template <> MP_Functional* create_MP2_Functional<std::vector, std::allocator>(int cv_level, int electron_pairs);
#ifdef HAVE_CUDA
template <> MP_Functional* create_MP2_Functional<thrust::device_vector, thrust::device_allocator>(int cv_level, int electron_pairs);
#endif

MP_Functional* create_Direct_MP2_Functional(int cv_level);
#endif  // CREATE_MP2_FUNCTIONAL_H_
