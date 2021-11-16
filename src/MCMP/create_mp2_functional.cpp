#include <iostream>

#include "create_mp2_functional.h"

#include "mp2_functional.h"
#include "direct_mp2_functional.h"
#ifdef HAVE_CUDA
#include "device_mp2_functional.h"
#endif

template <> 
MP_Functional* create_MP2_Functional<std::vector, std::allocator>(int cv_level, int) {
  MP_Functional* mcmp = nullptr;
  if (cv_level == 0) {
    mcmp = new MP2_Functional<0>;
  } else if (cv_level == 1) {
    mcmp = new MP2_Functional<1>;
  } else if (cv_level == 2) {
    mcmp = new MP2_Functional<2>;
  }
  
  if (mcmp == nullptr) {
    std::cerr << "MP2_Functional not supported with cv level " << cv_level << "\n";
    exit(0);
  }
  return mcmp;
}

MP_Functional* create_Direct_MP2_Functional(int cv_level) {
  MP_Functional* mcmp = nullptr;
  if (cv_level == 0) {
    mcmp = new Direct_MP2_Functional<0>;
  } else if (cv_level == 1) {
    mcmp = new Direct_MP2_Functional<1>;
  } else if (cv_level == 2) {
    mcmp = new Direct_MP2_Functional<2>;
  }
  
  if (mcmp == nullptr) {
    std::cerr << "MP2_Functional not supported with cv level " << cv_level << "\n";
    exit(0);
  }
  return mcmp;
}

#ifdef HAVE_CUDA
template <> 
MP_Functional* create_MP2_Functional<thrust::device_vector, thrust::device_allocator>(int cv_level, int electron_pairs) {
  MP_Functional* mcmp = nullptr;
  if (cv_level == 0) {
    mcmp = new Device_MP2_Functional<0>(electron_pairs);
  } else if (cv_level == 1) {
    mcmp = new Device_MP2_Functional<1>(electron_pairs);
  } else if (cv_level == 2) {
    mcmp = new Device_MP2_Functional<2>(electron_pairs);
  }
  
  if (mcmp == nullptr) {
    std::cerr << "MP2_Functional not supported with cv level " << cv_level << "\n";
    exit(0);
  }
  return mcmp;
}
#endif
