#ifndef OVPS_TEST_HELPER_H_
#define OVPS_TEST_HELPER_H_

#include <thrust/device_vector.h>
#include <vector>

#include "ovps_test_helper.h"

  void vfill(thrust::device_vector<double>& v) {
    thrust::fill(v.begin(), v.end(), -1);
  }
  
  std::vector<double> get_vector(thrust::device_vector<double>& v) {
    std::vector<double> w(v.size());
    thrust::copy(v.begin(), v.end(), w.begin());
    return w;
  }
  
  void vfill(std::vector<double>& v) {
    std::fill(v.begin(), v.end(), -1);
  }
  
  std::vector<double> get_vector(std::vector<double> v) {
    return v;
  }
#endif  // OVPS_TEST_HELPER_H_
