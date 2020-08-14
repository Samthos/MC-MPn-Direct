#include "wavefunction.cpp"

template <>
double* Wavefunction<thrust::device_vector, thrust::device_allocator>::get_raw_pointer(thrust::device_vector<double, thrust::device_allocator<double>>& v) {
  return v.data().get();
}

template <>
const double* Wavefunction<thrust::device_vector, thrust::device_allocator>::get_raw_pointer(const thrust::device_vector<double, thrust::device_allocator<double>>& v) {
  return v.data().get();
}
