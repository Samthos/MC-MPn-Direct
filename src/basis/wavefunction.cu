#include "cublas_v2.h"
#include "wavefunction.cpp"

template <>
void Wavefunction<thrust::device_vector, thrust::device_allocator>::ao_to_mo(const vector_double& ao_amplitudes) {
  cublasHandle_t handle;
  cublasCreate(&handle);
// cblas_dgemm(handle, CblasRowMajor, CblasNoTrans, CblasTrans,
//     electrons, lda, n_basis_functions,
//     1.0,
//     ao_amplitudes.data(), n_basis_functions,
//     movecs.data(), n_basis_functions,
//     0.0,
//     psi.data(), lda);
  cublasDestroy(handle);
}

template <>
double* Wavefunction<thrust::device_vector, thrust::device_allocator>::get_raw_pointer(thrust::device_vector<double, thrust::device_allocator<double>>& v) {
  return v.data().get();
}

template <>
const double* Wavefunction<thrust::device_vector, thrust::device_allocator>::get_raw_pointer(const thrust::device_vector<double, thrust::device_allocator<double>>& v) {
  return v.data().get();
}
