#include "cublas_v2.h"
#include "wavefunction.cpp"

template <>
void Wavefunction<thrust::device_vector, thrust::device_allocator>::ao_to_mo(const vector_double& ao_amplitudes) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  thrust::fill(psi.begin(), psi.end(), 0.0);

  double alpha = 1.00;
  double beta = 0.00;
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      lda, electrons, n_basis_functions,
      &alpha,
      movecs.data().get(), n_basis_functions,
      ao_amplitudes.data().get(), n_basis_functions,
      &beta,
      psi.data().get(), lda);
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
