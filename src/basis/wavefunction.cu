#include "cublas_v2.h"
#include "wavefunction.cpp"

template <>
std::shared_ptr<void> Wavefunction<thrust::device_vector, thrust::device_allocator>::create_handle() {
  std::shared_ptr<void> l_handle(new cublasHandle_t);
  auto handle = static_cast<cublasHandle_t*>(l_handle.get());
  cublasCreate(handle);
  return l_handle;
}

template <>
void Wavefunction<thrust::device_vector, thrust::device_allocator>::destroy_handle() {
  if (v_handle.unique()) {
    auto handle = static_cast<cublasHandle_t*>(v_handle.get());
    cublasCreate(handle);
  }
}

template <>
void Wavefunction<thrust::device_vector, thrust::device_allocator>::ao_to_mo(const vector_double& ao_amplitudes) {
  auto handle = std::static_pointer_cast<cublasHandle_t>(v_handle);
  double alpha = 1.00;
  double beta = 0.00;
  cublasDgemm(*handle, CUBLAS_OP_T, CUBLAS_OP_N,
      lda, electrons, n_basis_functions,
      &alpha,
      movecs.data().get(), n_basis_functions,
      ao_amplitudes.data().get(), n_basis_functions,
      &beta,
      psi.data().get(), lda);
}

template <>
double* Wavefunction<thrust::device_vector, thrust::device_allocator>::get_raw_pointer(thrust::device_vector<double, thrust::device_allocator<double>>& v) {
  return v.data().get();
}

template <>
const double* Wavefunction<thrust::device_vector, thrust::device_allocator>::get_raw_pointer(const thrust::device_vector<double, thrust::device_allocator<double>>& v) {
  return v.data().get();
}
