#include "basis.cpp"

__global__ 
void call_build_contraction(
    int n_walkers,
    int n_atomic_orbitals,
    Atomic_Orbital* atomic_orbitals,
    double* contraction_amplitudes,
    double* contraction_exp,
    double* contraction_coef,
    double* pos) {
  int walker = blockIdx.x * blockDim.x + threadIdx.x;
  int atomic_orbital = blockIdx.y * blockDim.y + threadIdx.y;

  if (walker < n_walkers && atomic_orbital < n_atomic_orbitals) {
    atomic_orbitals[atomic_orbital].evaluate_contraction(
        contraction_amplitudes + n_atomic_orbitals * walker,
        contraction_exp,
        contraction_coef,
        pos + 3 * walker);
  }
}

template <>
void Basis<thrust::device_vector, thrust::device_allocator>::build_contractions(const std::vector<std::array<double, 3>> &pos) {
  dim3 block_size(128, 1, 1);
  dim3 grid_size((pos.size() + 127) / 128, atomic_orbitals.size(), 1);

  std::vector<double> h_p(pos.size() * 3);
  for (int i = 0, idx = 0; i < pos.size(); i++) {
    for (int j = 0; j < 3; j++, idx++) {
      h_p[idx] = pos[i][j];
    }
  }
  thrust::device_vector<double> d_p(h_p);

  call_build_contraction<<<grid_size, block_size>>>(
      pos.size(),
      atomic_orbitals.size(),
      atomic_orbitals.data().get(),
      contraction_amplitudes.data().get(),
      contraction_exp.data().get(),
      contraction_coef.data().get(),
      d_p.data().get());
}

__global__ 
void call_build_contraction_with_derivative(
    int n_walkers,
    int n_atomic_orbitals,
    Atomic_Orbital* atomic_orbitals,
    double* contraction_amplitudes,
    double* contraction_amplitudes_derivative,
    double* contraction_exp,
    double* contraction_coef,
    double* pos) {
  int walker = blockIdx.x * blockDim.x + threadIdx.x;
  int atomic_orbital = blockIdx.y * blockDim.y + threadIdx.y;

  if (walker < n_walkers && atomic_orbital < n_atomic_orbitals) {
    atomic_orbitals[atomic_orbital].evaluate_contraction_with_derivative(
        contraction_amplitudes + n_atomic_orbitals * walker,
        contraction_amplitudes_derivative + n_atomic_orbitals * walker,
        contraction_exp,
        contraction_coef,
        pos + 3 * walker);
  }
}

template <>
void Basis<thrust::device_vector, thrust::device_allocator>::build_contractions_with_derivatives(const std::vector<std::array<double, 3>>& pos) {
  dim3 block_size(128, 1, 1);
  dim3 grid_size((pos.size() + 127) / 128, atomic_orbitals.size(), 1);

  std::vector<double> h_p(pos.size() * 3);
  for (int i = 0, idx = 0; i < pos.size(); i++) {
    for (int j = 0; j < 3; j++, idx++) {
      h_p[idx] = pos[i][j];
    }
  }
  thrust::device_vector<double> d_p(h_p);

  call_build_contraction_with_derivative<<<grid_size, block_size>>>(
      pos.size(),
      atomic_orbitals.size(),
      atomic_orbitals.data().get(),
      contraction_amplitudes.data().get(),
      contraction_amplitudes_derivative.data().get(),
      contraction_exp.data().get(),
      contraction_coef.data().get(),
      d_p.data().get());
}

template <>
void Basis<thrust::device_vector, thrust::device_allocator>::host_psi_get(Wavefunction&, std::vector<std::array<double, 3>>&) {}

template <>
void Basis<thrust::device_vector, thrust::device_allocator>::host_psi_get_dx(Wavefunction&, std::vector<std::array<double, 3>>&) {}

template <>
void Basis<thrust::device_vector, thrust::device_allocator>::host_psi_get_dy(Wavefunction&, std::vector<std::array<double, 3>>&) {}

template <>
void Basis<thrust::device_vector, thrust::device_allocator>::host_psi_get_dz(Wavefunction&, std::vector<std::array<double, 3>>&) {}

template <>
void Basis<thrust::device_vector, thrust::device_allocator>::build_ao_amplitudes(const std::vector<std::array<double, 3>>&) {}

template <>
void Basis<thrust::device_vector, thrust::device_allocator>::build_ao_amplitudes_dx(const std::vector<std::array<double, 3>>&) {}

template <>
void Basis<thrust::device_vector, thrust::device_allocator>::build_ao_amplitudes_dy(const std::vector<std::array<double, 3>>&) {}

template <>
void Basis<thrust::device_vector, thrust::device_allocator>::build_ao_amplitudes_dz(const std::vector<std::array<double, 3>>&) {}

template <>
std::vector<double> Basis<thrust::device_vector, thrust::device_allocator>::get_contraction_amplitudes(){
  std::vector<double> v(contraction_amplitudes.size());
  thrust::copy(contraction_amplitudes.begin(), contraction_amplitudes.end(), v.begin());
  return v;
}

template <>
std::vector<double> Basis<thrust::device_vector, thrust::device_allocator>::get_contraction_amplitudes_derivative(){
  std::vector<double> v(contraction_amplitudes_derivative.size());
  thrust::copy(contraction_amplitudes_derivative.begin(), contraction_amplitudes_derivative.end(), v.begin());
  return v;
}

