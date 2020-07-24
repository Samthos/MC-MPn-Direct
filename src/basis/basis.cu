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
  int walker = threadIdx.x;
  int atomic_orbital = threadIdx.y;
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
//  dim3 block_size;
//  dim3 grid_size;
//  call_build_contraction<<<>>>();

// for (int walker = 0; walker < pos.size(); walker++) {
//   for (auto &atomic_orbital : atomic_orbitals) {
//     atomic_orbital.evaluate_contraction(
//         contraction_amplitudes.data() + atomic_orbitals.size() * walker,
//         contraction_exp.data(),
//         contraction_coef.data(),
//         pos[walker].data());
//   }
// }
}

template <>
void Basis<thrust::device_vector, thrust::device_allocator>::build_contractions_with_derivatives(const std::vector<std::array<double, 3>>&) {}

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
