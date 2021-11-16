#include <fstream>
#include <string>

#include "../qc_mpi.h"
#include "basis.h"


template <template <typename, typename> typename Container, template <typename> typename Allocator>
Basis<Container, Allocator>::Basis(const int& mc_num_, const Basis_Parser& basis_parser) :
  mc_num(mc_num_),
  qc_nbf(basis_parser.n_atomic_orbitals),
  nShells(basis_parser.n_shells),
  nPrimatives(basis_parser.n_primatives),
  lspherical(basis_parser.is_spherical),
  contraction_exp(basis_parser.contraction_exponents),
  contraction_coef(basis_parser.contraction_coeficients),
  contraction_amplitudes(nShells * mc_num),
  contraction_amplitudes_derivative(nShells * mc_num),
  ao_amplitudes(qc_nbf * mc_num),
  atomic_orbitals(basis_parser.atomic_orbitals)
{ }

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Basis<Container, Allocator>::build_contractions(const vector_Point &pos) {
  for (int walker = 0; walker < pos.size(); walker++) {
    for (auto &atomic_orbital : atomic_orbitals) {
      atomic_orbital.evaluate_contraction(
          contraction_amplitudes.data() + atomic_orbitals.size() * walker,
          contraction_exp.data(),
          contraction_coef.data(),
          pos[walker]);
    }
  }
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Basis<Container, Allocator>::build_contractions_with_derivatives(const vector_Point& pos) {
  for (int walker = 0; walker < pos.size(); walker++) {
    for (auto &atomic_orbital : atomic_orbitals) {
      atomic_orbital.evaluate_contraction_with_derivative(
          contraction_amplitudes.data() + atomic_orbitals.size() * walker,
          contraction_amplitudes_derivative.data() + atomic_orbitals.size() * walker,
          contraction_exp.data(),
          contraction_coef.data(),
          pos[walker]);
    }
  }
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Basis<Container, Allocator>::build_ao_amplitudes(const vector_Point &pos) {
  for (int walker = 0; walker < pos.size(); walker++) {
    for (int shell = 0; shell < nShells; shell++) {
      atomic_orbitals[shell].evaluate_ao(
          ao_amplitudes.data() + walker * qc_nbf,
          contraction_amplitudes.data() + walker * nShells,
          pos[walker]);
    }
  }
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Basis<Container, Allocator>::build_ao_amplitudes_dx(const vector_Point& pos){
  for (int walker = 0, index = 0; walker < pos.size(); walker++) {
    for (int shell = 0; shell < nShells; shell++, index++) {
      atomic_orbitals[shell].evaluate_ao_dx(
          ao_amplitudes.data() + walker * qc_nbf,
          contraction_amplitudes.data() + walker * nShells,
          contraction_amplitudes_derivative.data() + walker * nShells,
          pos[walker]);
    }
  }
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Basis<Container, Allocator>::build_ao_amplitudes_dy(const vector_Point& pos){
  for (int walker = 0, index = 0; walker < pos.size(); walker++) {
    for (int shell = 0; shell < nShells; shell++, index++) {
      atomic_orbitals[shell].evaluate_ao_dy(
          ao_amplitudes.data() + walker * qc_nbf,
          contraction_amplitudes.data() + walker * nShells,
          contraction_amplitudes_derivative.data() + walker * nShells,
          pos[walker]);
    }
  }
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Basis<Container, Allocator>::build_ao_amplitudes_dz(const vector_Point& pos){
  for (int walker = 0, index = 0; walker < pos.size(); walker++) {
    for (int shell = 0; shell < nShells; shell++, index++) {
      atomic_orbitals[shell].evaluate_ao_dz(
          ao_amplitudes.data() + walker * qc_nbf,
          contraction_amplitudes.data() + walker * nShells,
          contraction_amplitudes_derivative.data() + walker * nShells,
          pos[walker]);
    }
  }
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
std::vector<double> Basis<Container, Allocator>::get_contraction_amplitudes(){
  throw std::exception();
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void Basis<Container, Allocator>::dump(const std::string& fname) {
  std::ofstream os(fname);
  os << "\n-----------------------------------------------------------------------------------------------------------\nBasis Dump\n";
  os << "qc_nbf: " << qc_nbf << "\n";       // number basis functions
  os << "nShells: " << nShells << "\n";      // number of shells
  os << "nPrimatives: " << nPrimatives << "\n";  // number of primitives
  os << "lspherical: " << lspherical << "\n";  // true if spherical
// for (int i = 0; i < nPrimatives; ++i) {
//   os << contraction_coef[i] << "\t" << contraction_exp[i] << "\n";
// }
// for (int i = 0; i < nShells; ++i) {
//   os << atomic_orbitals[i].ao_index << "\t";
//   os << atomic_orbitals[i].contraction_begin << "\t";
//   os << atomic_orbitals[i].contraction_end << "\t";
//   os << atomic_orbitals[i].angular_momentum << "\t";
//   os << atomic_orbitals[i].pos[0] << "\t";
//   os << atomic_orbitals[i].pos[1] << "\t";
//   os << atomic_orbitals[i].pos[2] << "\n";
// }
  os << "-----------------------------------------------------------------------------------------------------------\n\n";
}

template <>
std::vector<double> Basis<std::vector, std::allocator>::get_contraction_amplitudes() {
  return contraction_amplitudes;
}

template <>
std::vector<double> Basis<std::vector, std::allocator>::get_contraction_amplitudes_derivative() {
  return contraction_amplitudes_derivative;
}

template <>
std::vector<double> Basis<std::vector, std::allocator>::get_ao_amplitudes() {
  return ao_amplitudes;
}

template <>
void Basis<std::vector, std::allocator>::dump(const std::string& fname) {
  std::ofstream os(fname);
  os << "\n-----------------------------------------------------------------------------------------------------------\nBasis Dump\n";
  os << "qc_nbf: " << qc_nbf << "\n";       // number basis functions
  os << "nShells: " << nShells << "\n";      // number of shells
  os << "nPrimatives: " << nPrimatives << "\n";  // number of primitives
  os << "lspherical: " << lspherical << "\n";  // true if spherical
  for (auto &atomic_orbital : atomic_orbitals) {
    for (auto i = atomic_orbital.contraction_begin; i < atomic_orbital.contraction_end; i++) {
      printf("%16.8f ", contraction_coef[i]);
    }
    printf("\n");
  }
// for (int i = 0; i < nShells; ++i) {
//   os << atomic_orbitals[i].ao_index << "\t";
//   os << atomic_orbitals[i].contraction_begin << "\t";
//   os << atomic_orbitals[i].contraction_end << "\t";
//   os << atomic_orbitals[i].angular_momentum << "\t";
//   os << atomic_orbitals[i].pos[0] << "\t";
//   os << atomic_orbitals[i].pos[1] << "\t";
//   os << atomic_orbitals[i].pos[2] << "\n";
// }
  os << "-----------------------------------------------------------------------------------------------------------\n\n";
}

#ifdef HAVE_CUDA
thrust::device_vector<double> make_flat_pos(const std::vector<Point>& pos) {
  std::vector<double> h_p(pos.size() * 3);
  for (int i = 0, idx = 0; i < pos.size(); i++) {
    for (int j = 0; j < 3; j++, idx++) {
      h_p[idx] = pos[i][j];
    }
  }
  thrust::device_vector<double> d_p(h_p);
  return d_p;
}

__global__ 
void call_build_contraction(
    int n_walkers,
    int n_atomic_orbitals,
    Atomic_Orbital* atomic_orbitals,
    double* contraction_amplitudes,
    double* contraction_exp,
    double* contraction_coef,
    const Point* pos) {
  int walker = blockIdx.x * blockDim.x + threadIdx.x;
  int atomic_orbital = blockIdx.y * blockDim.y + threadIdx.y;

  if (walker < n_walkers && atomic_orbital < n_atomic_orbitals) {
    atomic_orbitals[atomic_orbital].evaluate_contraction(
        contraction_amplitudes + n_atomic_orbitals * walker,
        contraction_exp,
        contraction_coef,
        *(pos + walker));
  }
}

template <>
void Basis<thrust::device_vector, thrust::device_allocator>::build_contractions(const vector_Point &pos) {
  dim3 block_size(128, 1, 1);
  dim3 grid_size((pos.size() + 127) / 128, atomic_orbitals.size(), 1);

  call_build_contraction<<<grid_size, block_size>>>(
      pos.size(),
      atomic_orbitals.size(),
      atomic_orbitals.data().get(),
      contraction_amplitudes.data().get(),
      contraction_exp.data().get(),
      contraction_coef.data().get(),
      pos.data().get());
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
    const Point* pos) {
  int walker = blockIdx.x * blockDim.x + threadIdx.x;
  int atomic_orbital = blockIdx.y * blockDim.y + threadIdx.y;

  if (walker < n_walkers && atomic_orbital < n_atomic_orbitals) {
    atomic_orbitals[atomic_orbital].evaluate_contraction_with_derivative(
        contraction_amplitudes + n_atomic_orbitals * walker,
        contraction_amplitudes_derivative + n_atomic_orbitals * walker,
        contraction_exp,
        contraction_coef,
        *(pos + walker));
  }
}

template <>
void Basis<thrust::device_vector, thrust::device_allocator>::build_contractions_with_derivatives(const vector_Point& pos) {
  dim3 block_size(128, 1, 1);
  dim3 grid_size((pos.size() + 127) / 128, atomic_orbitals.size(), 1);

  call_build_contraction_with_derivative<<<grid_size, block_size>>>(
      pos.size(),
      atomic_orbitals.size(),
      atomic_orbitals.data().get(),
      contraction_amplitudes.data().get(),
      contraction_amplitudes_derivative.data().get(),
      contraction_exp.data().get(),
      contraction_coef.data().get(),
      pos.data().get());
}

__global__ 
void call_build_ao_amplitudes(
    int n_walkers,
    int n_atomic_orbitals,
    int n_basis_functions,
    Atomic_Orbital* atomic_orbitals,
    double* ao_amplitudes,
    double* contraction_amplitudes,
    const Point* pos) {
  int walker = blockIdx.x * blockDim.x + threadIdx.x;
  int atomic_orbital = blockIdx.y * blockDim.y + threadIdx.y;

  if (walker < n_walkers && atomic_orbital < n_atomic_orbitals) {
      atomic_orbitals[atomic_orbital].evaluate_ao(
          ao_amplitudes + walker * n_basis_functions,
          contraction_amplitudes + walker * n_atomic_orbitals,
          *(pos + walker));
  }
}

template <>
void Basis<thrust::device_vector, thrust::device_allocator>::build_ao_amplitudes(const vector_Point& pos) {
  dim3 block_size(128, 1, 1);
  dim3 grid_size((pos.size() + 127) / 128, atomic_orbitals.size(), 1);

  call_build_ao_amplitudes<<<grid_size, block_size>>>(
      pos.size(),
      atomic_orbitals.size(),
      qc_nbf,
      atomic_orbitals.data().get(),
      ao_amplitudes.data().get(),
      contraction_amplitudes.data().get(),
      pos.data().get());
}

__global__ 
void call_build_ao_amplitudes_dx(
    int n_walkers,
    int n_atomic_orbitals,
    int n_basis_functions,
    Atomic_Orbital* atomic_orbitals,
    double* ao_amplitudes,
    double* contraction_amplitudes,
    double* contraction_amplitudes_derivative,
    const Point* pos) {
  int walker = blockIdx.x * blockDim.x + threadIdx.x;
  int atomic_orbital = blockIdx.y * blockDim.y + threadIdx.y;

  if (walker < n_walkers && atomic_orbital < n_atomic_orbitals) {
      atomic_orbitals[atomic_orbital].evaluate_ao_dx(
          ao_amplitudes + walker * n_basis_functions,
          contraction_amplitudes + walker * n_atomic_orbitals,
          contraction_amplitudes_derivative + walker * n_atomic_orbitals,
          *(pos + walker));
  }
}

template <>
void Basis<thrust::device_vector, thrust::device_allocator>::build_ao_amplitudes_dx(const vector_Point &pos) {
  dim3 block_size(128, 1, 1);
  dim3 grid_size((pos.size() + 127) / 128, atomic_orbitals.size(), 1);

  call_build_ao_amplitudes_dx<<<grid_size, block_size>>>(
      pos.size(),
      atomic_orbitals.size(),
      qc_nbf,
      atomic_orbitals.data().get(),
      ao_amplitudes.data().get(),
      contraction_amplitudes.data().get(),
      contraction_amplitudes_derivative.data().get(),
      pos.data().get());
}

__global__ 
void call_build_ao_amplitudes_dy(
    int n_walkers,
    int n_atomic_orbitals,
    int n_basis_functions,
    Atomic_Orbital* atomic_orbitals,
    double* ao_amplitudes,
    double* contraction_amplitudes,
    double* contraction_amplitudes_derivative,
    const Point* pos) {
  int walker = blockIdx.x * blockDim.x + threadIdx.x;
  int atomic_orbital = blockIdx.y * blockDim.y + threadIdx.y;

  if (walker < n_walkers && atomic_orbital < n_atomic_orbitals) {
      atomic_orbitals[atomic_orbital].evaluate_ao_dy(
          ao_amplitudes + walker * n_basis_functions,
          contraction_amplitudes + walker * n_atomic_orbitals,
          contraction_amplitudes_derivative + walker * n_atomic_orbitals,
          *(pos + walker));
  }
}

template <>
void Basis<thrust::device_vector, thrust::device_allocator>::build_ao_amplitudes_dy(const vector_Point &pos) {
  dim3 block_size(128, 1, 1);
  dim3 grid_size((pos.size() + 127) / 128, atomic_orbitals.size(), 1);

  call_build_ao_amplitudes_dy<<<grid_size, block_size>>>(
      pos.size(),
      atomic_orbitals.size(),
      qc_nbf,
      atomic_orbitals.data().get(),
      ao_amplitudes.data().get(),
      contraction_amplitudes.data().get(),
      contraction_amplitudes_derivative.data().get(),
      pos.data().get());
}

__global__ 
void call_build_ao_amplitudes_dz(
    int n_walkers,
    int n_atomic_orbitals,
    int n_basis_functions,
    Atomic_Orbital* atomic_orbitals,
    double* ao_amplitudes,
    double* contraction_amplitudes,
    double* contraction_amplitudes_derivative,
    const Point* pos) {
  int walker = blockIdx.x * blockDim.x + threadIdx.x;
  int atomic_orbital = blockIdx.y * blockDim.y + threadIdx.y;

  if (walker < n_walkers && atomic_orbital < n_atomic_orbitals) {
      atomic_orbitals[atomic_orbital].evaluate_ao_dz(
          ao_amplitudes + walker * n_basis_functions,
          contraction_amplitudes + walker * n_atomic_orbitals,
          contraction_amplitudes_derivative + walker * n_atomic_orbitals,
          *(pos + walker));
  }
}

template <>
void Basis<thrust::device_vector, thrust::device_allocator>::build_ao_amplitudes_dz(const vector_Point &pos) {
  dim3 block_size(128, 1, 1);
  dim3 grid_size((pos.size() + 127) / 128, atomic_orbitals.size(), 1);

  call_build_ao_amplitudes_dz<<<grid_size, block_size>>>(
      pos.size(),
      atomic_orbitals.size(),
      qc_nbf,
      atomic_orbitals.data().get(),
      ao_amplitudes.data().get(),
      contraction_amplitudes.data().get(),
      contraction_amplitudes_derivative.data().get(),
      pos.data().get());
}

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

template <>
std::vector<double> Basis<thrust::device_vector, thrust::device_allocator>::get_ao_amplitudes(){
  std::vector<double> v(ao_amplitudes.size());
  thrust::copy(ao_amplitudes.begin(), ao_amplitudes.end(), v.begin());
  return v;
}
#endif
