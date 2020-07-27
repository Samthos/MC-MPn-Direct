#ifndef QC_BASIS_H_
#define QC_BASIS_H_

#ifdef HAVE_CUDA
#include <thrust/device_vector.h>
#endif // HAVE_CUDA

#include "../qc_mpi.h"

#include "atomic_orbital.h"
#include "wavefunction.h"
#include "shell.h"
#include "atom_basis.h"
#include "basis_parser.h"

#include "cartesian_poly.h"

template <template <class, class> class Container, template <class> class Allocator>
class Basis {
  typedef Container<double, Allocator<double>> vector_double;
  typedef Container<Atomic_Orbital, Allocator<Atomic_Orbital>> vector_atomic_orbital;

 public:
  Basis(const int&, const Basis_Parser&);

  void build_contractions(const std::vector<std::array<double, 3>>&);
  void build_contractions_with_derivatives(const std::vector<std::array<double, 3>>&);

  void host_psi_get(Wavefunction&, std::vector<std::array<double, 3>>&);
  void host_psi_get_dx(Wavefunction&, std::vector<std::array<double, 3>>&);
  void host_psi_get_dy(Wavefunction&, std::vector<std::array<double, 3>>&);
  void host_psi_get_dz(Wavefunction&, std::vector<std::array<double, 3>>&);

  void dump(const std::string&);

  std::vector<double> get_contraction_amplitudes();
  std::vector<double> get_contraction_amplitudes_derivative();

  // basis set info
  int mc_num;
  int qc_nbf;       // number basis functions
  int nShells;      // number of shells
  int nPrimatives;  // number of primitives
  bool lspherical;  // true if spherical

  vector_double ao_amplitudes;                      // stores AO amplidutes

 private:
  void build_ao_amplitudes(const std::vector<std::array<double, 3>>&);
  void build_ao_amplitudes_dx(const std::vector<std::array<double, 3>>&);
  void build_ao_amplitudes_dy(const std::vector<std::array<double, 3>>&);
  void build_ao_amplitudes_dz(const std::vector<std::array<double, 3>>&);

  vector_double contraction_exp;
  vector_double contraction_coef;
  vector_double contraction_amplitudes;             // stores contraction amplitudes
  vector_double contraction_amplitudes_derivative;  // stores contraction amplitudes
  vector_atomic_orbital atomic_orbitals;
};

template <> std::vector<double> Basis<std::vector, std::allocator>::get_contraction_amplitudes();
template <> std::vector<double> Basis<std::vector, std::allocator>::get_contraction_amplitudes_derivative();
template <> void Basis<std::vector, std::allocator>::dump(const std::string& fname);
template class Basis<std::vector, std::allocator>;
typedef Basis<std::vector, std::allocator> Basis_Host;

#ifdef HAVE_CUDA
template <> void Basis<thrust::device_vector, thrust::device_allocator>::build_contractions(const std::vector<std::array<double, 3>>&); 
template <> void Basis<thrust::device_vector, thrust::device_allocator>::build_contractions_with_derivatives(const std::vector<std::array<double, 3>>&); 
template <> void Basis<thrust::device_vector, thrust::device_allocator>::host_psi_get(Wavefunction&, std::vector<std::array<double, 3>>&); 
template <> void Basis<thrust::device_vector, thrust::device_allocator>::host_psi_get_dx(Wavefunction&, std::vector<std::array<double, 3>>&); 
template <> void Basis<thrust::device_vector, thrust::device_allocator>::host_psi_get_dy(Wavefunction&, std::vector<std::array<double, 3>>&); 
template <> void Basis<thrust::device_vector, thrust::device_allocator>::host_psi_get_dz(Wavefunction&, std::vector<std::array<double, 3>>&); 
template <> void Basis<thrust::device_vector, thrust::device_allocator>::build_ao_amplitudes(const std::vector<std::array<double, 3>>&); 
template <> void Basis<thrust::device_vector, thrust::device_allocator>::build_ao_amplitudes_dx(const std::vector<std::array<double, 3>>&); 
template <> void Basis<thrust::device_vector, thrust::device_allocator>::build_ao_amplitudes_dy(const std::vector<std::array<double, 3>>&); 
template <> void Basis<thrust::device_vector, thrust::device_allocator>::build_ao_amplitudes_dz(const std::vector<std::array<double, 3>>&); 
template <> std::vector<double> Basis<thrust::device_vector, thrust::device_allocator>::get_contraction_amplitudes();
template <> std::vector<double> Basis<thrust::device_vector, thrust::device_allocator>::get_contraction_amplitudes_derivative();

template class Basis<thrust::device_vector, thrust::device_allocator>;
#endif
#endif  // QC_BASIS_H_
