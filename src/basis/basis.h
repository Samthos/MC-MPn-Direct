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

template <template <typename, typename> typename Container, template <typename> typename Allocator>
class Basis {
  typedef std::vector<Point> vector_point;

  typedef Container<double, Allocator<double>> vector_double;
  typedef Container<Atomic_Orbital, Allocator<Atomic_Orbital>> vector_atomic_orbital;
  typedef Wavefunction<Container, Allocator> Wavefunction_Type;

 public:
  Basis(const int&, const Basis_Parser&);

  void build_contractions(const vector_point&);
  void build_contractions_with_derivatives(const vector_point&);
  void build_ao_amplitudes(const vector_point&);
  void build_ao_amplitudes_dx(const vector_point&);
  void build_ao_amplitudes_dy(const vector_point&);
  void build_ao_amplitudes_dz(const vector_point&);

  void dump(const std::string&);

  std::vector<double> get_contraction_amplitudes();
  std::vector<double> get_contraction_amplitudes_derivative();
  std::vector<double> get_ao_amplitudes();

  // basis set info
  int mc_num;
  int qc_nbf;       // number basis functions
  int nShells;      // number of shells
  int nPrimatives;  // number of primitives
  bool lspherical;  // true if spherical

  vector_double ao_amplitudes;                      // stores AO amplidutes

 private:

  vector_double contraction_exp;
  vector_double contraction_coef;
  vector_double contraction_amplitudes;             // stores contraction amplitudes
  vector_double contraction_amplitudes_derivative;  // stores contraction amplitudes
  vector_atomic_orbital atomic_orbitals;
};

template <> std::vector<double> Basis<std::vector, std::allocator>::get_contraction_amplitudes();
template <> std::vector<double> Basis<std::vector, std::allocator>::get_contraction_amplitudes_derivative();
template <> std::vector<double> Basis<std::vector, std::allocator>::get_ao_amplitudes();
template <> void Basis<std::vector, std::allocator>::dump(const std::string& fname);
template class Basis<std::vector, std::allocator>;
typedef Basis<std::vector, std::allocator> Basis_Host;

#ifdef HAVE_CUDA
template <> void Basis<thrust::device_vector, thrust::device_allocator>::build_contractions(const vector_point&); 
template <> void Basis<thrust::device_vector, thrust::device_allocator>::build_contractions_with_derivatives(const vector_point&); 
template <> void Basis<thrust::device_vector, thrust::device_allocator>::build_ao_amplitudes(const vector_point&); 
template <> void Basis<thrust::device_vector, thrust::device_allocator>::build_ao_amplitudes_dx(const vector_point&); 
template <> void Basis<thrust::device_vector, thrust::device_allocator>::build_ao_amplitudes_dy(const vector_point&); 
template <> void Basis<thrust::device_vector, thrust::device_allocator>::build_ao_amplitudes_dz(const vector_point&); 
template <> std::vector<double> Basis<thrust::device_vector, thrust::device_allocator>::get_contraction_amplitudes();
template <> std::vector<double> Basis<thrust::device_vector, thrust::device_allocator>::get_contraction_amplitudes_derivative();
template <> std::vector<double> Basis<thrust::device_vector, thrust::device_allocator>::get_ao_amplitudes();

template class Basis<thrust::device_vector, thrust::device_allocator>;
typedef Basis<thrust::device_vector, thrust::device_allocator> Basis_Device;
#endif
#endif  // QC_BASIS_H_
