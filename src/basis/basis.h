#ifndef QC_BASIS_H_
#define QC_BASIS_H_

#include <iostream>
#include <cmath>
#include "../qc_geom.h"
#include "../qc_input.h"
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
  Basis(IOPs&, const Basis_Parser&);

  // get psi vals
  void host_psi_get(Wavefunction&, std::vector<std::array<double, 3>>&);
  void host_psi_get_dx(Wavefunction&, std::vector<std::array<double, 3>>&);
  void host_psi_get_dy(Wavefunction&, std::vector<std::array<double, 3>>&);
  void host_psi_get_dz(Wavefunction&, std::vector<std::array<double, 3>>&);

  void build_contractions(const std::vector<std::array<double, 3>>&);
  void build_contractions_with_derivatives(const std::vector<std::array<double, 3>>&);

  // basis set info
  int mc_num;
  int qc_nbf;       // number basis functions
  int nShells;      // number of shells
  int nPrimatives;  // number of primitives
  bool lspherical;  // true if spherical

  vector_double contraction_exp;
  vector_double contraction_coef;
  vector_double contraction_amplitudes;             // stores contraction amplitudes
  vector_double contraction_amplitudes_derivative;  // stores contraction amplitudes
  vector_double ao_amplitudes;                      // stores AO amplidutes

  vector_atomic_orbital atomic_orbitals;

 private:
  void build_ao_amplitudes(const std::vector<std::array<double, 3>>&);
  void build_ao_amplitudes_dx(const std::vector<std::array<double, 3>>&);
  void build_ao_amplitudes_dy(const std::vector<std::array<double, 3>>&);
  void build_ao_amplitudes_dz(const std::vector<std::array<double, 3>>&);

  void dump(const std::string&);
};

template class Basis<std::vector, std::allocator>;
typedef Basis<std::vector, std::allocator> Basis_Host;
#endif  // QC_BASIS_H_
