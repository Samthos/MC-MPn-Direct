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

template <class Container>
class Basis {
  /*
   * Engine to compute molecular orbitals assuming
   * WARNING: Be extremely careful with this class. It manages memory explicitly.
   *
   * TODO
   *  -split of implement cartesian orbitals
   *  -specialize for GPU/CPU usage
   */
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

  Container contraction_exp;                    // dense vector of contraction exponents. Size if total number of primatives
  Container contraction_coef;                   // dense vector of contraction coeficients. Size if total number of primatives
  std::vector<Atomic_Orbital> atomic_orbitals;

  Container contraction_amplitudes;             // stores contraction amplitudes
  Container contraction_amplitudes_derivative;  // stores contraction amplitudes
  Container ao_amplitudes;                      // stores AO amplidutes

 private:
  void build_ao_amplitudes(const std::vector<std::array<double, 3>>&);
  void build_ao_amplitudes_dx(const std::vector<std::array<double, 3>>&);
  void build_ao_amplitudes_dy(const std::vector<std::array<double, 3>>&);
  void build_ao_amplitudes_dz(const std::vector<std::array<double, 3>>&);

  void dump(const std::string&);
};

template class Basis<std::vector<double>>;
#endif  // QC_BASIS_H_
