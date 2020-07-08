#ifndef BASIS_PARSER_H_
#define BASIS_PARSER_H_

#include "../qc_input.h"
#include "../qc_mpi.h"
#include "../qc_geom.h"

#include "atomic_orbital.h"
#include "shell.h"
#include "atom_basis.h"

class Basis_Parser {
 public:
  Basis_Parser(IOPs &, MPI_info &, Molec &);

  bool is_spherical;
  int n_atomic_orbitals;
  int n_shells;
  int n_primatives;

  std::vector<double> contraction_coeficients;
  std::vector<double> contraction_exponents;
  std::vector<BasisMetaData> atomic_orbitals;

 protected:
  void read(IOPs &, MPI_info &, Molec &);
  static void normalize_atom_basis(std::vector<AtomBasis>&);
  static void normalize_sp(SHELL::Shell& shell);
  static void normalize_s(SHELL::Shell& shell);
  static void normalize_p(SHELL::Shell& shell);
  static void normalize_d(SHELL::Shell& shell);
  static void normalize_f(SHELL::Shell& shell);
  static void normalize_g(SHELL::Shell& shell);
};
#endif  // BASIS_PARSER_H_
