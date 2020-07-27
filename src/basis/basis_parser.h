#ifndef BASIS_PARSER_H_
#define BASIS_PARSER_H_

#include "../qc_input.h"
#include "../qc_mpi.h"
#include "molecule.h"

#include "atomic_orbital.h"
#include "shell.h"
#include "atom_basis.h"

class Basis_Parser {
 public:
  Basis_Parser(IOPs &, MPI_info &, Molecule &);

  bool is_spherical;
  int n_atomic_orbitals;
  int n_shells;
  int n_primatives;

  std::vector<double> contraction_coeficients;
  std::vector<double> contraction_exponents;
  std::vector<Atomic_Orbital> atomic_orbitals;

 protected:
  void read(IOPs &, MPI_info &, Molecule &);
};
#endif  // BASIS_PARSER_H_
