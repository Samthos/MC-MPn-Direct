#include <fstream>
#include "movec_parser.h"
#include "../qc_mpi.h"

void Movec_Parser::broadcast() {
  MPI_info::barrier();
  MPI_info::broadcast_int(&n_basis_functions, 1);
  MPI_info::broadcast_int(&n_molecular_orbitals, 1);
  MPI_info::broadcast_int(&n_occupied_orbitals, 1);
  resize();
  MPI_info::broadcast_vector_double(orbital_energies);
  MPI_info::broadcast_vector_double(movecs);
}

void Movec_Parser::resize() {
  occupancy.resize(n_basis_functions);
  orbital_energies.resize(n_basis_functions);
  movecs.resize(n_basis_functions * n_molecular_orbitals);
}

void Movec_Parser::log_orbital_energies(std::string jobname) {
  std::ofstream output(jobname + ".orbital_energies");

  // print out data
  output << "# File contains orbitral eigenvalue energies\n";
  output << "# Indexing is 0 based.\n";
  output << "# Last occupied orbitals is " << iocc1 - 1 << "\n";
  for (auto it = 0; it < ivir2; it++) {
    output << orbital_energies[it] << std::endl;
  }
}

void Movec_Parser::freeze_core(const Molecule& molecule) {
  iocc1 = 0;
  for (auto &atom : molecule.atoms) {
    if (atom.znum > 3 && atom.znum < 10 && !atom.is_ghost) {
      iocc1 += 1;
    }
  }
}
