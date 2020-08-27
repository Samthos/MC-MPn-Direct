#include <numeric>
#include "dummy_movec_parser.h"

Dummy_Movec_Parser::Dummy_Movec_Parser() { 
  iocc1 = 1;
  iocc2 = 6;
  ivir1 = 6;
  ivir2 = 16;
  n_basis_functions = 20;
  n_molecular_orbitals = 16;
  n_core_orbitals = 1;
  n_occupied_orbitals = 6;

  occupancy.resize(n_molecular_orbitals);
  std::fill(occupancy.begin(), occupancy.end(), 0.0);
  std::fill(occupancy.begin(), occupancy.begin() + iocc2, 2.0);
  
  orbital_energies.resize(n_molecular_orbitals);
  std::iota(orbital_energies.begin(), orbital_energies.end(), -5.5);

  movecs.resize(n_molecular_orbitals * n_basis_functions);
  for (int j = 0; j < n_molecular_orbitals; j++) {
    for (int i = 0; i < n_basis_functions; i++) {
      movecs[j * n_basis_functions + i] = 1.0 / ((i+1) * (j+1));
    }
  }
}
