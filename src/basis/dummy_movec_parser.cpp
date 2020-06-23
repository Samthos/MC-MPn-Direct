#include <numeric>
#include "dummy_movec_parser.h"

Dummy_Movec_Parser::Dummy_Movec_Parser() { 
  iocc1 = 1;
  iocc2 = 6;
  ivir1 = 6;
  ivir2 = 16;
  n_basis_functions = 16;
  n_molecular_orbitals = 16;
  n_core_orbitals = 1;
  n_occupied_orbitals = 6;

  occupancy.resize(16);
  std::fill(occupancy.begin(), occupancy.end(), 0.0);
  std::fill(occupancy.begin(), occupancy.begin() + iocc2, 2.0);
  
  orbital_energies.resize(16);
  std::iota(orbital_energies.begin(), orbital_energies.end(), -5.5);

  movecs.resize(16 * 16);
  std::iota(movecs.begin(), movecs.end(), 0.0);
}
