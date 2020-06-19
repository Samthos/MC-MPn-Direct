#ifndef NW_VECTOR_H_
#define NW_VECTOR_H_

#include <vector>
#include "../qc_mpi.h"
#include "../qc_input.h"
#include "../qc_geom.h"

class NWChem_Movec_Parser {
 public:
  NWChem_Movec_Parser(IOPs& iops, MPI_info& mpi_info, Molec& molec, KEYS::KEYS source=KEYS::MOVECS);

  int iocc1, iocc2, ivir1, ivir2;

  int n_basis_functions;
  int n_molecular_orbitals;
  int n_core_orbitals;
  int n_occupied_orbitals;

  int nw_nsets;

  std::vector<double> occupancy;
  std::vector<double> orbital_energies;
  std::vector<double> movecs;

 private:
  void resize();
  void broadcast();
  void read(std::ifstream&, char*, bool);


  void log_orbital_energies(std::string);
  void parse_binary_movecs(std::string);
  void parse_ascii_movecs(std::string);
};

#endif  // NW_VECTOR_H_
