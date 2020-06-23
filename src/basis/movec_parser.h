#ifndef MOVEC_PASER_H_
#define MOVEC_PASER_H_

#include <memory>
#include <string>
#include <vector>
#include "../qc_mpi.h"
#include "../qc_input.h"
#include "../qc_geom.h"

class Movec_Parser {
 public:
  int iocc1;
  int iocc2;
  int ivir1;
  int ivir2;

  int n_basis_functions;
  int n_molecular_orbitals;
  int n_core_orbitals;
  int n_occupied_orbitals;

  std::vector<double> occupancy;
  std::vector<double> orbital_energies;
  std::vector<double> movecs;

 protected:
  void resize();
  void broadcast();
  void log_orbital_energies(std::string);

  virtual void parse_binary_movecs(std::string) = 0;
  virtual void parse_ascii_movecs(std::string) = 0;
};

namespace MOVEC_TYPE {
  enum MOVEC_TYPE {
    NWCHEM = 0,
    DUMMY,
  };
  const std::vector<std::string> movec_type_strings = {
    "NWCHEM",
    "DUMMY"
  };
}

std::shared_ptr<Movec_Parser> create_movec_parser(IOPs& iops, MPI_info& mpi_info, Molec& molec, KEYS::KEYS source=KEYS::MOVECS, MOVEC_TYPE::MOVEC_TYPE movec_type = MOVEC_TYPE::NWCHEM);
#endif  // MOVEC_PASER_H_
