#ifndef MOVEC_PARSER_H_
#define MOVEC_PARSER_H_

#include <memory>
#include <string>
#include <vector>

#include "../qc_mpi.h"
#include "molecule.h"
#include "movec_type.h"

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
  void freeze_core(const Molecule&);

  virtual void parse_binary_movecs(std::string) = 0;
  virtual void parse_ascii_movecs(std::string) = 0;
};

std::shared_ptr<Movec_Parser> create_movec_parser(MPI_info& mpi_info, Molecule& molec, const MOVEC_TYPE::MOVEC_TYPE&, const std::string& movecs_filename, bool set_frozen_core);
#endif  // MOVEC_PARSER_H_
