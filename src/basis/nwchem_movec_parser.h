#ifndef NWCHEM_MOVEC_PASER_H_
#define NWCHEM_MOVEC_PASER_H_

#include <vector>

#include "../qc_mpi.h"
#include "molecule.h"

#include "movec_parser.h"

class NWChem_Movec_Parser : public Movec_Parser {
 public:
  NWChem_Movec_Parser(MPI_info& mpi_info, Molecule& molec, const std::string&, bool);

 private:
  int nw_nsets;

  void read(std::ifstream&, char*, bool);
  void parse_binary_movecs(std::string) override;
  void parse_ascii_movecs(std::string) override;
};

#endif  // NWCHEM_MOVEC_PASER_H_
