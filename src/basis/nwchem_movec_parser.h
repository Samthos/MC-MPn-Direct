#ifndef NWCHEM_MOVEC_PASER_H_
#define NWCHEM_MOVEC_PASER_H_

#include <vector>
#include "movec_parser.h"
#include "../qc_mpi.h"
#include "../qc_input.h"
#include "../qc_geom.h"

class NWChem_Movec_Parser : public Movec_Parser {
 public:
  NWChem_Movec_Parser(IOPs& iops, MPI_info& mpi_info, Molec& molec, KEYS::KEYS source=KEYS::MOVECS);

 private:
  int nw_nsets;

  void read(std::ifstream&, char*, bool);
  void parse_binary_movecs(std::string) override;
  void parse_ascii_movecs(std::string) override;
};

#endif  // NWCHEM_MOVEC_PASER_H_
