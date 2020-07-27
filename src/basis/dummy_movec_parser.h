#ifndef DUMMY_MOVEC_PASER_H_
#define DUMMY_MOVEC_PASER_H_

#include <string>
#include <vector>

#include "../qc_mpi.h"
#include "../qc_geom.h"
#include "../qc_input.h"
#include "molecule.h"

#include "movec_parser.h"

class Dummy_Movec_Parser : public Movec_Parser {
 public:
  Dummy_Movec_Parser();

 private:
  void parse_binary_movecs(std::string) override {}
  void parse_ascii_movecs(std::string) override {}
};

#endif  // DUMMY_MOVEC_PASER_H_
