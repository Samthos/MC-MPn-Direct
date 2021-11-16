#include <memory>
#include "movec_parser.h"
#include "nwchem_movec_parser.h"
#include "dummy_movec_parser.h"

std::shared_ptr<Movec_Parser> create_movec_parser(MPI_info& mpi_info, Molecule& molec, const MOVEC_TYPE::MOVEC_TYPE& movec_type, const std::string& movecs_filename, bool set_frozen_core) {
  std::shared_ptr<Movec_Parser> movec_parser;
  switch (movec_type) {
    case MOVEC_TYPE::NWCHEM_BINARY: movec_parser = std::shared_ptr<Movec_Parser>(new NWChem_Movec_Parser(mpi_info, molec, movecs_filename, set_frozen_core)); break;
    case MOVEC_TYPE::DUMMY: movec_parser = std::shared_ptr<Movec_Parser>(new Dummy_Movec_Parser()); break;
    case MOVEC_TYPE::NWCHEM_ASCII: // movec_parser = std::shared_ptr<Movec_Parser>(new NWChem_Movec_Parser(mpi_info, molec, movecs_filename)); break;
    default: throw std::exception();
  }
  return movec_parser;
}
