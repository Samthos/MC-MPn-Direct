#include <memory>
#include "movec_parser.h"
#include "nwchem_movec_parser.h"
#include "dummy_movec_parser.h"

std::shared_ptr<Movec_Parser> create_movec_parser(IOPs& iops, MPI_info& mpi_info, Molec& molec, KEYS::KEYS source, MOVEC_TYPE::MOVEC_TYPE movec_type) {
  std::shared_ptr<Movec_Parser> movec_parser;
  switch (movec_type) {
    case MOVEC_TYPE::NWCHEM: movec_parser = std::shared_ptr<Movec_Parser>(new NWChem_Movec_Parser(iops, mpi_info, molec, source)); break;
    case MOVEC_TYPE::DUMMY:  movec_parser = std::shared_ptr<Movec_Parser>(new Dummy_Movec_Parser()); break;
  }
  return movec_parser;
}
