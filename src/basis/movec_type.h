#ifndef MOVEC_TYPE_H_
#define MOVEC_TYPE_H_

#include <string>
#include <vector>

namespace MOVEC_TYPE {
  enum MOVEC_TYPE {
    NWCHEM_BINARY = 0,
    NWCHEM_ASCII,
    DUMMY,
  };
  const std::vector<std::string> movec_type_strings = {
    "NWCHEM_BINARY",
    "NWCHEM_ASCII",
    "DUMMY"
  };
}
#endif  // MOVEC_TYPE_H_
