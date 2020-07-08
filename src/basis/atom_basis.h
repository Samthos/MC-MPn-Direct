#ifndef ATOM_BASIS_H_
#define ATOM_BASIS_H_
#include "shell.h"
struct AtomBasis {
  AtomBasis() { atomCharge = -1; }
  std::string basisName;
  std::string basisType;
  std::string atomName;
  std::vector<SHELL::Shell> shell;
  int atomCharge;
};
#endif // ATOM_BASIS_H_
