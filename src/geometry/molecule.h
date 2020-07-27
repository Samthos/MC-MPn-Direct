#ifndef MOLECULE_H_
#define MOLECULE_H_

#include <string>
#include <vector>

#include "../qc_mpi.h"
#include "atom.h"

class Molecule {
  public:
    Molecule() = default;
    Molecule(const std::vector<Atom>& atoms);
    Molecule(const MPI_info&, const std::string&);
    void print();

    std::vector<Atom> atoms;

  protected:
    void read_xyz(const MPI_info&, const std::string&);
};
#endif  // MOLECULE_H_
