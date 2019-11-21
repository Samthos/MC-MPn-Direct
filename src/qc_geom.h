// Copyright 2019
#ifndef QC_GEOM_H_
#define QC_GEOM_H_

#include <array>
#include <string>
#include <vector>

#include "qc_mpi.h"

struct Atom {
  double pos[3];
  int znum;
};

class Molec {
 public:
  void read(MPI_info&, std::string&);
  void print(int);

  std::vector<Atom> atom;
  int natom;
};
#endif  // QC_GEOM_H_
