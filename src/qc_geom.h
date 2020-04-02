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
  std::string tag;
};

class Molec {
 public:
  void read(const MPI_info&, const std::string&);
  void print();

  std::vector<Atom> atoms;
};
#endif  // QC_GEOM_H_
