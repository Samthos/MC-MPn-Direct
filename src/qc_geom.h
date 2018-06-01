// Copyright 2017

#include <array>
#include <string>
#include <vector>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "qc_mpi.h"

#ifndef QC_GEOM_H_
#define QC_GEOM_H_
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
