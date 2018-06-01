// Copyright 2017

#include <vector>
#include <string>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "qc_mpi.h"
#include "qc_geom.h"

#ifndef MC_BASIS_H_
#define MC_BASIS_H_
struct mc_basis_typ {
  int znum;
  double alpha[10], norm[10];
};

class MC_Basis{
 public:
  // variables
  int mc_nbas, mc_nprim;
  int natom;
  double g_wgt;

  // vectors
  std::vector<int> atom_ibas;
  std::vector<mc_basis_typ> mc_basis_list;

  // functions
  void read(MPI_info&, Molec&, std::string&);
  void mc_eri2v(MPI_info&, Molec&);
  int atom_to_mc_basis(int, Molec&);
};
#endif  // MC_BAIS_H_
