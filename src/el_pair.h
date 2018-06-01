// Copyright 2017 Hirata Lab

#include <array>
#include <vector>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "qc_geom.h"
#include "mc_basis.h"
#include "qc_random.h"

#ifndef EL_PAIR_H_
#define EL_PAIR_H_
class el_pair_typ {
private:
public:
  void init(int);
  void pos_init(Molec&, Random&);
  void weight_func_set(Molec&, MC_Basis&);
  void mc_move_scheme(int*, int*, double, Random&, Molec&, MC_Basis&);

  std::array<double,3> pos1, pos2;
  std::vector<double> psi1, psi2;
  double wgt, r12, rv;
  bool is_new;
};
#endif  // EL_PAIR_H_
