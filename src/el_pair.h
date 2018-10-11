// Copyright 2017 Hirata Lab

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <array>
#include <vector>

#include "weight_function.h"
#include "qc_geom.h"
#include "qc_random.h"

#ifndef EL_PAIR_H_
#define EL_PAIR_H_
class el_pair_typ {
 private:
  double r12();
  double calculate_r(double p, double alpha, double beta, double a);
  double calculate_phi(double p, double r, double alpha, double beta, double a);

 public:
  void init(const int);
  void mc_move_scheme(Random&, const Molec&, const GTO_Weight&);

  std::array<double, 3> pos1, pos2;
  std::vector<double> psi1, psi2;
  double wgt, rv;
};
#endif  // EL_PAIR_H_
