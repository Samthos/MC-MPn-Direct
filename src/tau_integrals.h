//
// Created by aedoran on 6/8/18.
//

#ifndef MC_MP3_DIRECT_TAU_INTEGRALS_H
#define MC_MP3_DIRECT_TAU_INTEGRALS_H

#include <algorithm>
#include "basis/qc_basis.h"
#include "qc_random.h"
class Stochastic_Tau {
 public:
  void resize(int dimm, Basis& basis) {
    tau.resize(dimm);
    wgt.resize(dimm);

    exp_tau.resize(dimm);
    for (auto &it : exp_tau) {
      it.resize(basis.ivir2);
    }

    scratch.resize(basis.ivir2);
    lambda = 2.0 * (basis.nw_en[basis.ivir1] - basis.nw_en[basis.iocc2 - 1]);
  }

  void new_tau(Basis& basis, Random& random) {
    // generate new tau point and weights
    for (auto i = 0; i < tau.size(); i++) {
      double p = random.get_rand();
      tau[i] = -log(1.0 - p) / lambda;
      wgt[i] = 1.0 / (lambda * (1.0 - p));

      for (auto im = basis.iocc1; im < basis.iocc2; im++) {
        exp_tau[i][im] = exp(basis.nw_en[im] * tau[i]);
      }
      for (auto am = basis.ivir1; am < basis.ivir2; am++) {
        exp_tau[i][am] = exp(-basis.nw_en[am] * tau[i]);
      }
    }
  }

  std::vector<double> get_tau(std::vector<int> index) {
    if (index.size() == 1) {
      return exp_tau[index[0]];
    } else {
      std::fill(scratch.begin(), scratch.end(), 0.0);
      for (auto &it : index) {
        std::transform(scratch.begin(), scratch.end(), exp_tau[it].begin(), scratch.begin(), std::multiplies<double>());
      }
      return scratch;
    }
  }
  double get_wgt(int dimm) {
    return std::accumulate(wgt.begin(), wgt.begin()+dimm, 1.0, std::multiplies<double>());
  }

  double lambda;
  std::vector<double> tau;
  std::vector<double> wgt;
  std::vector<double> scratch;
  std::vector<std::vector<double>> exp_tau;
};

#endif //MC_MP3_DIRECT_TAU_INTEGRALS_H
