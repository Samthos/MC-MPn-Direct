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
    iocc1 = basis.iocc1;
    iocc2 = basis.iocc2;
    ivir1 = basis.ivir1;
    ivir2 = basis.ivir2;

    evals.resize(ivir2);
    std::copy(basis.nw_en, basis.nw_en + ivir2, evals.begin());

    tau.resize(dimm);
    wgt.resize(dimm);

    exp_tau.resize(dimm);
    for (auto &it : exp_tau) {
      it.resize(ivir2);
    }

    scratch.resize(ivir2);
    lambda = 2.0 * (evals[ivir1] - evals[iocc2 - 1]);
  }

  void new_tau(Random& random) {
    // generate new tau point and weights
    for (auto i = 0; i < tau.size(); i++) {
      double p = random.get_rand();
      tau[i] = -log(1.0 - p) / lambda;
      wgt[i] = 1.0 / (lambda * (1.0 - p));

      for (auto im = iocc1; im < iocc2; im++) {
        exp_tau[i][im] = exp(evals[im] * tau[i]);
      }
      for (auto am = ivir1; am < ivir2; am++) {
        exp_tau[i][am] = exp(-evals[am] * tau[i]);
      }
    }
  }

  std::vector<double> get_exp_tau(std::vector<int> index) {
    if (index.size() == 1) {
      return exp_tau[index[0]];
    } else {
      std::fill(scratch.begin(), scratch.end(), 1.0);
      for (auto &it : index) {
        std::transform(scratch.begin(), scratch.end(), exp_tau[it].begin(), scratch.begin(), std::multiplies<double>());
      }
      return scratch;
    }
  }
  double get_tau(int index) {
    return tau[index];
  }
  double get_gfn_tau(std::vector<int> index, int offset, int conjugate) {
    double s(1.0);
    if (index.size() == 1) {
      s = exp_tau[index[0]][iocc2 - 1 + offset];
    } else {
      for (auto &it : index) {
        s *= exp_tau[it][iocc2 - 1 + offset];
      }
    }
    if (conjugate && offset <= 0) {
      s = 1.0 / s;
    } else if (!conjugate && offset > 0) {
      s = 1.0 / s;
    }
    return s;
  }
  double get_wgt(int dimm) {
    return std::accumulate(wgt.begin(), wgt.begin()+dimm, 1.0, std::multiplies<double>());
  }

 private:
  int iocc1, iocc2, ivir1, ivir2;
  double lambda;
  std::vector<double> evals;
  std::vector<double> tau;
  std::vector<double> wgt;
  std::vector<double> scratch;
  std::vector<std::vector<double>> exp_tau;
};

#endif //MC_MP3_DIRECT_TAU_INTEGRALS_H
