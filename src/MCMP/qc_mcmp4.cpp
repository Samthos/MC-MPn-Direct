//
// Created by aedoran on 6/13/18.
//
#include "../qc_monte.h"

void MP::mcmp4_energy(double& emp4, std::vector<double>& control) {
  emp4 = 0.0;
  control[0] = 0.0;
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double en_jkl = 0;
    double ct_jkl = 0;
    for (auto jt = it+1; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
      auto ij = it * iops.iopns[KEYS::MC_NPAIR] + jt;

      double en_kl = 0;
      double ct_kl = 0;
      for (auto kt = jt+1; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
        auto ik = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jk = jt * iops.iopns[KEYS::MC_NPAIR] + kt;

        double en_l = 0;
        double ct_l = 0;
        for (auto lt = kt+1; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
          auto il = it * iops.iopns[KEYS::MC_NPAIR] + lt;
          auto jl = jt * iops.iopns[KEYS::MC_NPAIR] + lt;
          auto kl = kt * iops.iopns[KEYS::MC_NPAIR] + lt;

          double en = 0.0;

          #include "qc_mcmp4_ijkl.h"

          en_l += en * el_pair_list[lt].rv;
          ct_l += en / el_pair_list[lt].wgt;
        }
        ct_kl += ct_l / el_pair_list[kt].wgt;
        en_kl += en_l * el_pair_list[kt].rv;
      }
      ct_jkl += ct_kl / el_pair_list[jt].wgt;
      en_jkl += en_kl * el_pair_list[jt].rv;
    }
    control[0] += ct_jkl / el_pair_list[it].wgt;
    emp4       += en_jkl * el_pair_list[it].rv;
  }

  auto tau_wgt = tau.get_wgt(3);
  std::transform(control.begin(), control.end(), control.begin(),
                 [&](double x) { return x * tau_wgt; });
  emp4 *= tau_wgt;

  auto nsamp = static_cast<double>(iops.iopns[KEYS::MC_NPAIR]);
  nsamp *= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 1);
  nsamp *= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 2);
  nsamp *= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 3);
  nsamp /= 24;
  emp4 /= nsamp;
  std::transform(control.begin(), control.end(), control.begin(),
                 [nsamp](double x) { return x / nsamp; });
}
