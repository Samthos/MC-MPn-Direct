//
// Created by aedoran on 6/1/18.
//

#include <algorithm>
#include <vector>
#include "../qc_monte.h"

void MP3::mcmp3_energy(double& emp3, std::vector<double>& control) {
  std::array<double, 6> en;

  double en_k, en_jk;
  std::array<double, 6> c_k, c_jk;

  emp3 = 0.0;
  std::fill(control.begin(), control.end(), 0.0);

  for (auto i = 0; i < iops.iopns[KEYS::MC_NPAIR]; i++) {
    en_jk = 0;
    c_jk.fill(0);
    for (auto j = i + 1; j < iops.iopns[KEYS::MC_NPAIR]; j++) {
      auto ij = i * iops.iopns[KEYS::MC_NPAIR] + j;

      en_k = 0;
      c_k.fill(0);

      for (auto k = j + 1; k < iops.iopns[KEYS::MC_NPAIR]; k++) {
        auto ik = i * iops.iopns[KEYS::MC_NPAIR] + k;
        auto jk = j * iops.iopns[KEYS::MC_NPAIR] + k;

        en[0] = (2 * ovps.v_set[0][0].s_11[ij] * ovps.v_set[0][0].s_22[ij] - 1 * ovps.v_set[0][0].s_12[ij] * ovps.v_set[0][0].s_21[ij]) * ovps.o_set[1][0].s_11[ik] * ovps.o_set[1][0].s_22[ik] * ovps.v_set[1][1].s_11[jk] * ovps.v_set[1][1].s_22[jk];
        en[1] = (2 * ovps.o_set[0][0].s_21[ij] * ovps.v_set[1][1].s_22[jk] - 4 * ovps.o_set[0][0].s_22[ij] * ovps.v_set[1][1].s_12[jk]) * ovps.v_set[0][0].s_22[ij] * ovps.o_set[1][0].s_12[ik] * ovps.v_set[1][0].s_11[ik] * ovps.o_set[1][1].s_11[jk];
        en[2] = (8 * ovps.o_set[0][0].s_22[ij] * ovps.v_set[1][1].s_11[jk] - 4 * ovps.o_set[0][0].s_21[ij] * ovps.v_set[1][1].s_21[jk]) * ovps.v_set[0][0].s_22[ij] * ovps.o_set[1][0].s_12[ik] * ovps.v_set[1][0].s_12[ik] * ovps.o_set[1][1].s_11[jk];
        en[3] = (2 * ovps.o_set[0][0].s_22[ij] * ovps.v_set[1][1].s_12[jk] - 4 * ovps.o_set[0][0].s_21[ij] * ovps.v_set[1][1].s_22[jk]) * ovps.v_set[0][0].s_12[ij] * ovps.o_set[1][0].s_12[ik] * ovps.v_set[1][0].s_21[ik] * ovps.o_set[1][1].s_11[jk];
        en[4] = (2 * ovps.o_set[0][0].s_21[ij] * ovps.v_set[1][1].s_21[jk] - 4 * ovps.o_set[0][0].s_22[ij] * ovps.v_set[1][1].s_11[jk]) * ovps.v_set[0][0].s_12[ij] * ovps.o_set[1][0].s_12[ik] * ovps.v_set[1][0].s_22[ik] * ovps.o_set[1][1].s_11[jk];
        en[5] = (2 * ovps.o_set[0][0].s_21[ij] * ovps.o_set[0][0].s_12[ij] - 1 * ovps.o_set[0][0].s_11[ij] * ovps.o_set[0][0].s_22[ij]) * ovps.v_set[1][0].s_12[ik] * ovps.v_set[1][0].s_21[ik] * ovps.o_set[1][1].s_11[jk] * ovps.o_set[1][1].s_22[jk];

        std::transform(c_k.begin(), c_k.end(), en.begin(), c_k.begin(),
                       [&](double x, double y) { return x + y / el_pair_list[k].wgt; });
        en_k += std::accumulate(en.begin(), en.end(), 0.0) * el_pair_list[k].rv;
      }
      std::transform(c_jk.begin(), c_jk.end(), c_k.begin(), c_jk.begin(),
                     [&](double x, double y) { return x + y / el_pair_list[j].wgt; });
      en_jk += en_k * el_pair_list[j].rv;
    }
    std::transform(control.begin(), control.end(), c_jk.begin(), control.begin(),
                   [&](double x, double y) { return x + y / el_pair_list[i].wgt; });
    emp3 += en_jk * el_pair_list[i].rv;
  }

  auto tau_wgt = tau.get_wgt(2);
  std::transform(control.begin(), control.end(), control.begin(),
                 [&](double x) { return x * tau_wgt; });
  emp3 *= tau_wgt;

  // divide by number of RW samples
  auto nsamp = static_cast<double>(iops.iopns[KEYS::MC_NPAIR]);
  nsamp *= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 1);
  nsamp *= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 2);
  nsamp /= 6;
  emp3 = emp3 / nsamp;
  std::transform(control.begin(), control.end(), control.begin(),
                 [nsamp](double x) { return x / nsamp; });
}
