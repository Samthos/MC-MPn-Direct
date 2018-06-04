//
// Created by aedoran on 6/1/18.
//

#include <algorithm>
#include <vector>
#include "../qc_monte.h"

void MP3::mcmp3_energy(double& emp3, std::vector<double>& control) {
  std::vector<double> en(control.size(), 0);

  emp3 = 0.0;
  std::fill(control.begin(), control.end(), 0.0);

  for (auto i = 0; i < iops.iopns[KEYS::MC_NPAIR]; i++) {
    double en_jk = 0;
    std::vector<double> c_jk(control.size(), 0);
    for (auto j = i + 1; j < iops.iopns[KEYS::MC_NPAIR]; j++) {
      auto ij = i * iops.iopns[KEYS::MC_NPAIR] + j;
      double en_k = 0;
      std::vector<double> c_k(control.size(), 0);

      for (auto k = j + 1; k < iops.iopns[KEYS::MC_NPAIR]; k++) {
        auto ik = i * iops.iopns[KEYS::MC_NPAIR] + k;
        auto jk = j * iops.iopns[KEYS::MC_NPAIR] + k;

        en[0] = (2 * ovps.d_ovps.vs_13[ij] * ovps.d_ovps.vs_24[ij] -1 * ovps.d_ovps.vs_14[ij] * ovps.d_ovps.vs_23[ij]) * ovps.d_ovps.os_15[ik] * ovps.d_ovps.os_26[ik] * ovps.d_ovps.vs_35[jk] * ovps.d_ovps.vs_46[jk];
        en[1] = (2 * ovps.d_ovps.os_23[ij] * ovps.d_ovps.vs_46[jk] - 4 * ovps.d_ovps.os_24[ij] * ovps.d_ovps.vs_36[jk]) * ovps.d_ovps.vs_24[ij] * ovps.d_ovps.os_16[ik] * ovps.d_ovps.vs_15[ik] * ovps.d_ovps.os_35[jk];
        en[2] = (8 * ovps.d_ovps.os_24[ij] * ovps.d_ovps.vs_35[jk] - 4 * ovps.d_ovps.os_23[ij] * ovps.d_ovps.vs_45[jk]) * ovps.d_ovps.vs_24[ij] * ovps.d_ovps.os_16[ik] * ovps.d_ovps.vs_16[ik] * ovps.d_ovps.os_35[jk];
        en[3] = (2 * ovps.d_ovps.os_24[ij] * ovps.d_ovps.vs_36[jk] - 4 * ovps.d_ovps.os_23[ij] * ovps.d_ovps.vs_46[jk]) * ovps.d_ovps.vs_14[ij] * ovps.d_ovps.os_16[ik] * ovps.d_ovps.vs_25[ik] * ovps.d_ovps.os_35[jk];
        en[4] = (2 * ovps.d_ovps.os_23[ij] * ovps.d_ovps.vs_45[jk] - 4 * ovps.d_ovps.os_24[ij] * ovps.d_ovps.vs_35[jk]) * ovps.d_ovps.vs_14[ij] * ovps.d_ovps.os_16[ik] * ovps.d_ovps.vs_26[ik] * ovps.d_ovps.os_35[jk];
        en[5] = (2 * ovps.d_ovps.os_23[ij] * ovps.d_ovps.os_14[ij] -1 * ovps.d_ovps.os_13[ij] * ovps.d_ovps.os_24[ij]) * ovps.d_ovps.vs_16[ik] * ovps.d_ovps.vs_25[ik] * ovps.d_ovps.os_35[jk] * ovps.d_ovps.os_46[jk];

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

  std::transform(control.begin(), control.end(), control.begin(),
                 [&](double x) { return x * ovps.t2_twgt; });
  emp3 *= ovps.t2_twgt;

  // divide by number of RW samples
  auto nsamp = static_cast<double>(iops.iopns[KEYS::MC_NPAIR]);
  nsamp *= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 1);
  nsamp *= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 2);
  nsamp /= 6;
  emp3 = emp3 / nsamp;
  std::transform(control.begin(), control.end(), control.begin(),
                 [nsamp](double x) { return x / nsamp; });
}
