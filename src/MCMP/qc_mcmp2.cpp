//
// Created by aedoran on 6/1/18.
//

#ifdef HAVE_CONFIG_H_
#include "config.h"
#endif

#include <algorithm>
#include <functional>

#include "../qc_monte.h"

void MP2::mcmp2_energy(double& emp2, std::vector<double>& control) {
  int im, am;
  double icount2;

  double o_13, o_14, o_23, o_24;
  double v_13, v_14, v_23, v_24;
  double a_resk, emp2a;
  double b_resk, emp2b;

  tau.new_tau(basis, random);
  auto tau_values = tau.get_tau({0});
  std::vector<double> psi1Tau(ivir2), psi2Tau(ivir2);

  emp2 = 0.0;
  std::fill(control.begin(), control.end(), 0.0);
  icount2 = static_cast<double>(el_pair_list.size()) * static_cast<double>(el_pair_list.size() - 1);

  double s1, s2, emp2_rvj;
  std::vector<double> control_j(control.size());
  for (int group = 0; group < iops.iopns[KEYS::MC_PAIR_GROUPS]; group++) {
    auto iStart = el_pair_list.begin() + group * iops.iopns[KEYS::MC_NPAIR];
    auto iStop = el_pair_list.begin() + (group + 1) * iops.iopns[KEYS::MC_NPAIR] - 1;
    auto jStop = el_pair_list.begin() + (group + 1) * iops.iopns[KEYS::MC_NPAIR];

    for (auto it = iStart; it != iStop; it++) {
      emp2_rvj = 0;
      std::fill(control_j.begin(), control_j.end(), 0.0);
      std::transform(it->psi1.begin(), it->psi1.end(), tau_values.begin(),
                     psi1Tau.begin(), std::multiplies<double>());
      std::transform(it->psi2.begin(), it->psi2.end(), tau_values.begin(),
                     psi2Tau.begin(), std::multiplies<double>());
      for (auto jt = it + 1; jt != jStop; jt++) {
        o_13 = 0.0;
        o_14 = 0.0;
        o_23 = 0.0;
        o_24 = 0.0;
        v_13 = 0.0;
        v_14 = 0.0;
        v_23 = 0.0;
        v_24 = 0.0;

        for (im = iocc1; im < iocc2; im++) {
          o_13 = o_13 + psi1Tau[im] * jt->psi1[im];
          o_14 = o_14 + psi1Tau[im] * jt->psi2[im];
          o_23 = o_23 + psi2Tau[im] * jt->psi1[im];
          o_24 = o_24 + psi2Tau[im] * jt->psi2[im];
        }
        for (am = ivir1; am < ivir2; am++) {
          v_13 = v_13 + psi1Tau[am] * jt->psi1[am];
          v_14 = v_14 + psi1Tau[am] * jt->psi2[am];
          v_23 = v_23 + psi2Tau[am] * jt->psi1[am];
          v_24 = v_24 + psi2Tau[am] * jt->psi2[am];
        }

        a_resk = (o_13 * o_24 * v_13 * v_24);
        b_resk = (o_14 * o_23 * v_13 * v_24);

        a_resk = a_resk + (o_14 * o_23 * v_14 * v_23);
        b_resk = b_resk + (o_13 * o_24 * v_14 * v_23);

        emp2a = a_resk * jt->rv;
        emp2b = b_resk * jt->rv;
        emp2_rvj = emp2_rvj - 2.0 * emp2a + emp2b;
        control_j[0] = control_j[0] + a_resk / jt->wgt;
        control_j[1] = control_j[1] + b_resk / jt->wgt;
      }
      emp2 += emp2_rvj * it->rv;
      std::transform(control_j.begin(), control_j.end(), control.begin(), control.begin(), [&](double x, double y) { return y + x / it->wgt; });
    }

    auto tau_wgt = tau.get_wgt(1);
    emp2 = emp2 * tau_wgt / icount2;
    std::transform(control.begin(), control.end(), control.begin(), [&](double c) { return c * tau_wgt / icount2; });
  }
}

void MP3::mcmp2_energy(double& emp2, std::vector<double>& control) {
  double icount2;

  double a_resk, emp2a;
  double b_resk, emp2b;

  emp2 = 0.0;
  std::fill(control.begin(), control.end(), 0.0);
  icount2 = static_cast<double>(el_pair_list.size()) * static_cast<double>(el_pair_list.size() - 1);

  double emp2_rvj;
  std::vector<double> control_j(control.size());

  for (auto it = 0; it != iops.iopns[KEYS::MC_NPAIR]; it++) {
    emp2_rvj = 0;
    std::fill(control_j.begin(), control_j.end(), 0.0);

    for (auto jt = it + 1; jt !=  iops.iopns[KEYS::MC_NPAIR]; jt++) {
      auto ijIndex = it * iops.iopns[KEYS::MC_NPAIR] + jt;
      a_resk = (ovps.d_ovps.os_13[ijIndex] * ovps.d_ovps.os_24[ijIndex] * ovps.d_ovps.vs_13[ijIndex] * ovps.d_ovps.vs_24[ijIndex]);
      b_resk = (ovps.d_ovps.os_14[ijIndex] * ovps.d_ovps.os_23[ijIndex] * ovps.d_ovps.vs_13[ijIndex] * ovps.d_ovps.vs_24[ijIndex]);

      a_resk = a_resk + (ovps.d_ovps.os_14[ijIndex] * ovps.d_ovps.os_23[ijIndex] * ovps.d_ovps.vs_14[ijIndex] * ovps.d_ovps.vs_23[ijIndex]);
      b_resk = b_resk + (ovps.d_ovps.os_13[ijIndex] * ovps.d_ovps.os_24[ijIndex] * ovps.d_ovps.vs_14[ijIndex] * ovps.d_ovps.vs_23[ijIndex]);

      emp2a = a_resk * el_pair_list[jt].rv;
      emp2b = b_resk * el_pair_list[jt].rv;
      emp2_rvj = emp2_rvj - 2.0 * emp2a + emp2b;
      control_j[0] = control_j[0] + a_resk / el_pair_list[jt].wgt;
      control_j[1] = control_j[1] + b_resk / el_pair_list[jt].wgt;
    }
    emp2 += emp2_rvj * el_pair_list[it].rv;
    std::transform(control_j.begin(), control_j.end(), control.begin(), control.begin(), [&](double x, double y) { return y + x / el_pair_list[it].wgt; });
  }
  emp2 = emp2 * ovps.t1_twgt / icount2;
  std::transform(control.begin(), control.end(), control.begin(), [&](double c) { return c * ovps.t1_twgt / icount2; });
}

