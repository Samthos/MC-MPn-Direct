//
// Created by aedoran on 6/1/18.
//

#ifdef HAVE_CONFIG_H_
#include "config.h"
#endif

#include <algorithm>
#include <functional>

#include "../qc_monte.h"

void MP::mcmp2_energy_fast(double& emp2, std::vector<double>& control) {
  /* This functions computes the second-order MP2 energy for a single MC steps
   * The traces of the HF greens's functions are computed in-place in this function.
   * If higher order corrections are desired, the mcmp2_energy fucntions is more appropriate as it does not compute the HF green's function traces
   *
   * The wavefunction psi is obtained from the basis varaible declared in the MP class
   * The imaginary tiem varaiables are obtained from tau_values declared in the MP class
   *
   * Input Parameters:
   *   emp2: reference to double. stores the energy calcaulted by this function
   *   control: reference to vector<double>. stores the control varaites to the mp2 energy for calcaulted by this function.
   */
  int im, am;
  double icount2;

  double o_13, o_14, o_23, o_24;
  double v_13, v_14, v_23, v_24;
  double a_resk, emp2a;
  double b_resk, emp2b;

  auto tau_values = tau.get_exp_tau(0, 0);

  emp2 = 0.0;
  std::fill(control.begin(), control.end(), 0.0);
  icount2 = static_cast<double>(el_pair_list.size()) * static_cast<double>(el_pair_list.size() - 1);

  double emp2_rvj;
  std::vector<double> control_j(control.size());
  for (auto it = 0; it < el_pair_list.size(); ++it) {
    emp2_rvj = 0;
    std::fill(control_j.begin(), control_j.end(), 0.0);
    std::transform(basis.h_basis.psi1 + it * (ivir2-iocc1), basis.h_basis.psi1 + (it+1) * (ivir2-iocc1), tau_values.begin() + iocc1, basis.h_basis.psiTau1, std::multiplies<>());
    std::transform(basis.h_basis.psi2 + it * (ivir2-iocc1), basis.h_basis.psi2 + (it+1) * (ivir2-iocc1), tau_values.begin() + iocc1, basis.h_basis.psiTau2, std::multiplies<>());
    for (auto jt = it + 1; jt != el_pair_list.size(); jt++) {
      o_13 = 0.0;
      o_14 = 0.0;
      o_23 = 0.0;
      o_24 = 0.0;
      v_13 = 0.0;
      v_14 = 0.0;
      v_23 = 0.0;
      v_24 = 0.0;

      // compute the traces of the the HF green's functions
      for (im = 0; im < iocc2-iocc1; im++) {
        o_13 = o_13 + basis.h_basis.psiTau1[im] * basis.h_basis.psi1[jt * (ivir2-iocc1) + (im)];
        o_14 = o_14 + basis.h_basis.psiTau1[im] * basis.h_basis.psi2[jt * (ivir2-iocc1) + (im)];
        o_23 = o_23 + basis.h_basis.psiTau2[im] * basis.h_basis.psi1[jt * (ivir2-iocc1) + (im)];
        o_24 = o_24 + basis.h_basis.psiTau2[im] * basis.h_basis.psi2[jt * (ivir2-iocc1) + (im)];
      }
      for (am = iocc2-iocc1; am < ivir2-iocc1; am++) {
        v_13 = v_13 + basis.h_basis.psiTau1[am] * basis.h_basis.psi1[jt * (ivir2-iocc1) + (am)];
        v_14 = v_14 + basis.h_basis.psiTau1[am] * basis.h_basis.psi2[jt * (ivir2-iocc1) + (am)];
        v_23 = v_23 + basis.h_basis.psiTau2[am] * basis.h_basis.psi1[jt * (ivir2-iocc1) + (am)];
        v_24 = v_24 + basis.h_basis.psiTau2[am] * basis.h_basis.psi2[jt * (ivir2-iocc1) + (am)];
      }

      // compute the energy
      a_resk = (o_13 * o_24 * v_13 * v_24);
      b_resk = (o_14 * o_23 * v_13 * v_24);

      a_resk = a_resk + (o_14 * o_23 * v_14 * v_23);
      b_resk = b_resk + (o_13 * o_24 * v_14 * v_23);

      emp2a = a_resk * el_pair_list[jt].rv;
      emp2b = b_resk * el_pair_list[jt].rv;
      emp2_rvj = emp2_rvj - 2.0 * emp2a + emp2b;
      control_j[0] = control_j[0] + a_resk / el_pair_list[jt].wgt;
      control_j[1] = control_j[1] + b_resk / el_pair_list[jt].wgt;

      control_j[2] = control_j[2] + a_resk / el_pair_list[jt].wgt;
      control_j[3] = control_j[3] + b_resk / el_pair_list[jt].wgt;

      control_j[4] = control_j[4] + a_resk * el_pair_list[jt].rv;
      control_j[5] = control_j[5] + b_resk * el_pair_list[jt].rv;
    }
    emp2 += emp2_rvj * el_pair_list[it].rv;
    std::transform(control_j.begin(), control_j.begin()+2, control.begin(), control.begin(), [&](double x, double y) { return y + x * el_pair_list[it].rv; });
    std::transform(control_j.begin()+2, control_j.end(), control.begin()+2, control.begin()+2, [&](double x, double y) { return y + x / el_pair_list[it].wgt; });
  }

  auto tau_wgt = tau.get_wgt(1);
  emp2 = emp2 * tau_wgt / icount2;
  std::transform(control.begin(), control.end(), control.begin(), [&](double c) { return c * tau_wgt / icount2; });
}

void MP::mcmp2_energy(double& emp2, std::vector<double>& control) {
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
      a_resk = (ovps.o_set[0][0].s_11[ijIndex] * ovps.o_set[0][0].s_22[ijIndex] * ovps.v_set[0][0].s_11[ijIndex] * ovps.v_set[0][0].s_22[ijIndex]);
      b_resk = (ovps.o_set[0][0].s_12[ijIndex] * ovps.o_set[0][0].s_21[ijIndex] * ovps.v_set[0][0].s_11[ijIndex] * ovps.v_set[0][0].s_22[ijIndex]);

      a_resk = a_resk + (ovps.o_set[0][0].s_12[ijIndex] * ovps.o_set[0][0].s_21[ijIndex] * ovps.v_set[0][0].s_12[ijIndex] * ovps.v_set[0][0].s_21[ijIndex]);
      b_resk = b_resk + (ovps.o_set[0][0].s_11[ijIndex] * ovps.o_set[0][0].s_22[ijIndex] * ovps.v_set[0][0].s_12[ijIndex] * ovps.v_set[0][0].s_21[ijIndex]);

      emp2a = a_resk * el_pair_list[jt].rv;
      emp2b = b_resk * el_pair_list[jt].rv;
      emp2_rvj = emp2_rvj - 2.0 * emp2a + emp2b;
      control_j[0] = control_j[0] + a_resk / el_pair_list[jt].wgt;
      control_j[1] = control_j[1] + b_resk / el_pair_list[jt].wgt;

      control_j[2] = control_j[2] + a_resk / el_pair_list[jt].wgt;
      control_j[3] = control_j[3] + b_resk / el_pair_list[jt].wgt;

      control_j[4] = control_j[4] + a_resk * el_pair_list[jt].rv;
      control_j[5] = control_j[5] + b_resk * el_pair_list[jt].rv;
    }
    emp2 += emp2_rvj * el_pair_list[it].rv;
    std::transform(control_j.begin(), control_j.begin()+2, control.begin(), control.begin(), [&](double x, double y) { return y + x * el_pair_list[it].rv; });
    std::transform(control_j.begin()+2, control_j.end(), control.begin()+2, control.begin()+2, [&](double x, double y) { return y + x / el_pair_list[it].wgt; });
  }
  auto tau_wgt = tau.get_wgt(1);
  emp2 = emp2 * tau_wgt / icount2;
  std::transform(control.begin(), control.end(), control.begin(), [&](double c) { return c * tau_wgt / icount2; });
}

