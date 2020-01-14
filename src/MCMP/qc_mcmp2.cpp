//
// Created by aedoran on 6/1/18.
//

#ifdef HAVE_CONFIG_H_
#include "config.h"
#endif

#include <algorithm>
#include <functional>

#include "../qc_monte.h"

void MP::mcmp2_energy_fast(double& emp2, std::vector<double>& control2) {
  /* This functions computes the second-order MP2 energy for a single MC steps
   * The traces of the HF greens's functions are computed in-place in this function.
   * If higher order corrections are desired, the mcmp2_energy fucntions is more appropriate as it does not compute the HF green's function traces
   *
   * The wavefunction psi is obtained from the basis varaible declared in the MP class
   * The imaginary tiem varaiables are obtained from tau_values declared in the MP class
   *
   * Input Parameters:
   *   emp2: reference to double. stores the energy calcaulted by this function
   *   control2: reference to vector<double>. stores the control2 varaites to the mp2 energy for calcaulted by this function.
   */
  double o_13, o_14, o_23, o_24;
  double v_13, v_14, v_23, v_24;

  auto tau_values = tau->get_exp_tau(0, 0);

  double en2 = 0.0;
  std::vector<double> ctrl(control2.size(), 0.0);

  std::array<double, 2> en{0, 0};
  std::array<double, 2> en_rj{0, 0};
#if MP2CV >= 2
  std::array<double, 2> en_wj{0, 0};
#endif

  for (auto it = 0; it < electron_pair_list->size(); ++it) {
    en_rj.fill(0.0);
#if MP2CV >= 2
    en_wj.fill(0.0);
#endif
    std::transform(electron_pair_psi1.data() + it * electron_pair_psi1.lda,
        electron_pair_psi1.data() + (it+1) * electron_pair_psi1.lda, 
        tau_values.begin(),
        electron_pair_psi1.dataTau(),
        std::multiplies<>());
    std::transform(electron_pair_psi2.data() + it * electron_pair_psi2.lda, 
        electron_pair_psi2.data() + (it+1) * electron_pair_psi2.lda,
        tau_values.begin(),
        electron_pair_psi2.dataTau(),
        std::multiplies<>());
    for (auto jt = it + 1; jt != electron_pair_list->size(); jt++) {
      o_13 = 0.0;
      o_14 = 0.0;
      o_23 = 0.0;
      o_24 = 0.0;
      v_13 = 0.0;
      v_14 = 0.0;
      v_23 = 0.0;
      v_24 = 0.0;

      // compute the traces of the the HF green's functions
      for (auto im = iocc1; im < iocc2; im++) {
        o_13 = o_13 + electron_pair_psi1.psiTau.data()[im] * electron_pair_psi1.psi.data()[jt * electron_pair_psi1.lda + (im)];
        o_14 = o_14 + electron_pair_psi1.psiTau.data()[im] * electron_pair_psi2.psi.data()[jt * electron_pair_psi2.lda + (im)];
        o_23 = o_23 + electron_pair_psi2.psiTau.data()[im] * electron_pair_psi1.psi.data()[jt * electron_pair_psi1.lda + (im)];
        o_24 = o_24 + electron_pair_psi2.psiTau.data()[im] * electron_pair_psi2.psi.data()[jt * electron_pair_psi2.lda + (im)];
      }
      for (auto am = ivir1; am < ivir2; am++) {
        v_13 = v_13 + electron_pair_psi1.psiTau.data()[am] * electron_pair_psi1.psi.data()[jt * electron_pair_psi1.lda + (am)];
        v_14 = v_14 + electron_pair_psi1.psiTau.data()[am] * electron_pair_psi2.psi.data()[jt * electron_pair_psi2.lda + (am)];
        v_23 = v_23 + electron_pair_psi2.psiTau.data()[am] * electron_pair_psi1.psi.data()[jt * electron_pair_psi1.lda + (am)];
        v_24 = v_24 + electron_pair_psi2.psiTau.data()[am] * electron_pair_psi2.psi.data()[jt * electron_pair_psi2.lda + (am)];
      }

      // compute the energy
      en[0] = (o_13 * o_24 * v_13 * v_24);
      en[1] = (o_14 * o_23 * v_13 * v_24);

      en[0] = en[0] + (o_14 * o_23 * v_14 * v_23);
      en[1] = en[1] + (o_13 * o_24 * v_14 * v_23);

      std::transform(en.begin(), en.end(), en_rj.begin(), en_rj.begin(), [&](double x, double y) {return y + x * electron_pair_list->rv[jt]; });
#if MP2CV >= 2
      std::transform(en.begin(), en.end(), en_wj.begin(), en_wj.begin(), [&](double x, double y) {return y + x / electron_pair_list->wgt[jt]; });
#endif
    }
    en2 += (en_rj[1] - 2.0 * en_rj[0]) * electron_pair_list->rv[it];
#if MP2CV >= 1
    std::transform(en_rj.begin(), en_rj.end(), ctrl.begin()+0, ctrl.begin()+0, [&](double x, double y) { return y + x / electron_pair_list->wgt[it]; });
#endif
#if MP2CV >= 2
    std::transform(en_wj.begin(), en_wj.end(), ctrl.begin()+2, ctrl.begin()+2, [&](double x, double y) { return y + x * electron_pair_list->rv[it]; });
    std::transform(en_wj.begin(), en_wj.end(), ctrl.begin()+4, ctrl.begin()+4, [&](double x, double y) { return y + x / electron_pair_list->wgt[it]; });
#endif
  }

  auto tau_wgt = tau->get_wgt(1);
  tau_wgt /= static_cast<double>(electron_pair_list->size());
  tau_wgt /= static_cast<double>(electron_pair_list->size() - 1);
  emp2 += en2 * tau_wgt;
#if MP2CV >= 1
  std::transform(ctrl.begin(), ctrl.end(), control2.begin(), control2.begin(), [&](double c, double total) { return total + c * tau_wgt; });
#endif
}

void MP::mcmp2_energy(double& emp2, std::vector<double>& control2) {
  double en2 = 0.0;
  std::vector<double> ctrl(control2.size(), 0.0);

  std::array<double, 2> en{0, 0};
  std::array<double, 2> en_rj{0, 0};
#if MP2CV >= 2
  std::array<double, 2> en_wj{0, 0};
#endif

  for (auto it = 0; it != iops.iopns[KEYS::MC_NPAIR]; it++) {
    en_rj.fill(0.0);
#if MP2CV >= 2
    en_wj.fill(0.0);
#endif

    for (auto jt = it + 1; jt !=  iops.iopns[KEYS::MC_NPAIR]; jt++) {
      auto ijIndex = it * iops.iopns[KEYS::MC_NPAIR] + jt;
      en[0] = (ovps.o_set[0][0].s_11[ijIndex] * ovps.o_set[0][0].s_22[ijIndex] * ovps.v_set[0][0].s_11[ijIndex] * ovps.v_set[0][0].s_22[ijIndex]);
      en[1] = (ovps.o_set[0][0].s_12[ijIndex] * ovps.o_set[0][0].s_21[ijIndex] * ovps.v_set[0][0].s_11[ijIndex] * ovps.v_set[0][0].s_22[ijIndex]);

      en[0] = en[0] + (ovps.o_set[0][0].s_12[ijIndex] * ovps.o_set[0][0].s_21[ijIndex] * ovps.v_set[0][0].s_12[ijIndex] * ovps.v_set[0][0].s_21[ijIndex]);
      en[1] = en[1] + (ovps.o_set[0][0].s_11[ijIndex] * ovps.o_set[0][0].s_22[ijIndex] * ovps.v_set[0][0].s_12[ijIndex] * ovps.v_set[0][0].s_21[ijIndex]);

      std::transform(en.begin(), en.end(), en_rj.begin(), en_rj.begin(), [&](double x, double y) {return y + x * electron_pair_list->rv[jt];});
#if MP2CV >= 2
      std::transform(en.begin(), en.end(), en_wj.begin(), en_wj.begin(), [&](double x, double y) {return y + x / electron_pair_list->wgt[jt];});
#endif
    }
    en2 += (en_rj[1] - 2.0 * en_rj[0]) * electron_pair_list->rv[it];
#if MP2CV >= 1
    std::transform(en_rj.begin(), en_rj.end(), ctrl.begin()+0, ctrl.begin()+0, [&](double x, double y) { return y + x / electron_pair_list->wgt[it]; });
#endif
#if MP2CV >= 2
    std::transform(en_wj.begin(), en_wj.end(), ctrl.begin()+2, ctrl.begin()+2, [&](double x, double y) { return y + x * electron_pair_list->rv[it]; });
    std::transform(en_wj.begin(), en_wj.end(), ctrl.begin()+4, ctrl.begin()+4, [&](double x, double y) { return y + x / electron_pair_list->wgt[it]; });
#endif
  }
  auto tau_wgt = tau->get_wgt(1);
  tau_wgt /= static_cast<double>(electron_pair_list->size());
  tau_wgt /= static_cast<double>(electron_pair_list->size() - 1);
  emp2 = emp2 + en2 * tau_wgt;
#if MP2CV >= 1
  std::transform(ctrl.begin(), ctrl.end(), control2.begin(), control2.begin(), [&](double c, double total) { return total + c * tau_wgt; });
#endif
}

