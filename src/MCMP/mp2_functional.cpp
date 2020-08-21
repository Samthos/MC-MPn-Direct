//
// Created by aedoran on 6/1/18.
//

#ifdef HAVE_CONFIG_H_
#include "config.h"
#endif

#include <algorithm>
#include <functional>

#include "mp2_functional.h"

MP_Functional* create_MP2_Functional(int cv_level) {
  MP_Functional* mcmp = nullptr;
  if (cv_level == 0) {
    mcmp = new MP2_Functional<0>;
  } else if (cv_level == 1) {
    mcmp = new MP2_Functional<1>;
  } else if (cv_level == 2) {
    mcmp = new MP2_Functional<2>;
  }
  
  if (mcmp == nullptr) {
    std::cerr << "MP2_Functional not supported with cv level " << cv_level << "\n";
    exit(0);
  }
  return mcmp;
}

template <int CVMP2>
void MP2_Functional<CVMP2>::energy(double& emp, std::vector<double>& control, OVPS_Host& ovps, Electron_Pair_List* electron_pair_list, Tau* tau) {
  double en2 = 0.0;
  std::vector<double> ctrl(control.size(), 0.0);

  std::array<double, 2> en{0, 0};
  std::array<double, 2> en_rj{0, 0};
  std::array<double, 2> en_wj{0, 0};

  for (auto it = 0; it != electron_pair_list->size(); it++) {
    en_rj.fill(0.0);
    if (CVMP2 >= 2) {
      en_wj.fill(0.0);
    }

    for (auto jt = it + 1; jt !=  electron_pair_list->size(); jt++) {
      auto ijIndex = it * electron_pair_list->size() + jt;
      en[0] = (ovps.o_set[0][0].s_11[ijIndex] * ovps.o_set[0][0].s_22[ijIndex] * ovps.v_set[0][0].s_11[ijIndex] * ovps.v_set[0][0].s_22[ijIndex]);
      en[1] = (ovps.o_set[0][0].s_12[ijIndex] * ovps.o_set[0][0].s_21[ijIndex] * ovps.v_set[0][0].s_11[ijIndex] * ovps.v_set[0][0].s_22[ijIndex]);

      en[0] = en[0] + (ovps.o_set[0][0].s_12[ijIndex] * ovps.o_set[0][0].s_21[ijIndex] * ovps.v_set[0][0].s_12[ijIndex] * ovps.v_set[0][0].s_21[ijIndex]);
      en[1] = en[1] + (ovps.o_set[0][0].s_11[ijIndex] * ovps.o_set[0][0].s_22[ijIndex] * ovps.v_set[0][0].s_12[ijIndex] * ovps.v_set[0][0].s_21[ijIndex]);

      std::transform(en.begin(), en.end(), en_rj.begin(), en_rj.begin(), [&](double x, double y) {return y + x * electron_pair_list->rv[jt];});
      if (CVMP2 >= 2) {
        std::transform(en.begin(), en.end(), en_wj.begin(), en_wj.begin(), [&](double x, double y) {return y + x / electron_pair_list->wgt[jt];});
      }
    }
    en2 += (en_rj[1] - 2.0 * en_rj[0]) * electron_pair_list->rv[it];
    if (CVMP2 >= 1) {
      std::transform(en_rj.begin(), en_rj.end(), ctrl.begin()+0, ctrl.begin()+0, [&](double x, double y) { return y + x / electron_pair_list->wgt[it]; });
    }
    if (CVMP2 >= 2) {
      std::transform(en_wj.begin(), en_wj.end(), ctrl.begin()+2, ctrl.begin()+2, [&](double x, double y) { return y + x * electron_pair_list->rv[it]; });
      std::transform(en_wj.begin(), en_wj.end(), ctrl.begin()+4, ctrl.begin()+4, [&](double x, double y) { return y + x / electron_pair_list->wgt[it]; });
    }
  }
  auto tau_wgt = tau->get_wgt(1);
  tau_wgt /= static_cast<double>(electron_pair_list->size());
  tau_wgt /= static_cast<double>(electron_pair_list->size() - 1);
  emp = emp + en2 * tau_wgt;
  if (CVMP2 >= 1) {
    std::transform(ctrl.begin(), ctrl.end(), control.begin(), control.begin(), [&](double c, double total) { return total + c * tau_wgt; });
  }
}

MP_Functional* create_Fast_MP2_Functional(int cv_level) {
  MP_Functional* mcmp = nullptr;
  if (cv_level == 0) {
    mcmp = new Fast_MP2_Functional<0>;
  } else if (cv_level == 1) {
    mcmp = new Fast_MP2_Functional<1>;
  } else if (cv_level == 2) {
    mcmp = new Fast_MP2_Functional<2>;
  }
  
  if (mcmp == nullptr) {
    std::cerr << "MP2_Functional not supported with cv level " << cv_level << "\n";
    exit(0);
  }
  return mcmp;
}

template <int CVMP2>
void Fast_MP2_Functional<CVMP2>::energy(double& emp, std::vector<double>& control, Wavefunction_Type& electron_pair_psi1, Wavefunction_Type& electron_pair_psi2, Electron_Pair_List* electron_pair_list, Tau* tau) {
  /* This functions computes the second-order MP2 energy for a single MC steps
   * The traces of the HF greens's functions are computed in-place in this function.
   * If higher order corrections are desired, the mcmp2_energy fucntions is more appropriate as it does not compute the HF green's function traces
   *
   * The wavefunction psi is obtained from the basis varaible declared in the MP class
   * The imaginary tiem varaiables are obtained from tau_values declared in the MP class
   *
   * Input Parameters:
   *   emp: reference to double. stores the energy calcaulted by this function
   *   control: reference to vector<double>. stores the control2 varaites to the mp2 energy for calcaulted by this function.
   */
  double o_13, o_14, o_23, o_24;
  double v_13, v_14, v_23, v_24;

  auto tau_values = tau->get_exp_tau(0, 0);

  double en2 = 0.0;
  std::vector<double> ctrl(control.size(), 0.0);

  std::array<double, 2> en{0, 0};
  std::array<double, 2> en_rj{0, 0};
  std::array<double, 2> en_wj{0, 0};

  int iocc1 = electron_pair_psi1.iocc1;
  int iocc2 = electron_pair_psi1.iocc2;
  int ivir1 = electron_pair_psi1.ivir1;
  int ivir2 = electron_pair_psi1.ivir2;

  for (auto it = 0; it < electron_pair_list->size(); ++it) {
    en_rj.fill(0.0);
    if (CVMP2 >= 2) {
      en_wj.fill(0.0);
    }
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
      if (CVMP2 >= 2) {
        std::transform(en.begin(), en.end(), en_wj.begin(), en_wj.begin(), [&](double x, double y) {return y + x / electron_pair_list->wgt[jt]; });
      }
    }
    en2 += (en_rj[1] - 2.0 * en_rj[0]) * electron_pair_list->rv[it];
    if (CVMP2 >= 1) {
      std::transform(en_rj.begin(), en_rj.end(), ctrl.begin()+0, ctrl.begin()+0, [&](double x, double y) { return y + x / electron_pair_list->wgt[it]; });
    }
    if (CVMP2 >= 2) {
      std::transform(en_wj.begin(), en_wj.end(), ctrl.begin()+2, ctrl.begin()+2, [&](double x, double y) { return y + x * electron_pair_list->rv[it]; });
      std::transform(en_wj.begin(), en_wj.end(), ctrl.begin()+4, ctrl.begin()+4, [&](double x, double y) { return y + x / electron_pair_list->wgt[it]; });
    }
  }

  auto tau_wgt = tau->get_wgt(1);
  tau_wgt /= static_cast<double>(electron_pair_list->size());
  tau_wgt /= static_cast<double>(electron_pair_list->size() - 1);
  emp += en2 * tau_wgt;
  if (CVMP2 >= 1) {
    std::transform(ctrl.begin(), ctrl.end(), control.begin(), control.begin(), [&](double c, double total) { return total + c * tau_wgt; });
  }
}

