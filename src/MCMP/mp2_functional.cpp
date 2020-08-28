//
// Created by aedoran on 6/1/18.
//

#include <algorithm>
#include <functional>
#include <iostream>

#include "mp2_functional.h"

template <int CVMP2>
void MP2_Functional<CVMP2>::energy(double& emp, std::vector<double>& control, OVPS_Type& ovps, Electron_Pair_List* electron_pair_list, Tau* tau) {
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

    for (auto jt = it+1; jt !=  electron_pair_list->size(); jt++) {
      auto ijIndex = it * electron_pair_list->size() + jt;
      en[0] = (ovps.o_set[0][0].s_11[ijIndex] * ovps.o_set[0][0].s_22[ijIndex] * ovps.v_set[0][0].s_11[ijIndex] * ovps.v_set[0][0].s_22[ijIndex]);
      en[1] = (ovps.o_set[0][0].s_12[ijIndex] * ovps.o_set[0][0].s_21[ijIndex] * ovps.v_set[0][0].s_11[ijIndex] * ovps.v_set[0][0].s_22[ijIndex]);

      en[0] = en[0] + (ovps.o_set[0][0].s_12[ijIndex] * ovps.o_set[0][0].s_21[ijIndex] * ovps.v_set[0][0].s_12[ijIndex] * ovps.v_set[0][0].s_21[ijIndex]);
      en[1] = en[1] + (ovps.o_set[0][0].s_11[ijIndex] * ovps.o_set[0][0].s_22[ijIndex] * ovps.v_set[0][0].s_12[ijIndex] * ovps.v_set[0][0].s_21[ijIndex]);

      std::transform(en.begin(), en.end(), en_rj.begin(), en_rj.begin(), [&](double x, double y) {return y + x * electron_pair_list->rv[jt];});
      if (CVMP2 >= 2) {
        std::transform(en.begin(), en.end(), en_wj.begin(), en_wj.begin(), [&](double x, double y) {return y + x * electron_pair_list->inverse_weight[jt];});
      }
    }
    en2 += (en_rj[1] - 2.0 * en_rj[0]) * electron_pair_list->rv[it];
    if (CVMP2 >= 1) {
      std::transform(en_rj.begin(), en_rj.end(), ctrl.begin()+0, ctrl.begin()+0, [&](double x, double y) { return y + x * electron_pair_list->inverse_weight[it]; });
    }
    if (CVMP2 >= 2) {
      std::transform(en_wj.begin(), en_wj.end(), ctrl.begin()+2, ctrl.begin()+2, [&](double x, double y) { return y + x * electron_pair_list->rv[it]; });
      std::transform(en_wj.begin(), en_wj.end(), ctrl.begin()+4, ctrl.begin()+4, [&](double x, double y) { return y + x * electron_pair_list->inverse_weight[it]; });
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
