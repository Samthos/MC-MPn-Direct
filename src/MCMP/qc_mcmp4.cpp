//
// Created by aedoran on 6/13/18.
//
#include "../qc_monte.h"

void MP::mcmp4_energy(double& emp4, std::vector<double>& control) {
  emp4 = 0.0;
  control[0] = 0.0;
  /*
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double en_jkl = 0;
    double ct_jkl = 0;
    for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
      if (it == jt) continue;
      auto ij = it * iops.iopns[KEYS::MC_NPAIR] + jt;

      double en_kl = 0;
      double ct_kl = 0;
      for (auto kt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
        if (it == kt || jt == kt) continue;
        auto ik = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jk = jt * iops.iopns[KEYS::MC_NPAIR] + kt;

        double en_l = 0;
        double ct_l = 0;
        for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
          if (it == lt || jt == lt || kt == lt) continue;
          auto il = it * iops.iopns[KEYS::MC_NPAIR] + lt;
          auto jl = jt * iops.iopns[KEYS::MC_NPAIR] + lt;
          auto kl = kt * iops.iopns[KEYS::MC_NPAIR] + lt;

          double en = 0;

          #include "qc_mcmp4_ijkl_org.h"

          en_l += en * el_pair_list[lt].rv;
          ct_l += en / el_pair_list[lt].wgt;
        }
        en_kl += en_l * el_pair_list[kt].rv;
        ct_kl += ct_l / el_pair_list[kt].wgt;
      }
      ct_jkl += ct_kl / el_pair_list[jt].wgt;
      en_jkl += en_kl * el_pair_list[jt].rv;
    }
    control[0] -= ct_jkl / el_pair_list[it].wgt;
    emp4       -= en_jkl * el_pair_list[it].rv;
  }
  */

  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double en_jkl = 0;
    double ct_jkl = 0;
    for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
      if (it == jt) continue;
      auto ij = it * iops.iopns[KEYS::MC_NPAIR] + jt;

      double en_kl = 0;
      double ct_kl = 0;
      for (auto kt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
        if (it == kt || jt == kt) continue;
        auto ik = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jk = jt * iops.iopns[KEYS::MC_NPAIR] + kt;

        double en_l = 0;
        double ct_l = 0;
        for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
          if (it == lt || jt == lt || kt == lt) continue;
          auto il = it * iops.iopns[KEYS::MC_NPAIR] + lt;
          auto jl = jt * iops.iopns[KEYS::MC_NPAIR] + lt;
          auto kl = kt * iops.iopns[KEYS::MC_NPAIR] + lt;

          double en = 0;

// #include "qc_mcmp4_ijkl_ij.h"
#include "qc_mcmp4_ijkl_ik.h"
#include "qc_mcmp4_ijkl_il.h"

          en_l += en * el_pair_list[lt].rv;
          ct_l += en / el_pair_list[lt].wgt;
        }
        en_kl += en_l * el_pair_list[kt].rv;
        ct_kl += ct_l / el_pair_list[kt].wgt;
      }
      ct_jkl += ct_kl / el_pair_list[jt].wgt;
      en_jkl += en_kl * el_pair_list[jt].rv;
    }
    control[0] += ct_jkl / el_pair_list[it].wgt;
    emp4       += en_jkl * el_pair_list[it].rv;
  }


  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double en_i = 0;
    double ct_i = 0;
    for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
      if (it == jt) continue;
      auto ij = it * iops.iopns[KEYS::MC_NPAIR] + jt;
      std::array<double, 36> en_k, ct_k;
      en_k.fill(0); ct_k.fill(0);
      for (auto kt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
        if (it == kt || jt == kt) continue;
        auto ik = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jk = jt * iops.iopns[KEYS::MC_NPAIR] + kt;

        std::array<double, 36> en;
#include "qc_mcmp4_ij_k.h"
        std::transform(en_k.begin(), en_k.end(), en.begin(), en_k.begin(), [&](double x, double y) {return x + y * el_pair_list[kt].rv;});
        std::transform(ct_k.begin(), ct_k.end(), en.begin(), ct_k.begin(), [&](double x, double y) {return x + y / el_pair_list[kt].wgt;});
      }

      std::array<double, 36> en_l, ct_l;
      en_l.fill(0); ct_l.fill(0);
      for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
        if (it == lt || jt == lt) continue;
        auto il = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto jl = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;
#include "qc_mcmp4_ij_l.h"
        std::transform(en_l.begin(), en_l.end(), en.begin(), en_l.begin(), [&](double x, double y) {return x + y * el_pair_list[lt].rv;});
        std::transform(ct_l.begin(), ct_l.end(), en.begin(), ct_l.begin(), [&](double x, double y) {return x + y / el_pair_list[lt].wgt;});
      }

      double en_corr = 0;
      double ct_corr = 0;
      for (auto kt = 0, lt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++, lt++) {
        if (it == kt || jt == kt) continue;
        auto ik = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jk = jt * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto il = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto jl = jt * iops.iopns[KEYS::MC_NPAIR] + lt;
        double en = 0;
#include "qc_mcmp4_ij.h"
        en_corr += en * el_pair_list[kt].rv * el_pair_list[lt].rv;
        ct_corr += en / el_pair_list[kt].wgt / el_pair_list[lt].wgt;
      }

      en_i += std::inner_product(en_k.begin(), en_k.end(), en_l.begin(), -en_corr) * el_pair_list[jt].rv;
      ct_i += std::inner_product(ct_k.begin(), ct_k.end(), ct_l.begin(), -ct_corr) / el_pair_list[jt].wgt;
    }
    emp4 += en_i * el_pair_list[it].rv;
    control[0] += ct_i / el_pair_list[it].wgt;
  }
  
  /*
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double en_i = 0;
    double ct_i = 0;
    for (auto kt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
      if (it == kt) continue;
      auto ik = it * iops.iopns[KEYS::MC_NPAIR] + kt;

      std::array<double, 36> en_j, ct_j;
      en_j.fill(0); ct_j.fill(0);
      for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
        if (it == jt || kt == jt) continue;
        auto ij = it * iops.iopns[KEYS::MC_NPAIR] + jt;
        auto jk = jt * iops.iopns[KEYS::MC_NPAIR] + kt;

        std::array<double, 36> en;
#include "qc_mcmp4_ik_j.h"
        std::transform(en_j.begin(), en_j.end(), en.begin(), en_j.begin(), [&](double x, double y) {return x + y * el_pair_list[jt].rv;});
        std::transform(ct_j.begin(), ct_j.end(), en.begin(), ct_j.begin(), [&](double x, double y) {return x + y / el_pair_list[jt].wgt;});
      }

      std::array<double, 36> en_l, ct_l;
      en_l.fill(0); ct_l.fill(0);
      for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
        if (it == lt || kt == lt) continue;
        auto il = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto kl = kt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;
#include "qc_mcmp4_ik_l.h"
        std::transform(en_l.begin(), en_l.end(), en.begin(), en_l.begin(), [&](double x, double y) {return x + y * el_pair_list[lt].rv;});
        std::transform(ct_l.begin(), ct_l.end(), en.begin(), ct_l.begin(), [&](double x, double y) {return x + y / el_pair_list[lt].wgt;});
      }

      double en_corr = 0;
      double ct_corr = 0;
      for (auto jt = 0, lt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++, lt++) {
        if (it == jt || kt == jt) continue;
        auto ij = it * iops.iopns[KEYS::MC_NPAIR] + jt;
        auto jk = jt * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto il = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto kl = kt * iops.iopns[KEYS::MC_NPAIR] + lt;
        double en = 0;
#include "qc_mcmp4_ik.h"
        en_corr += en * el_pair_list[jt].rv * el_pair_list[lt].rv;
        ct_corr += en / el_pair_list[jt].wgt / el_pair_list[lt].wgt;
      }

      en_i += std::inner_product(en_j.begin(), en_j.end(), en_l.begin(), -en_corr) * el_pair_list[kt].rv;
      ct_i += std::inner_product(ct_j.begin(), ct_j.end(), ct_l.begin(), -ct_corr) / el_pair_list[kt].wgt;
    }
    emp4 += en_i * el_pair_list[it].rv;
    control[0] += ct_i / el_pair_list[it].wgt;
  }
  */

  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double en_jkl = 0;
    double ct_jkl = 0;
    for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
      if (it == jt) continue;
      auto ij = it * iops.iopns[KEYS::MC_NPAIR] + jt;

      double en_kl = 0;
      double ct_kl = 0;
      for (auto kt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
        if (it == kt || jt == kt) continue;
        auto ik = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jk = jt * iops.iopns[KEYS::MC_NPAIR] + kt;

        std::array<double, 117> en_l;
        std::array<double, 117> ct_l;
        en_l.fill(0.0);
        ct_l.fill(0.0);
        for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
          if (it == lt || jt == lt || kt == lt) continue;
          auto il = it * iops.iopns[KEYS::MC_NPAIR] + lt;
          auto jl = jt * iops.iopns[KEYS::MC_NPAIR] + lt;
          auto kl = kt * iops.iopns[KEYS::MC_NPAIR] + lt;

          std::array<double, 117> en;

          #include "qc_mcmp4_ijkl.h"

          std::transform(en_l.begin(), en_l.end(), en.begin(), en_l.begin(), [&](double x, double y){return x + y * el_pair_list[lt].rv;});
          std::transform(ct_l.begin(), ct_l.end(), en.begin(), ct_l.begin(), [&](double x, double y){return x + y / el_pair_list[lt].wgt;});
        }
        double en_l_t = 0;
        double ct_l_t = 0;
#include "qc_mcmp4_ijk.h"
        en_kl += en_l_t * el_pair_list[kt].rv;
        ct_kl += ct_l_t / el_pair_list[kt].wgt;
      }
      en_jkl += en_kl * el_pair_list[jt].rv;
      ct_jkl += ct_kl / el_pair_list[jt].wgt;
    }
    emp4       += en_jkl * el_pair_list[it].rv;
    control[0] += ct_jkl / el_pair_list[it].wgt;
  }

  auto tau_wgt = tau.get_wgt(3);
  std::transform(control.begin(), control.end(), control.begin(),
                 [&](double x) { return x * tau_wgt; });
  emp4 *= tau_wgt;

  auto nsamp = static_cast<double>(iops.iopns[KEYS::MC_NPAIR]);
  nsamp *= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 1);
  nsamp *= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 2);
  nsamp *= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 3);
  emp4 /= nsamp;
  std::transform(control.begin(), control.end(), control.begin(),
                 [nsamp](double x) { return x / nsamp; });
}
