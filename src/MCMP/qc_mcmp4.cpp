//
// Created by aedoran on 6/13/18.
//
#include "../qc_monte.h"

void MP::mcmp4_energy(double& emp4, std::vector<double>& control) {
  emp4 = 0.0;
  std::fill(control.begin(), control.end(), 0.0);

  mcmp4_energy_ij(emp4, control);
  mcmp4_energy_ik(emp4, control);
  mcmp4_energy_il(emp4, control);
  mcmp4_energy_ijkl(emp4, control);

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

void MP::mcmp4_energy_ij(double& emp4, std::vector<double>& control) {
  // ij contracted sums
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double en_i = 0;
    double ct_i = 0;
    for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
      if (it == jt) continue;
      auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
      std::array<double, 36> en_kt, ct_kt;
      en_kt.fill(0); ct_kt.fill(0);
      for (auto kt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
        if (it == kt || jt == kt) continue;
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;

        std::array<double, 36> en;
#include "qc_mcmp4_ij_k.h"
        std::transform(en_kt.begin(), en_kt.end(), en.begin(), en_kt.begin(), [&](double x, double y) {return x + y * el_pair_list[kt].rv;});
        std::transform(ct_kt.begin(), ct_kt.end(), en.begin(), ct_kt.begin(), [&](double x, double y) {return x + y / el_pair_list[kt].wgt;});
      }

      std::array<double, 36> en_lt, ct_lt;
      en_lt.fill(0); ct_lt.fill(0);
      for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
        if (it == lt || jt == lt) continue;
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;
#include "qc_mcmp4_ij_l.h"
        std::transform(en_lt.begin(), en_lt.end(), en.begin(), en_lt.begin(), [&](double x, double y) {return x + y * el_pair_list[lt].rv;});
        std::transform(ct_lt.begin(), ct_lt.end(), en.begin(), ct_lt.begin(), [&](double x, double y) {return x + y / el_pair_list[lt].wgt;});
      }

      double en_corr = 0;
      double ct_corr = 0;
      for (auto kt = 0, lt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++, lt++) {
        if (it == kt || jt == kt) continue;
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;
        double en = 0;
#include "qc_mcmp4_ij.h"
        en_corr += en * el_pair_list[kt].rv * el_pair_list[lt].rv;
        ct_corr += en / el_pair_list[kt].wgt / el_pair_list[lt].wgt;
      }

      en_i += std::inner_product(en_kt.begin(), en_kt.end(), en_lt.begin(), -en_corr) * el_pair_list[jt].rv;
      ct_i += std::inner_product(ct_kt.begin(), ct_kt.end(), ct_lt.begin(), -ct_corr) / el_pair_list[jt].wgt;
    }
    emp4 += en_i * el_pair_list[it].rv;
    control[0] += ct_i / el_pair_list[it].wgt;
  }
}

void MP::mcmp4_energy_ik(double& emp4, std::vector<double>& control) {
  // ik contracted sums
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double en_i = 0;
    double ct_i = 0;
    for (auto kt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
      if (it == kt) continue;
      auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
      std::array<double, 36> en_jt, ct_jt;
      en_jt.fill(0); ct_jt.fill(0);
      for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
        if (it == jt || jt == kt) continue;
        auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;

        std::array<double, 36> en;
#include "qc_mcmp4_ik_j.h"
        std::transform(en_jt.begin(), en_jt.end(), en.begin(), en_jt.begin(), [&](double x, double y) {return x + y * el_pair_list[jt].rv;});
        std::transform(ct_jt.begin(), ct_jt.end(), en.begin(), ct_jt.begin(), [&](double x, double y) {return x + y / el_pair_list[jt].wgt;});
      }

      std::array<double, 36> en_lt, ct_lt;
      en_lt.fill(0); ct_lt.fill(0);
      for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
        if (it == lt || kt == lt) continue;
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto kt_lt = kt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;
#include "qc_mcmp4_ik_l.h"
        std::transform(en_lt.begin(), en_lt.end(), en.begin(), en_lt.begin(), [&](double x, double y) {return x + y * el_pair_list[lt].rv;});
        std::transform(ct_lt.begin(), ct_lt.end(), en.begin(), ct_lt.begin(), [&](double x, double y) {return x + y / el_pair_list[lt].wgt;});
      }

      double en_corr = 0;
      double ct_corr = 0;
      for (auto jt = 0, lt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++, lt++) {
        if (it == jt || kt == jt) continue;
        auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto kt_lt = kt * iops.iopns[KEYS::MC_NPAIR] + lt;
        double en = 0;
#include "qc_mcmp4_ik.h"
        en_corr += en * el_pair_list[jt].rv * el_pair_list[lt].rv;
        ct_corr += en / el_pair_list[jt].wgt / el_pair_list[lt].wgt;
      }

      en_i += std::inner_product(en_jt.begin(), en_jt.end(), en_lt.begin(), -en_corr) * el_pair_list[kt].rv;
      ct_i += std::inner_product(ct_jt.begin(), ct_jt.end(), ct_lt.begin(), -ct_corr) / el_pair_list[kt].wgt;
    }
    emp4 += en_i * el_pair_list[it].rv;
    control[1] += ct_i / el_pair_list[it].wgt;
  }
}

void MP::mcmp4_energy_il(double& emp4, std::vector<double>& control) {
  // il contracted sums
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double en_i = 0;
    double ct_i = 0;
    for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
      if (it == lt) continue;
      auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
      std::array<double, 36> en_kt, ct_kt;
      en_kt.fill(0); ct_kt.fill(0);
      for (auto kt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
        if (it == kt || lt == kt) continue;
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto kt_lt = kt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;
#include "qc_mcmp4_il_k.h"
        std::transform(en_kt.begin(), en_kt.end(), en.begin(), en_kt.begin(), [&](double x, double y) {return x + y * el_pair_list[kt].rv;});
        // std::transform(ct_kt.begin(), ct_kt.end(), en.begin(), ct_kt.begin(), [&](double x, double y) {return x + y / el_pair_list[kt].wgt;});
      }

      std::array<double, 36> en_jt, ct_jt;
      en_jt.fill(0); ct_jt.fill(0);
      for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
        if (it == jt || lt == jt) continue;
        auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;
#include "qc_mcmp4_il_j.h"
        std::transform(en_jt.begin(), en_jt.end(), en.begin(), en_jt.begin(), [&](double x, double y) {return x + y * el_pair_list[jt].rv;});
        // std::transform(ct_jt.begin(), ct_jt.end(), en.begin(), ct_jt.begin(), [&](double x, double y) {return x + y / el_pair_list[jt].wgt;});
      }

      double en_corr = 0;
      double ct_corr = 0;
      for (auto kt = 0, jt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++, jt++) {
        if (it == kt || lt == kt) continue;
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto kt_lt = kt * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;
        double en = 0;
#include "qc_mcmp4_il.h"
        en_corr += en * el_pair_list[kt].rv * el_pair_list[jt].rv;
        // ct_corr += en / el_pair_list[kt].wgt / el_pair_list[jt].wgt;
      }

      en_i += std::inner_product(en_kt.begin(), en_kt.end(), en_jt.begin(), -en_corr) * el_pair_list[lt].rv;
      ct_i += std::inner_product(en_kt.begin(), en_kt.end(), en_jt.begin(), -en_corr) / el_pair_list[lt].wgt;
    }
    emp4 += en_i * el_pair_list[it].rv;
    control[2] += en_i / el_pair_list[it].wgt;
    control[3] += ct_i / el_pair_list[it].wgt;
    control[4] += ct_i * el_pair_list[it].rv;
  }
}

void MP::mcmp4_energy_ijkl(double& emp4, std::vector<double>& control) {
  // fourth order sums
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

        std::array<double, 30> en_l;
        std::array<double, 30> ct_l;
        en_l.fill(0.0);
        ct_l.fill(0.0);
        for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
          if (it == lt || jt == lt || kt == lt) continue;
          auto il = it * iops.iopns[KEYS::MC_NPAIR] + lt;
          auto jl = jt * iops.iopns[KEYS::MC_NPAIR] + lt;
          auto kl = kt * iops.iopns[KEYS::MC_NPAIR] + lt;

          std::array<double, 30> en;

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
    control[3] += ct_jkl / el_pair_list[it].wgt;
  }
}

