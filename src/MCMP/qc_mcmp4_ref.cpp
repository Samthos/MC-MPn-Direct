//
// Created by aedoran on 6/13/18.
//
#include "../qc_monte.h"
#include "cblas.h"

void MCMP4_REF::mcmp4_energy(double& emp4, std::vector<double>& control4) {
  double en4 = 0.0;
  std::vector<double> control4(control.size(), 0.0);
  mcmp4_energy_ij(en4, control4);
  mcmp4_energy_ik(en4, control4);
  mcmp4_energy_il(en4, control4);
  mcmp4_energy_ijkl(en4, control4);

  auto nsamp_tauwgt = tau->get_wgt(3);
  nsamp_tauwgt /= static_cast<double>(iops.iopns[KEYS::MC_NPAIR]);
  nsamp_tauwgt /= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 1);
  nsamp_tauwgt /= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 2);
  nsamp_tauwgt /= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 3);
  emp4 = emp4 + en4 * nsamp_tauwgt;
#if MCMP4_REF4CV >= 1
  std::transform(ctrl.begin(), ctrl.end(), control4.begin(), control4.begin(), [&](double c, double total) { return total + c * nsamp_tauwgt; });
#endif
}

/*
WARNING:
WARNING: Don't delete the four following function.
WARNING: They are a much slower reference implementation of MC-MCMP4_REF4
WARNING: The MCMP4_REF4 class and the following should produce the exact same output 
WARNING:
*/
void MCMP4_REF::mcmp4_energy_ij(double& emp4, std::vector<double>& control) {
  // ij contracted sums
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double jr_kr_lr_corr = 0;
    double jr_kw_lr_corr = 0;
    double jr_kr_lw_corr = 0;
    double jr_kw_lw_corr = 0;
    double jw_kr_lr_corr = 0;
    double jw_kw_lr_corr = 0;
    double jw_kr_lw_corr = 0;
    double jw_kw_lw_corr = 0;
    for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
      if (it == jt) continue;
      auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
      std::array<double, 36> en_kt, ct_kt;
      en_kt.fill(0); ct_kt.fill(0);
      for (auto kt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;

        std::array<double, 36> en;

        #include "qc_mcmp4_ij_k.h"

        std::transform(en_kt.begin(), en_kt.end(), en.begin(), en_kt.begin(), [&](double x, double y) {return x + y * el_pair_list->get(kt).rv;});
        std::transform(ct_kt.begin(), ct_kt.end(), en.begin(), ct_kt.begin(), [&](double x, double y) {return x + y / el_pair_list->get(kt).wgt;});
      }

      std::array<double, 36> en_lt, ct_lt;
      en_lt.fill(0); ct_lt.fill(0);
      for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;

        #include "qc_mcmp4_ij_l.h"

        std::transform(en_lt.begin(), en_lt.end(), en.begin(), en_lt.begin(), [&](double x, double y) {return x + y * el_pair_list->get(lt).rv;});
        std::transform(ct_lt.begin(), ct_lt.end(), en.begin(), ct_lt.begin(), [&](double x, double y) {return x + y / el_pair_list->get(lt).wgt;});
      }

      double kr_lr_corr = 0;
      double kr_lw_corr = 0;
      double kw_lr_corr = 0;
      double kw_lw_corr = 0;
      for (auto kt = 0, lt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++, lt++) {
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        double en = 0;

        #include "qc_mcmp4_ij.h"

        kr_lr_corr += en * el_pair_list->get(kt).rv * el_pair_list->get(lt).rv;
        kr_lw_corr += en * el_pair_list->get(kt).rv / el_pair_list->get(lt).wgt;
        kw_lr_corr += en / el_pair_list->get(kt).wgt * el_pair_list->get(lt).rv;
        kw_lw_corr += en / el_pair_list->get(kt).wgt / el_pair_list->get(lt).wgt;
      }

      jr_kr_lr_corr += std::inner_product(en_kt.begin(), en_kt.end(), en_lt.begin(), -kr_lr_corr) * el_pair_list->get(jt).rv;
      jr_kw_lr_corr += std::inner_product(ct_kt.begin(), ct_kt.end(), en_lt.begin(), -kw_lr_corr) * el_pair_list->get(jt).rv;
      jw_kr_lr_corr += std::inner_product(en_kt.begin(), en_kt.end(), en_lt.begin(), -kr_lr_corr) / el_pair_list->get(jt).wgt;
      jw_kw_lr_corr += std::inner_product(ct_kt.begin(), ct_kt.end(), en_lt.begin(), -kw_lr_corr) / el_pair_list->get(jt).wgt;
      jr_kr_lw_corr += std::inner_product(en_kt.begin(), en_kt.end(), ct_lt.begin(), -kr_lw_corr) * el_pair_list->get(jt).rv;
      jr_kw_lw_corr += std::inner_product(ct_kt.begin(), ct_kt.end(), ct_lt.begin(), -kw_lw_corr) * el_pair_list->get(jt).rv;
      jw_kr_lw_corr += std::inner_product(en_kt.begin(), en_kt.end(), ct_lt.begin(), -kr_lw_corr) / el_pair_list->get(jt).wgt;
      jw_kw_lw_corr += std::inner_product(ct_kt.begin(), ct_kt.end(), ct_lt.begin(), -kw_lw_corr) / el_pair_list->get(jt).wgt;
    }
    emp4       += jr_kr_lr_corr * el_pair_list->get(it).rv;
    control[0]  += jr_kr_lr_corr / el_pair_list->get(it).wgt;
    control[1]  += jw_kr_lr_corr / el_pair_list->get(it).wgt;
    control[2]  += jr_kr_lw_corr * el_pair_list->get(it).rv;
    control[3]  += jr_kr_lw_corr / el_pair_list->get(it).wgt;
    control[4]  += jw_kr_lw_corr * el_pair_list->get(it).rv;
    control[5]  += jw_kr_lw_corr / el_pair_list->get(it).wgt;
    control[6]  += jr_kw_lr_corr / el_pair_list->get(it).wgt;
    control[7]  += jw_kw_lr_corr / el_pair_list->get(it).wgt;
    control[8]  += jr_kw_lw_corr * el_pair_list->get(it).rv;
    control[9]  += jr_kw_lw_corr / el_pair_list->get(it).wgt;
    control[10] += jw_kw_lw_corr * el_pair_list->get(it).rv;
    control[11] += jw_kw_lw_corr / el_pair_list->get(it).wgt;
  }
}
void MCMP4_REF::mcmp4_energy_ik(double& emp4, std::vector<double>& control) {
  // ik contracted sums
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double jr_kr_lr_corr = 0;
    double jr_kw_lr_corr = 0;
    double jr_kr_lw_corr = 0;
    double jr_kw_lw_corr = 0;
    double jw_kr_lr_corr = 0;
    double jw_kw_lr_corr = 0;
    double jw_kr_lw_corr = 0;
    double jw_kw_lw_corr = 0;
    for (auto kt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
      if (it == kt) continue;
      auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
      std::array<double, 36> en_jt, ct_jt;
      en_jt.fill(0); ct_jt.fill(0);
      for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
        auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;

        std::array<double, 36> en;

#include "qc_mcmp4_ik_j.h"

        std::transform(en_jt.begin(), en_jt.end(), en.begin(), en_jt.begin(), [&](double x, double y) {return x + y * el_pair_list->get(jt).rv;});
        std::transform(ct_jt.begin(), ct_jt.end(), en.begin(), ct_jt.begin(), [&](double x, double y) {return x + y / el_pair_list->get(jt).wgt;});
      }

      std::array<double, 36> en_lt, ct_lt;
      en_lt.fill(0); ct_lt.fill(0);
      for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto kt_lt = kt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;

#include "qc_mcmp4_ik_l.h"

        std::transform(en_lt.begin(), en_lt.end(), en.begin(), en_lt.begin(), [&](double x, double y) {return x + y * el_pair_list->get(lt).rv;});
        std::transform(ct_lt.begin(), ct_lt.end(), en.begin(), ct_lt.begin(), [&](double x, double y) {return x + y / el_pair_list->get(lt).wgt;});
      }

      double jr_lr_corr = 0;
      double jr_lw_corr = 0;
      double jw_lr_corr = 0;
      double jw_lw_corr = 0;
      for (auto jt = 0, lt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++, lt++) {
        auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto kt_lt = kt * iops.iopns[KEYS::MC_NPAIR] + lt;

        double en = 0;

#include "qc_mcmp4_ik.h"

        jr_lr_corr += en * el_pair_list->get(jt).rv * el_pair_list->get(lt).rv;
        jr_lw_corr += en * el_pair_list->get(jt).rv / el_pair_list->get(lt).wgt;
        jw_lr_corr += en / el_pair_list->get(jt).wgt * el_pair_list->get(lt).rv;
        jw_lw_corr += en / el_pair_list->get(jt).wgt / el_pair_list->get(lt).wgt;
      }

      jr_kr_lr_corr += std::inner_product(en_jt.begin(), en_jt.end(), en_lt.begin(), -jr_lr_corr) * el_pair_list->get(kt).rv;
      jr_kr_lw_corr += std::inner_product(en_jt.begin(), en_jt.end(), ct_lt.begin(), -jr_lw_corr) * el_pair_list->get(kt).rv;
      jw_kr_lr_corr += std::inner_product(ct_jt.begin(), ct_jt.end(), en_lt.begin(), -jw_lr_corr) * el_pair_list->get(kt).rv;
      jw_kr_lw_corr += std::inner_product(ct_jt.begin(), ct_jt.end(), ct_lt.begin(), -jw_lw_corr) * el_pair_list->get(kt).rv;
      jr_kw_lr_corr += std::inner_product(en_jt.begin(), en_jt.end(), en_lt.begin(), -jr_lr_corr) / el_pair_list->get(kt).wgt;
      jr_kw_lw_corr += std::inner_product(en_jt.begin(), en_jt.end(), ct_lt.begin(), -jr_lw_corr) / el_pair_list->get(kt).wgt;
      jw_kw_lr_corr += std::inner_product(ct_jt.begin(), ct_jt.end(), en_lt.begin(), -jw_lr_corr) / el_pair_list->get(kt).wgt;
      jw_kw_lw_corr += std::inner_product(ct_jt.begin(), ct_jt.end(), ct_lt.begin(), -jw_lw_corr) / el_pair_list->get(kt).wgt;
    }
    emp4        += jr_kr_lr_corr * el_pair_list->get(it).rv;
    control[12] += jr_kr_lr_corr / el_pair_list->get(it).wgt;
    control[13] += jr_kw_lr_corr / el_pair_list->get(it).wgt;
    control[14] += jr_kr_lw_corr * el_pair_list->get(it).rv;
    control[15] += jr_kr_lw_corr / el_pair_list->get(it).wgt;
    control[16] += jr_kw_lw_corr * el_pair_list->get(it).rv;
    control[17] += jr_kw_lw_corr / el_pair_list->get(it).wgt;
    control[18] += jw_kr_lr_corr / el_pair_list->get(it).wgt;
    control[19] += jw_kw_lr_corr / el_pair_list->get(it).wgt;
    control[20] += jw_kr_lw_corr * el_pair_list->get(it).rv;
    control[21] += jw_kr_lw_corr / el_pair_list->get(it).wgt;
    control[22] += jw_kw_lw_corr * el_pair_list->get(it).rv;
    control[23] += jw_kw_lw_corr / el_pair_list->get(it).wgt;
  }
}
void MCMP4_REF::mcmp4_energy_il(double& emp4, std::vector<double>& control) {
  // il contracted sums
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    double jr_kr_lr_corr = 0;
    double jr_kw_lr_corr = 0;
    double jr_kr_lw_corr = 0;
    double jr_kw_lw_corr = 0;
    double jw_kr_lr_corr = 0;
    double jw_kw_lr_corr = 0;
    double jw_kr_lw_corr = 0;
    double jw_kw_lw_corr = 0;
    for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
      if (it == lt) continue;
      auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
      std::array<double, 36> en_kt, ct_kt;
      en_kt.fill(0); ct_kt.fill(0);
      for (auto kt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto kt_lt = kt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;

#include "qc_mcmp4_il_k.h"

        std::transform(en_kt.begin(), en_kt.end(), en.begin(), en_kt.begin(), [&](double x, double y) {return x + y * el_pair_list->get(kt).rv;});
        std::transform(ct_kt.begin(), ct_kt.end(), en.begin(), ct_kt.begin(), [&](double x, double y) {return x + y / el_pair_list->get(kt).wgt;});
      }

      std::array<double, 36> en_jt, ct_jt;
      en_jt.fill(0); ct_jt.fill(0);
      for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
        auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;

#include "qc_mcmp4_il_j.h"

        std::transform(en_jt.begin(), en_jt.end(), en.begin(), en_jt.begin(), [&](double x, double y) {return x + y * el_pair_list->get(jt).rv;});
        std::transform(ct_jt.begin(), ct_jt.end(), en.begin(), ct_jt.begin(), [&](double x, double y) {return x + y / el_pair_list->get(jt).wgt;});
      }

      double jr_kr_corr = 0;
      double jr_kw_corr = 0;
      double jw_kr_corr = 0;
      double jw_kw_corr = 0;
      for (auto kt = 0, jt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++, jt++) {
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto kt_lt = kt * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        double en = 0;

#include "qc_mcmp4_il.h"

        jr_kr_corr += en * el_pair_list->get(jt).rv  * el_pair_list->get(kt).rv;
        jr_kw_corr += en * el_pair_list->get(jt).rv  / el_pair_list->get(kt).wgt;
        jw_kr_corr += en / el_pair_list->get(jt).wgt * el_pair_list->get(kt).rv;
        jw_kw_corr += en / el_pair_list->get(jt).wgt / el_pair_list->get(kt).wgt;
      }

      jr_kr_lr_corr += std::inner_product(en_jt.begin(), en_jt.end(), en_kt.begin(), -jr_kr_corr) * el_pair_list->get(lt).rv;
      jr_kr_lw_corr += std::inner_product(en_jt.begin(), en_jt.end(), en_kt.begin(), -jr_kr_corr) / el_pair_list->get(lt).wgt;
      jr_kw_lr_corr += std::inner_product(en_jt.begin(), en_jt.end(), ct_kt.begin(), -jr_kw_corr) * el_pair_list->get(lt).rv;
      jr_kw_lw_corr += std::inner_product(en_jt.begin(), en_jt.end(), ct_kt.begin(), -jr_kw_corr) / el_pair_list->get(lt).wgt;
      jw_kw_lr_corr += std::inner_product(ct_jt.begin(), ct_jt.end(), ct_kt.begin(), -jw_kw_corr) * el_pair_list->get(lt).rv;
      jw_kw_lw_corr += std::inner_product(ct_jt.begin(), ct_jt.end(), ct_kt.begin(), -jw_kw_corr) / el_pair_list->get(lt).wgt;
      jw_kr_lr_corr += std::inner_product(ct_jt.begin(), ct_jt.end(), en_kt.begin(), -jw_kr_corr) * el_pair_list->get(lt).rv;
      jw_kr_lw_corr += std::inner_product(ct_jt.begin(), ct_jt.end(), en_kt.begin(), -jw_kr_corr) / el_pair_list->get(lt).wgt;
    }
    emp4       += jr_kr_lr_corr * el_pair_list->get(it).rv;
    control[24] += jr_kr_lr_corr / el_pair_list->get(it).wgt;
    control[25] += jr_kr_lw_corr / el_pair_list->get(it).wgt;
    control[26] += jr_kr_lw_corr * el_pair_list->get(it).rv;
    control[27] += jw_kr_lr_corr / el_pair_list->get(it).wgt;
    control[28] += jw_kr_lw_corr / el_pair_list->get(it).wgt;
    control[29] += jw_kr_lw_corr * el_pair_list->get(it).rv;
    control[30] += jr_kw_lr_corr / el_pair_list->get(it).wgt;
    control[31] += jr_kw_lw_corr / el_pair_list->get(it).wgt;
    control[32] += jr_kw_lw_corr * el_pair_list->get(it).rv;
    control[33] += jw_kw_lr_corr / el_pair_list->get(it).wgt;
    control[34] += jw_kw_lw_corr / el_pair_list->get(it).wgt;
    control[35] += jw_kw_lw_corr * el_pair_list->get(it).rv;
  }
}
void MCMP4_REF::mcmp4_energy_ijkl(double& emp4, std::vector<double>& control) {
  // fourth order sums
  for (auto it = 0; it < iops.iopns[KEYS::MC_NPAIR]; it++) {
    std::array<double, 3> jr_kr_lr; jr_kr_lr.fill(0.0);
    std::array<double, 3> jr_kr_lw; jr_kr_lw.fill(0.0);
    std::array<double, 3> jr_kw_lr; jr_kw_lr.fill(0.0);
    std::array<double, 3> jr_kw_lw; jr_kw_lw.fill(0.0);
    std::array<double, 3> jw_kr_lr; jw_kr_lr.fill(0.0);
    std::array<double, 3> jw_kr_lw; jw_kr_lw.fill(0.0);
    std::array<double, 3> jw_kw_lr; jw_kw_lr.fill(0.0);
    std::array<double, 3> jw_kw_lw; jw_kw_lw.fill(0.0);
    for (auto jt = 0; jt < iops.iopns[KEYS::MC_NPAIR]; jt++) {
      auto ij = it * iops.iopns[KEYS::MC_NPAIR] + jt;

      std::array<double, 3> kr_lr; kr_lr.fill(0.0);
      std::array<double, 3> kr_lw; kr_lw.fill(0.0);
      std::array<double, 3> kw_lr; kw_lr.fill(0.0);
      std::array<double, 3> kw_lw; kw_lw.fill(0.0);
      for (auto kt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
        auto ik = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jk = jt * iops.iopns[KEYS::MC_NPAIR] + kt;

        std::array<double, 30> en_l;
        std::array<double, 30> ct_l;
        en_l.fill(0.0);
        ct_l.fill(0.0);
        for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
          auto il = it * iops.iopns[KEYS::MC_NPAIR] + lt;
          auto jl = jt * iops.iopns[KEYS::MC_NPAIR] + lt;
          auto kl = kt * iops.iopns[KEYS::MC_NPAIR] + lt;

          std::array<double, 30> en;

          #include "qc_mcmp4_ijkl.h"

          std::transform(en_l.begin(), en_l.end(), en.begin(), en_l.begin(), [&](double x, double y){return x + y * el_pair_list->get(lt).rv;});
          std::transform(ct_l.begin(), ct_l.end(), en.begin(), ct_l.begin(), [&](double x, double y){return x + y / el_pair_list->get(lt).wgt;});
        }
        std::array<double, 3> en_l_t; en_l_t.fill(0.0);
        std::array<double, 3> ct_l_t; ct_l_t.fill(0.0);
#include "qc_mcmp4_ijk.h"
        for (int group = 0; group < 3; group++) {
          kr_lr[group] += en_l_t[group] * el_pair_list->get(kt).rv;
          kr_lw[group] += ct_l_t[group] * el_pair_list->get(kt).rv;
          kw_lr[group] += en_l_t[group] / el_pair_list->get(kt).wgt;
          kw_lw[group] += ct_l_t[group] / el_pair_list->get(kt).wgt;
        }
      }
      for (int group = 0; group < 3; group++) {
        jr_kr_lr[group] += kr_lr[group] * el_pair_list->get(jt).rv;
        jr_kr_lw[group] += kr_lw[group] * el_pair_list->get(jt).rv;
        jr_kw_lr[group] += kw_lr[group] * el_pair_list->get(jt).rv;
        jr_kw_lw[group] += kw_lw[group] * el_pair_list->get(jt).rv;
        jw_kr_lr[group] += kr_lr[group] / el_pair_list->get(jt).wgt;
        jw_kr_lw[group] += kr_lw[group] / el_pair_list->get(jt).wgt;
        jw_kw_lr[group] += kw_lr[group] / el_pair_list->get(jt).wgt;
        jw_kw_lw[group] += kw_lw[group] / el_pair_list->get(jt).wgt;
      }
    }
    for (int group = 0; group < 3; group++) {
      emp4             += jr_kr_lr[group] * el_pair_list->get(it).rv;
      control[36 + 12 * group + 0] += jr_kr_lw[group] * el_pair_list->get(it).rv;
      control[36 + 12 * group + 1] += jw_kr_lw[group] * el_pair_list->get(it).rv;
      control[36 + 12 * group + 2] += jr_kw_lw[group] * el_pair_list->get(it).rv;
      control[36 + 12 * group + 3] += jw_kw_lw[group] * el_pair_list->get(it).rv;
      control[36 + 12 * group + 4] += jr_kr_lr[group] / el_pair_list->get(it).wgt;
      control[36 + 12 * group + 5] += jr_kr_lw[group] / el_pair_list->get(it).wgt;
      control[36 + 12 * group + 6] += jr_kw_lr[group] / el_pair_list->get(it).wgt;
      control[36 + 12 * group + 7] += jw_kr_lw[group] / el_pair_list->get(it).wgt;
      control[36 + 12 * group + 8] += jw_kr_lr[group] / el_pair_list->get(it).wgt;
      control[36 + 12 * group + 9] += jr_kw_lw[group] / el_pair_list->get(it).wgt;
      control[36 + 12 * group +10] += jw_kw_lr[group] / el_pair_list->get(it).wgt;
      control[36 + 12 * group +11] += jw_kw_lw[group] / el_pair_list->get(it).wgt;
    }
  }
}
