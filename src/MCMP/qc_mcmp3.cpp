//
// Created by aedoran on 6/1/18.
//

#include <algorithm>
#include <vector>
#include <array>

#include "cblas.h"
#include "../blas_calls.h"
#include "../qc_monte.h"

void Prep_IK(std::vector<double>& A_ij,
    std::vector<double>& A_jk,
    std::vector<double>& rv,
    std::vector<double>& A_ik,
    int mc_pair_num) {
  for (int tidx = 0; tidx < mc_pair_num; ++tidx) {
    for (int tidy = 0; tidy < mc_pair_num; ++tidy) {
      int index = tidx * mc_pair_num + tidy;
      A_ik[index] = 0;
      A_ik[index] -= A_ij[tidx*mc_pair_num + tidx] * A_jk[tidx*mc_pair_num + tidy] * rv[tidx];
      A_ik[index] -= A_ij[tidx*mc_pair_num + tidy] * A_jk[tidy*mc_pair_num + tidy] * rv[tidy];
    }
  }
}
double mcmp3_helper(unsigned int mc_pair_num, double constant,
    std::vector<double>& A_ij_1, std::vector<double>& A_ij_2,
    std::vector<double>& A_ik_1, std::vector<double>& A_ik_2,
    std::vector<double>& A_jk_1, std::vector<double>& A_jk_2,
    std::vector<double>& rv) {
  std::vector<double> A_ij(mc_pair_num * mc_pair_num);
  std::vector<double> A_ik(mc_pair_num * mc_pair_num);
  std::vector<double> A_jk(mc_pair_num * mc_pair_num);

  std::transform(A_ij_1.begin(), A_ij_1.end(), A_ij_2.begin(), A_ij.begin(), std::multiplies<>());
  std::transform(A_jk_1.begin(), A_jk_1.end(), A_jk_2.begin(), A_jk.begin(), std::multiplies<>());

  Prep_IK(A_ij, A_jk, rv, A_ik, mc_pair_num);


  Ddgmm(DDGMM_SIDE_RIGHT,
      mc_pair_num, mc_pair_num,
      A_jk.data(), mc_pair_num,
      rv.data(), 1,
      A_jk.data(), mc_pair_num);

  cblas_dgemm(CblasColMajor,
      CblasNoTrans, CblasNoTrans,
      mc_pair_num, mc_pair_num, mc_pair_num,
      1.0,
      A_jk.data(), mc_pair_num,
      A_ij.data(), mc_pair_num,
      1.0,
      A_ik.data(), mc_pair_num);

  std::transform(A_ik.begin(), A_ik.end(), A_ik_1.begin(), A_ik.begin(), std::multiplies<>());
  std::transform(A_ik.begin(), A_ik.end(), A_ik_2.begin(), A_ik.begin(), std::multiplies<>());

  for (int i = 0; i < mc_pair_num * mc_pair_num; i += mc_pair_num+1) {
    A_ik[i] = 0;
  }

  cblas_dgemv(CblasColMajor,
      CblasTrans,
      mc_pair_num, mc_pair_num,
      1.0,
      A_ik.data(), mc_pair_num,
      rv.data(), 1,
      0.0,
      A_ij.data(), 1);
  auto en = constant * std::inner_product(rv.begin(), rv.end(), A_ij.begin(), 0.0);

  /*
  for (auto i = 0; i < mc_pair_num; i++) {
    for (auto k = 0; k < mc_pair_num; k++) {
      auto ik = i * mc_pair_num + k;
      printf("%12.6f", A_ik[ik] * 1000);
    }
    printf("\n");
  }
  */

  //printf("%12.6f\n", en);
  return en;
}

double mcmp3_ref(unsigned int mc_pair_num, double constant,
    std::vector<double>& A_ij_1, std::vector<double>& A_ij_2,
    std::vector<double>& A_ik_1, std::vector<double>& A_ik_2,
    std::vector<double>& A_jk_1, std::vector<double>& A_jk_2,
    std::vector<double>& rv)  {
  std::vector<double> A_ij(mc_pair_num * mc_pair_num);
  std::vector<double> A_ik(mc_pair_num * mc_pair_num);
  std::vector<double> A_jk(mc_pair_num * mc_pair_num);

  std::transform(A_ij_1.begin(), A_ij_1.end(), A_ij_2.begin(), A_ij.begin(), std::multiplies<>());
  std::transform(A_ik_1.begin(), A_ik_1.end(), A_ik_2.begin(), A_ik.begin(), std::multiplies<>());
  std::transform(A_jk_1.begin(), A_jk_1.end(), A_jk_2.begin(), A_jk.begin(), std::multiplies<>());

  double emp3 = 0.0;
  double en_jk, en_k, en;

  for (auto i = 0; i < mc_pair_num; i++) {
    en_jk = 0;
    for (auto k = 0; k < mc_pair_num; k++) {
      auto ik = i * mc_pair_num + k;

      en_k = 0;

      for (auto j = 0; j < mc_pair_num; j++) {
        auto ij = i * mc_pair_num + j;
        auto jk = j * mc_pair_num + k;
        en = A_ij[ij] * A_jk[jk] * rv[j];
        en_k += en;
      }
      {
        auto j = i;
        auto ij = i * mc_pair_num + j;
        auto jk = j * mc_pair_num + k;
        en = A_ij[ij] * A_jk[jk] * rv[j];
        en_k -= en;
      }
      {
        auto j = k;
        auto ij = i * mc_pair_num + j;
        auto jk = j * mc_pair_num + k;
        en = A_ij[ij] * A_jk[jk] * rv[j];
        en_k -= en;
      }
//      printf("%12.6f", en_k * 1000);
      en_jk += en_k * A_ik[ik] * rv[k] * (i != k);
    }
//    printf("\n");
    emp3 += en_jk * rv[i];
  }
  emp3 *= constant;
//  printf("%12.6f\n", emp3);
  return emp3;
}

void MP::mcmp3_energy(double& emp3, std::vector<double>& control) {
  std::array<double, 6> en;

  double en_jk;
  std::array<double, 6> en_k;
  std::array<double, 6> c_k;
  std::array<double, 12> c_jk;

  emp3 = 0.0;
  std::fill(control.begin(), control.end(), 0.0);

  std::vector<double> rv(iops.iopns[KEYS::MC_NPAIR]);
  std::vector<double> wgt(iops.iopns[KEYS::MC_NPAIR]);
  std::transform(el_pair_list.begin(), el_pair_list.end(), rv.begin(), [](el_pair_typ ept){return ept.rv;});
  std::transform(el_pair_list.begin(), el_pair_list.end(), wgt.begin(), [](el_pair_typ ept){return ept.wgt;});

  emp3 += mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], 2, ovps.v_set[0][0].s_11, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, rv);
  emp3 += mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], 2, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11, rv);
  emp3 += mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], 8, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_11, rv);
  emp3 += mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], 2, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_11, rv);
  emp3 += mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], 2, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11, rv);
  emp3 += mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], 2, ovps.o_set[0][0].s_21, ovps.o_set[0][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22, rv);

  emp3 += mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], -1, ovps.v_set[0][0].s_12, ovps.v_set[0][0].s_21, ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, rv);
  emp3 += mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], -4, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_11, rv);
  emp3 += mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], -4, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11, rv);
  emp3 += mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], -4, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11, rv);
  emp3 += mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], -4, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_11, rv);
  emp3 += mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], -1, ovps.o_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22, rv);

  /*
  emp3 -= mcmp3_ref(iops.iopns[KEYS::MC_NPAIR], 2, ovps.v_set[0][0].s_11, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, rv);
  emp3 -= mcmp3_ref(iops.iopns[KEYS::MC_NPAIR], 2, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11, rv);
  emp3 -= mcmp3_ref(iops.iopns[KEYS::MC_NPAIR], 8, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_11, rv);
  emp3 -= mcmp3_ref(iops.iopns[KEYS::MC_NPAIR], 2, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_11, rv);
  emp3 -= mcmp3_ref(iops.iopns[KEYS::MC_NPAIR], 2, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11, rv);
  emp3 -= mcmp3_ref(iops.iopns[KEYS::MC_NPAIR], 2, ovps.o_set[0][0].s_21, ovps.o_set[0][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22, rv);

  emp3 -= mcmp3_ref(iops.iopns[KEYS::MC_NPAIR], -1, ovps.v_set[0][0].s_12, ovps.v_set[0][0].s_21, ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, rv);
  emp3 -= mcmp3_ref(iops.iopns[KEYS::MC_NPAIR], -4, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_11, rv);
  emp3 -= mcmp3_ref(iops.iopns[KEYS::MC_NPAIR], -4, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11, rv);
  emp3 -= mcmp3_ref(iops.iopns[KEYS::MC_NPAIR], -4, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11, rv);
  emp3 -= mcmp3_ref(iops.iopns[KEYS::MC_NPAIR], -4, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_11, rv);
  emp3 -= mcmp3_ref(iops.iopns[KEYS::MC_NPAIR], -1, ovps.o_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22, rv);
  */

  for (auto i = 0; i < iops.iopns[KEYS::MC_NPAIR]; i++) {
    en_jk = 0;
    c_jk.fill(0);
    for (auto j = 0; j < iops.iopns[KEYS::MC_NPAIR]; j++) {
      if (i == j) continue;
      auto ij = i * iops.iopns[KEYS::MC_NPAIR] + j;

      en_k.fill(0);
      c_k.fill(0);

      for (auto k = 0; k < iops.iopns[KEYS::MC_NPAIR]; k++) {
        if (i == k || j == k) continue;
        auto ik = i * iops.iopns[KEYS::MC_NPAIR] + k;
        auto jk = j * iops.iopns[KEYS::MC_NPAIR] + k;

        en.fill(0.0);
        //en[0] = (2 * ovps.v_set[0][0].s_11[ij] * ovps.v_set[0][0].s_22[ij] - 1 * ovps.v_set[0][0].s_12[ij] * ovps.v_set[0][0].s_21[ij]) * ovps.o_set[1][0].s_11[ik] * ovps.o_set[1][0].s_22[ik] * ovps.v_set[1][1].s_11[jk] * ovps.v_set[1][1].s_22[jk];
        //en[1] = (2 * ovps.o_set[0][0].s_21[ij] * ovps.v_set[1][1].s_22[jk] - 4 * ovps.o_set[0][0].s_22[ij] * ovps.v_set[1][1].s_12[jk]) * ovps.v_set[0][0].s_22[ij] * ovps.o_set[1][0].s_12[ik] * ovps.v_set[1][0].s_11[ik] * ovps.o_set[1][1].s_11[jk];
        //en[2] = (8 * ovps.o_set[0][0].s_22[ij] * ovps.v_set[1][1].s_11[jk] - 4 * ovps.o_set[0][0].s_21[ij] * ovps.v_set[1][1].s_21[jk]) * ovps.v_set[0][0].s_22[ij] * ovps.o_set[1][0].s_12[ik] * ovps.v_set[1][0].s_12[ik] * ovps.o_set[1][1].s_11[jk];
        //en[3] = (2 * ovps.o_set[0][0].s_22[ij] * ovps.v_set[1][1].s_12[jk] - 4 * ovps.o_set[0][0].s_21[ij] * ovps.v_set[1][1].s_22[jk]) * ovps.v_set[0][0].s_12[ij] * ovps.o_set[1][0].s_12[ik] * ovps.v_set[1][0].s_21[ik] * ovps.o_set[1][1].s_11[jk];
        //en[4] = (2 * ovps.o_set[0][0].s_21[ij] * ovps.v_set[1][1].s_21[jk] - 4 * ovps.o_set[0][0].s_22[ij] * ovps.v_set[1][1].s_11[jk]) * ovps.v_set[0][0].s_12[ij] * ovps.o_set[1][0].s_12[ik] * ovps.v_set[1][0].s_22[ik] * ovps.o_set[1][1].s_11[jk];
        //en[5] = (2 * ovps.o_set[0][0].s_21[ij] * ovps.o_set[0][0].s_12[ij] - 1 * ovps.o_set[0][0].s_11[ij] * ovps.o_set[0][0].s_22[ij]) * ovps.v_set[1][0].s_12[ik] * ovps.v_set[1][0].s_21[ik] * ovps.o_set[1][1].s_11[jk] * ovps.o_set[1][1].s_22[jk];

        std::transform(en_k.begin(), en_k.end(), en.begin(), en_k.begin(), [&](double x, double y) { return x + y * el_pair_list[k].rv; });
        std::transform(c_k.begin(), c_k.end(), en.begin(), c_k.begin(), [&](double x, double y) { return x + y / el_pair_list[k].wgt; });
      }
      std::transform(c_jk.begin(), c_jk.begin()+6, en_k.begin(), c_jk.begin(), [&](double x, double y) { return x + y * el_pair_list[j].rv; });
      std::transform(c_jk.begin()+6, c_jk.end(), c_k.begin(), c_jk.begin()+6, [&](double x, double y) { return x + y * el_pair_list[j].rv; });
      en_jk += std::accumulate(en_k.begin(), en_k.end(), 0.0) * el_pair_list[j].rv;
    }
    std::transform(control.begin(), control.begin()+12, c_jk.begin(), control.begin(), [&](double x, double y) { return x + y / el_pair_list[i].wgt; });
    std::transform(control.begin()+12, control.end(), c_jk.begin()+6, control.begin()+12, [&](double x, double y) { return x + y * el_pair_list[i].rv; });
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
  emp3 = emp3 / nsamp;
  std::transform(control.begin(), control.end(), control.begin(),
                 [nsamp](double x) { return x / nsamp; });
}
