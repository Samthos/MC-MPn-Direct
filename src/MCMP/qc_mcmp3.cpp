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
std::array<double, 7> mcmp3_helper(unsigned int mc_pair_num, double constant,
    std::vector<double>& A_ij_1, std::vector<double>& A_ij_2,
    std::vector<double>& A_ik_1, std::vector<double>& A_ik_2,
    std::vector<double>& A_jk_1, std::vector<double>& A_jk_2,
    std::vector<double>& rv, std::vector<double>& wgt) {
  std::array<double, 7> en;
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
      A_jk.data(), 1);
  en[0] = constant * std::inner_product(rv.begin(), rv.end(), A_jk.begin(), 0.0);
  en[1] = constant * std::inner_product(wgt.begin(), wgt.end(), A_jk.begin(), 0.0);

  cblas_dgemv(CblasColMajor,
      CblasTrans,
      mc_pair_num, mc_pair_num,
      1.0,
      A_ik.data(), mc_pair_num,
      wgt.data(), 1,
      0.0,
      A_jk.data(), 1);
  en[2] = constant * std::inner_product(rv.begin(), rv.end(), A_jk.begin(), 0.0);
  en[3] = constant * std::inner_product(wgt.begin(), wgt.end(), A_jk.begin(), 0.0);



  std::transform(A_jk_1.begin(), A_jk_1.end(), A_jk_2.begin(), A_jk.begin(), std::multiplies<>());
  Prep_IK(A_ij, A_jk, wgt, A_ik, mc_pair_num);

  Ddgmm(DDGMM_SIDE_RIGHT,
      mc_pair_num, mc_pair_num,
      A_jk.data(), mc_pair_num,
      wgt.data(), 1,
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
      A_jk.data(), 1);
  en[4] = constant * std::inner_product(wgt.begin(), wgt.end(), A_jk.begin(), 0.0);

  cblas_dgemv(CblasColMajor,
      CblasTrans,
      mc_pair_num, mc_pair_num,
      1.0,
      A_ik.data(), mc_pair_num,
      wgt.data(), 1,
      0.0,
      A_jk.data(), 1);
  en[5] = constant * std::inner_product(wgt.begin(), wgt.end(), A_jk.begin(), 0.0);
  en[6] = constant * std::inner_product(rv.begin(), rv.end(), A_jk.begin(), 0.0);
  return en;
}

void MP::mcmp3_energy(double& emp3, std::vector<double>& control) {
  std::array<double, 7> en3;
  emp3 = 0.0;
  std::fill(control.begin(), control.end(), 0.0);

  std::vector<double> rv(iops.iopns[KEYS::MC_NPAIR]);
  std::vector<double> wgt(iops.iopns[KEYS::MC_NPAIR]);
  std::transform(el_pair_list.begin(), el_pair_list.end(), rv.begin(), [](el_pair_typ ept){return ept.rv;});
  std::transform(el_pair_list.begin(), el_pair_list.end(), wgt.begin(), [](el_pair_typ ept){return 1.0/ept.wgt;});

  auto ctrl = control.begin();
  en3 = mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], 2, ovps.v_set[0][0].s_11, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, rv, wgt);
  emp3 += en3[0]; std::transform(en3.begin()+1, en3.end(), ctrl, ctrl, std::plus<>()); ctrl += en3.size() -1;
  en3 = mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], 2, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11, rv, wgt);
  emp3 += en3[0]; std::transform(en3.begin()+1, en3.end(), ctrl, ctrl, std::plus<>()); ctrl += en3.size() -1;
  en3 = mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], 8, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_11, rv, wgt);
  emp3 += en3[0]; std::transform(en3.begin()+1, en3.end(), ctrl, ctrl, std::plus<>()); ctrl += en3.size() -1;
  en3 = mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], 2, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_11, rv, wgt);
  emp3 += en3[0]; std::transform(en3.begin()+1, en3.end(), ctrl, ctrl, std::plus<>()); ctrl += en3.size() -1;
  en3 = mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], 2, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11, rv, wgt);
  emp3 += en3[0]; std::transform(en3.begin()+1, en3.end(), ctrl, ctrl, std::plus<>()); ctrl += en3.size() -1;
  en3 = mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], 2, ovps.o_set[0][0].s_21, ovps.o_set[0][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22, rv, wgt);
  emp3 += en3[0]; std::transform(en3.begin()+1, en3.end(), ctrl, ctrl, std::plus<>()); ctrl += en3.size() -1;

  ctrl = control.begin();
  en3 = mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], -1, ovps.v_set[0][0].s_12, ovps.v_set[0][0].s_21, ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, rv, wgt);
  emp3 += en3[0]; std::transform(en3.begin()+1, en3.end(), ctrl, ctrl, std::plus<>()); ctrl += en3.size() -1;
  en3 = mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], -4, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_11, rv, wgt);
  emp3 += en3[0]; std::transform(en3.begin()+1, en3.end(), ctrl, ctrl, std::plus<>()); ctrl += en3.size() -1;
  en3 = mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], -4, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11, rv, wgt);
  emp3 += en3[0]; std::transform(en3.begin()+1, en3.end(), ctrl, ctrl, std::plus<>()); ctrl += en3.size() -1;
  en3 = mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], -4, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11, rv, wgt);
  emp3 += en3[0]; std::transform(en3.begin()+1, en3.end(), ctrl, ctrl, std::plus<>()); ctrl += en3.size() -1;
  en3 = mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], -4, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_11, rv, wgt);
  emp3 += en3[0]; std::transform(en3.begin()+1, en3.end(), ctrl, ctrl, std::plus<>()); ctrl += en3.size() -1;
  en3 = mcmp3_helper(iops.iopns[KEYS::MC_NPAIR], -1, ovps.o_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22, rv, wgt);
  emp3 += en3[0]; std::transform(en3.begin()+1, en3.end(), ctrl, ctrl, std::plus<>()); ctrl += en3.size() -1;

  auto tau_wgt = tau.get_wgt(2);
  std::transform(control.begin(), control.end(), control.begin(), [&](double x) { return x * tau_wgt; });
  emp3 *= tau_wgt;

  // divide by number of RW samples
  auto nsamp = static_cast<double>(iops.iopns[KEYS::MC_NPAIR]);
  nsamp *= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 1);
  nsamp *= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 2);
  emp3 = emp3 / nsamp;
  std::transform(control.begin(), control.end(), control.begin(), [nsamp](double x) { return x / nsamp; });
}
