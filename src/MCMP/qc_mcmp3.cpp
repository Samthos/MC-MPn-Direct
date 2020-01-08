//
// Created by aedoran on 6/1/18.
//

#include <algorithm>
#include <vector>
#include <array>

#include "cblas.h"
#include "../blas_calls.h"
#include "../qc_monte.h"

void mcmp3_helper(
    double& en3, std::vector<double>& control, const int offset,
    unsigned int mc_pair_num, double constant,
    std::vector<double>& A_ij_1, std::vector<double>& A_ij_2,
    std::vector<double>& A_ik_1, std::vector<double>& A_ik_2,
    std::vector<double>& A_jk_1, std::vector<double>& A_jk_2,
    std::vector<double>& rv, std::vector<double>& wgt) {
  std::vector<double> A_ij(mc_pair_num * mc_pair_num);
  std::vector<double> A_ik(mc_pair_num * mc_pair_num);
  std::vector<double> A_jk(mc_pair_num * mc_pair_num);

  // build ij jk intermetiates
  std::transform(A_ij_1.begin(), A_ij_1.end(), A_ij_2.begin(), A_ij.begin(), std::multiplies<>());
  std::transform(A_jk_1.begin(), A_jk_1.end(), A_jk_2.begin(), A_jk.begin(), std::multiplies<>());

  // rescale jk with rv
  Ddgmm(DDGMM_SIDE_RIGHT,
      mc_pair_num, mc_pair_num,
      A_jk.data(), mc_pair_num,
      rv.data(), 1,
      A_jk.data(), mc_pair_num);

  // A_ik = A_jk . A_ij
  cblas_dgemm(CblasColMajor,
      CblasNoTrans, CblasNoTrans,
      mc_pair_num, mc_pair_num, mc_pair_num,
      1.0,
      A_jk.data(), mc_pair_num,
      A_ij.data(), mc_pair_num,
      0.0,
      A_ik.data(), mc_pair_num);

  // scale A_ik by ik_1 and ik_2
  std::transform(A_ik.begin(), A_ik.end(), A_ik_1.begin(), A_ik.begin(), std::multiplies<>());
  std::transform(A_ik.begin(), A_ik.end(), A_ik_2.begin(), A_ik.begin(), std::multiplies<>());

  // A_ik . rv
  cblas_dgemv(CblasColMajor,
      CblasTrans,
      mc_pair_num, mc_pair_num,
      1.0,
      A_ik.data(), mc_pair_num,
      rv.data(), 1,
      0.0,
      A_jk.data(), 1);
  en3 += constant * std::inner_product(rv.begin(), rv.end(), A_jk.begin(), 0.0); // r * r * r
#if MP3CV >= 1
  control[offset + 0] += constant * std::inner_product(wgt.begin(), wgt.end(), A_jk.begin(), 0.0); // w * r * r
#endif

#if MP3CV >= 2
  // A_ik . wgt
  cblas_dgemv(CblasColMajor,
      CblasTrans,
      mc_pair_num, mc_pair_num,
      1.0,
      A_ik.data(), mc_pair_num,
      wgt.data(), 1,
      0.0,
      A_jk.data(), 1);
  control[offset + 1] += constant * std::inner_product(rv.begin(), rv.end(), A_jk.begin(), 0.0); // r * r * w
  control[offset + 2] += constant * std::inner_product(wgt.begin(), wgt.end(), A_jk.begin(), 0.0); // w * r * w
#endif

#if MP3CV >= 3
  // recompute A_jk
  std::transform(A_jk_1.begin(), A_jk_1.end(), A_jk_2.begin(), A_jk.begin(), std::multiplies<>());

  // scale A_jk by wgt
  Ddgmm(DDGMM_SIDE_RIGHT,
      mc_pair_num, mc_pair_num,
      A_jk.data(), mc_pair_num,
      wgt.data(), 1,
      A_jk.data(), mc_pair_num);

  // A_ik = A_jk . A_ij
  cblas_dgemm(CblasColMajor,
      CblasNoTrans, CblasNoTrans,
      mc_pair_num, mc_pair_num, mc_pair_num,
      1.0,
      A_jk.data(), mc_pair_num,
      A_ij.data(), mc_pair_num,
      0.0,
      A_ik.data(), mc_pair_num);

  // scale A_ik by ik_1 and ik_2
  std::transform(A_ik.begin(), A_ik.end(), A_ik_1.begin(), A_ik.begin(), std::multiplies<>());
  std::transform(A_ik.begin(), A_ik.end(), A_ik_2.begin(), A_ik.begin(), std::multiplies<>());

  // A_ik . rv
  cblas_dgemv(CblasColMajor,
      CblasTrans,
      mc_pair_num, mc_pair_num,
      1.0,
      A_ik.data(), mc_pair_num,
      rv.data(), 1,
      0.0,
      A_jk.data(), 1);
  control[offset + 3] += constant * std::inner_product(wgt.begin(), wgt.end(), A_jk.begin(), 0.0); // w * w * r

  // A_ik . wgt
  cblas_dgemv(CblasColMajor,
      CblasTrans,
      mc_pair_num, mc_pair_num,
      1.0,
      A_ik.data(), mc_pair_num,
      wgt.data(), 1,
      0.0,
      A_jk.data(), 1);
  control[offset + 4] += constant * std::inner_product(wgt.begin(), wgt.end(), A_jk.begin(), 0.0); // r * w * w
  control[offset + 5] += constant * std::inner_product(rv.begin(), rv.end(), A_jk.begin(), 0.0); // w * w * w
#endif
}

void MP::mcmp3_energy(double& emp3, std::vector<double>& control3) {
  double en3 = 0;
  std::vector<double> ctrl(control3.size(), 0.0);

  std::vector<double>& rv = el_pair_list->rv;
  std::vector<double>& wgt = el_pair_list->wgt;
  // std::transform(el_pair_list->begin(), el_pair_list->end(), rv.begin(), [](Electron_Pair ept){return ept.rv;});
  // std::transform(el_pair_list->begin(), el_pair_list->end(), wgt.begin(), [](Electron_Pair ept){return 1.0/ept.wgt;});

  mcmp3_helper(en3, ctrl,  0, iops.iopns[KEYS::MC_NPAIR], 2, ovps.v_set[0][0].s_11, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, rv, wgt);
  mcmp3_helper(en3, ctrl,  6, iops.iopns[KEYS::MC_NPAIR], 2, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11, rv, wgt);
  mcmp3_helper(en3, ctrl, 12, iops.iopns[KEYS::MC_NPAIR], 8, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_11, rv, wgt);
  mcmp3_helper(en3, ctrl, 18, iops.iopns[KEYS::MC_NPAIR], 2, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_11, rv, wgt);
  mcmp3_helper(en3, ctrl, 24, iops.iopns[KEYS::MC_NPAIR], 2, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11, rv, wgt);
  mcmp3_helper(en3, ctrl, 30, iops.iopns[KEYS::MC_NPAIR], 2, ovps.o_set[0][0].s_21, ovps.o_set[0][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22, rv, wgt);

  mcmp3_helper(en3, ctrl,  0,iops.iopns[KEYS::MC_NPAIR], -1, ovps.v_set[0][0].s_12, ovps.v_set[0][0].s_21, ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, rv, wgt);
  mcmp3_helper(en3, ctrl,  6,iops.iopns[KEYS::MC_NPAIR], -4, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_11, rv, wgt);
  mcmp3_helper(en3, ctrl, 12,iops.iopns[KEYS::MC_NPAIR], -4, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11, rv, wgt);
  mcmp3_helper(en3, ctrl, 18,iops.iopns[KEYS::MC_NPAIR], -4, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11, rv, wgt);
  mcmp3_helper(en3, ctrl, 24,iops.iopns[KEYS::MC_NPAIR], -4, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_11, rv, wgt);
  mcmp3_helper(en3, ctrl, 30,iops.iopns[KEYS::MC_NPAIR], -1, ovps.o_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22, rv, wgt);

  // divide by number of RW samples
  auto nsamp_tauwgt = tau->get_wgt(2);
  nsamp_tauwgt /= static_cast<double>(iops.iopns[KEYS::MC_NPAIR]);
  nsamp_tauwgt /= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 1);
  nsamp_tauwgt /= static_cast<double>(iops.iopns[KEYS::MC_NPAIR] - 2);
  emp3 = emp3 + en3 * nsamp_tauwgt;
#if MP3CV >= 1
  std::transform(ctrl.begin(), ctrl.end(), control3.begin(), control3.begin(), [&](double c, double total) { return total + c * nsamp_tauwgt; });
#endif
}
