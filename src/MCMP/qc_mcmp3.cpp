//
// Created by aedoran on 6/1/18.
//

#include <algorithm>
#include <vector>
#include <array>

#include "cblas.h"
#include "../blas_calls.h"
#include "../qc_monte.h"
#include "qc_mcmp3.h"

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

MCMP* create_MCMP3(int cv_level) {
  MCMP* mcmp = nullptr;
  if (cv_level == 0) {
    mcmp = new MCMP3<0>;
  } else if (cv_level == 1) {
    mcmp = new MCMP3<1>;
  } else if (cv_level == 2) {
    mcmp = new MCMP3<2>;
  } else if (cv_level == 3) {
    mcmp = new MCMP3<3>;
  }
  
  if (mcmp == nullptr) {
    std::cerr << "MCMP3 not supported with cv level " << cv_level << "\n";
    exit(0);
  }
  return mcmp;
}

template <int CVMP3>
void MCMP3<CVMP3>::mcmp3_helper(
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
  if (CVMP3 >= 1) {
    control[offset + 0] += constant * std::inner_product(wgt.begin(), wgt.end(), A_jk.begin(), 0.0); // w * r * r
  }

  if (CVMP3 >= 2) {
    // A_ik . wgt
    cblas_dgemv(CblasColMajor,
        CblasTrans,
        mc_pair_num, mc_pair_num,
        1.0,
        A_ik.data(), mc_pair_num,
        wgt.data(), 1,
        0.0,
        A_jk.data(), 1);
    control[offset + 6] += constant * std::inner_product(rv.begin(), rv.end(), A_jk.begin(), 0.0); // r * r * w
    control[offset + 12] += constant * std::inner_product(wgt.begin(), wgt.end(), A_jk.begin(), 0.0); // w * r * w
  }

  if (CVMP3 >= 3) {
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
    control[offset + 18] += constant * std::inner_product(wgt.begin(), wgt.end(), A_jk.begin(), 0.0); // w * w * r

    // A_ik . wgt
    cblas_dgemv(CblasColMajor,
        CblasTrans,
        mc_pair_num, mc_pair_num,
        1.0,
        A_ik.data(), mc_pair_num,
        wgt.data(), 1,
        0.0,
        A_jk.data(), 1);
    control[offset + 24] += constant * std::inner_product(wgt.begin(), wgt.end(), A_jk.begin(), 0.0); // r * w * w
    control[offset + 30] += constant * std::inner_product(rv.begin(), rv.end(), A_jk.begin(), 0.0); // w * w * w
  }
}

template <int CVMP3>
void MCMP3<CVMP3>::energy(double& emp, std::vector<double>& control, OVPs& ovps, Electron_Pair_List* electron_pair_list, Tau* tau) {
  double en3 = 0;
  std::vector<double> ctrl(control.size(), 0.0);

  std::vector<double>& rv = electron_pair_list->rv;
  std::vector<double> wgt(electron_pair_list->size());
  std::transform(electron_pair_list->wgt.begin(), electron_pair_list->wgt.end(), wgt.begin(), [](double w){return 1.0/w;});

  mcmp3_helper(en3, ctrl, 0, electron_pair_list->size(), 2, ovps.v_set[0][0].s_11, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, rv, wgt);
  mcmp3_helper(en3, ctrl, 1, electron_pair_list->size(), 2, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11, rv, wgt);
  mcmp3_helper(en3, ctrl, 2, electron_pair_list->size(), 8, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_11, rv, wgt);
  mcmp3_helper(en3, ctrl, 3, electron_pair_list->size(), 2, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_11, rv, wgt);
  mcmp3_helper(en3, ctrl, 4, electron_pair_list->size(), 2, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11, rv, wgt);
  mcmp3_helper(en3, ctrl, 5, electron_pair_list->size(), 2, ovps.o_set[0][0].s_21, ovps.o_set[0][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22, rv, wgt);

  mcmp3_helper(en3, ctrl, 0, electron_pair_list->size(), -1, ovps.v_set[0][0].s_12, ovps.v_set[0][0].s_21, ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, rv, wgt);
  mcmp3_helper(en3, ctrl, 1, electron_pair_list->size(), -4, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_11, rv, wgt);
  mcmp3_helper(en3, ctrl, 2, electron_pair_list->size(), -4, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11, rv, wgt);
  mcmp3_helper(en3, ctrl, 3, electron_pair_list->size(), -4, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11, rv, wgt);
  mcmp3_helper(en3, ctrl, 4, electron_pair_list->size(), -4, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_11, rv, wgt);
  mcmp3_helper(en3, ctrl, 5, electron_pair_list->size(), -1, ovps.o_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22, rv, wgt);

  // divide by number of RW samples
  auto nsamp_tauwgt = tau->get_wgt(2);
  nsamp_tauwgt /= static_cast<double>(electron_pair_list->size());
  nsamp_tauwgt /= static_cast<double>(electron_pair_list->size() - 1);
  nsamp_tauwgt /= static_cast<double>(electron_pair_list->size() - 2);
  emp = emp + en3 * nsamp_tauwgt;
  if (CVMP3 >= 1) {
    std::transform(ctrl.begin(), ctrl.end(), control.begin(), control.begin(), [&](double c, double total) { return total + c * nsamp_tauwgt; });
  }
}
