//
// Created by aedoran on 6/1/18.
//

#include <algorithm>
#include <vector>
#include <array>

#include "cblas.h"
#include "../qc_monte.h"
#include "mp3_functional.h"

MP_Functional* create_MP3_Functional(int cv_level, int electron_pairs) {
  MP_Functional* mcmp = nullptr;
  if (cv_level == 0) {
    mcmp = new MP3_Functional<0>(electron_pairs);
  } else if (cv_level == 1) {
    mcmp = new MP3_Functional<1>(electron_pairs);
  } else if (cv_level == 2) {
    mcmp = new MP3_Functional<2>(electron_pairs);
  } else if (cv_level == 3) {
    mcmp = new MP3_Functional<3>(electron_pairs);
  }
  
  if (mcmp == nullptr) {
    std::cerr << "MP3_Functional not supported with cv level " << cv_level << "\n";
    exit(0);
  }
  return mcmp;
}

template <int CVMP3>
MP3_Functional<CVMP3>::MP3_Functional(int electron_pairs) : 
    Standard_MP_Functional<std::vector, std::allocator>(3 * CVMP3 * (1 + CVMP3), 2, "23"),
    A_ij(electron_pairs * electron_pairs),
    A_ik(electron_pairs * electron_pairs),
    A_jk(electron_pairs * electron_pairs) {
}

template <int CVMP3>
void MP3_Functional<CVMP3>::mcmp3_helper(
    double& en3, std::vector<double>& control, const int offset,
    unsigned int electron_pairs, double constant,
    vector_double& A_ij_1, vector_double& A_ij_2,
    vector_double& A_ik_1, vector_double& A_ik_2,
    vector_double& A_jk_1, vector_double& A_jk_2,
    vector_double& rv, vector_double& wgt) {
  double ddot_result;

  // build ij jk intermetiates
  this->blas_wrapper.transform_multiplies(A_ij_1, A_ij_2, A_ij);
  this->blas_wrapper.transform_multiplies(A_jk_1, A_jk_2, A_jk);

  // rescale jk with rv
  this->blas_wrapper.Ddgmm(true,
      electron_pairs, electron_pairs,
      A_jk, electron_pairs,
      rv, 1,
      A_jk, electron_pairs);

  // A_ik = A_jk . A_ij
  this->blas_wrapper.dgemm(false, false,
      electron_pairs, electron_pairs, electron_pairs,
      1.0,
      A_jk, electron_pairs,
      A_ij, electron_pairs,
      0.0,
      A_ik, electron_pairs);

  // scale A_ik by ik_1 and ik_2
  this->blas_wrapper.transform_multiplies(A_ik, A_ik_1, A_ik);
  this->blas_wrapper.transform_multiplies(A_ik, A_ik_2, A_ik);

  // A_ik . rv
  this->blas_wrapper.dgemv(true,
      electron_pairs, electron_pairs,
      1.0,
      A_ik, electron_pairs,
      rv, 1,
      0.0,
      A_jk, 1);
  this->blas_wrapper.ddot(rv.size(), rv, 1, A_jk, 1, &ddot_result);
  en3 += constant * ddot_result;
  if (CVMP3 >= 1) {
    this->blas_wrapper.ddot(rv.size(), wgt, 1, A_jk, 1, &ddot_result);
    control[offset + 0] += constant * ddot_result;
  }

  if (CVMP3 >= 2) {
    // A_ik . wgt
    this->blas_wrapper.dgemv(true,
        electron_pairs, electron_pairs,
        1.0,
        A_ik, electron_pairs,
        wgt, 1,
        0.0,
        A_jk, 1);
    this->blas_wrapper.ddot(rv.size(), rv, 1, A_jk, 1, &ddot_result); // r * r * w
    control[offset + 6] += constant * ddot_result;
    this->blas_wrapper.ddot(wgt.size(), wgt, 1, A_jk, 1, &ddot_result); // w * r * w
    control[offset + 12] += constant * ddot_result;
  }

  if (CVMP3 >= 3) {
    // recompute A_jk
    this->blas_wrapper.transform_multiplies(A_jk_1, A_jk_2, A_jk);

    // scale A_jk by wgt
    this->blas_wrapper.Ddgmm(true,
        electron_pairs, electron_pairs,
        A_jk, electron_pairs,
        wgt, 1,
        A_jk, electron_pairs);

    // A_ik = A_jk . A_ij
    this->blas_wrapper.dgemm(false, false,
        electron_pairs, electron_pairs, electron_pairs,
        1.0,
        A_jk, electron_pairs,
        A_ij, electron_pairs,
        0.0,
        A_ik, electron_pairs);

    // scale A_ik by ik_1 and ik_2
    this->blas_wrapper.transform_multiplies(A_ik, A_ik_1, A_ik);
    this->blas_wrapper.transform_multiplies(A_ik, A_ik_2, A_ik);

    // A_ik . rv
    this->blas_wrapper.dgemv(true,
        electron_pairs, electron_pairs,
        1.0,
        A_ik, electron_pairs,
        rv, 1,
        0.0,
        A_jk, 1);
    this->blas_wrapper.ddot(wgt.size(), wgt, 1, A_jk, 1, &ddot_result); // w * w * r
    control[offset + 18] += constant * ddot_result;

    // A_ik . wgt
    this->blas_wrapper.dgemv(true,
        electron_pairs, electron_pairs,
        1.0,
        A_ik, electron_pairs,
        wgt, 1,
        0.0,
        A_jk, 1);
    this->blas_wrapper.ddot(wgt.size(), wgt, 1, A_jk, 1, &ddot_result); // r * w * w
    control[offset + 24] += constant * ddot_result;
    this->blas_wrapper.ddot( rv.size(),  rv, 1, A_jk, 1, &ddot_result); // w * w * w
    control[offset + 30] += constant * ddot_result;
  }
}

template <int CVMP3>
void MP3_Functional<CVMP3>::energy(double& emp, std::vector<double>& control, OVPS_Host& ovps, Electron_Pair_List_Type* electron_pair_list, Tau* tau) {
  double en3 = 0;
  std::vector<double> ctrl(control.size(), 0.0);

  vector_double& rv = electron_pair_list->rv;
  vector_double& wgt = electron_pair_list->inverse_weight;

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
