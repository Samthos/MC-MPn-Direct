//
// Created by aedoran on 6/1/18.
//

#include <algorithm>
#include <vector>
#include <array>

#include "cblas.h"
#include "../qc_monte.h"
#include "mp3_functional.h"

template <int CVMP3, template <typename, typename> typename Container, template <typename> typename Allocator>
MP3_Functional<CVMP3, Container, Allocator>::MP3_Functional(int electron_pairs_in) : 
    Standard_MP_Functional<Container, Allocator>(3 * CVMP3 * (1 + CVMP3), 2, "23"),
    electron_pairs(electron_pairs_in),
    A_ij(electron_pairs * electron_pairs),
    A_ik(electron_pairs * electron_pairs),
    A_jk(electron_pairs * electron_pairs),
    A_i(electron_pairs * (6 - 3 * CVMP3 + 3 * CVMP3 * CVMP3)),
    A((6 + 5 * CVMP3 + CVMP3 * CVMP3 * CVMP3)),
    ctrl((6 + 5 * CVMP3 + CVMP3 * CVMP3 * CVMP3)) {
}

template <int CVMP3, template <typename, typename> typename Container, template <typename> typename Allocator>
void MP3_Functional<CVMP3, Container, Allocator>::mcmp3_helper(
    const int offset, double constant,
    vector_double& A_ij_1, vector_double& A_ij_2,
    vector_double& A_ik_1, vector_double& A_ik_2,
    vector_double& A_jk_1, vector_double& A_jk_2,
    vector_double& j_operator, vector_double& k_operator) {

  // build ij jk intermetiates
  this->blas_wrapper.multiplies(A_ij_1.begin(), A_ij_1.end(), A_ij_2.begin(), A_ij.begin());
  this->blas_wrapper.multiplies(A_jk_1.begin(), A_jk_1.end(), A_jk_2.begin(), A_jk.begin());

  // rescale jk with rv
  this->blas_wrapper.ddgmm(true,
      electron_pairs, electron_pairs,
      A_jk, electron_pairs,
      j_operator, 1,
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
  this->blas_wrapper.multiplies(A_ik.begin(), A_ik.end(), A_ik_1.begin(), A_ik.begin());
  this->blas_wrapper.multiplies(A_ik.begin(), A_ik.end(), A_ik_2.begin(), A_ik.begin());

  // A_ik . rv
  if (CVMP3 < 2) {
    this->blas_wrapper.dgemv(true,
        electron_pairs, electron_pairs,
        constant,
        A_ik, 0, electron_pairs,
        k_operator, 0, 1,
        1.0,
        A_i, offset * electron_pairs, 1);
  } if (CVMP3 >= 2) {
    this->blas_wrapper.dgemm(true, false,
        electron_pairs, 2, electron_pairs,
        constant,
        A_ik, 0, electron_pairs,
        k_operator, 0, electron_pairs,
        1.0,
        A_i, offset * electron_pairs, 6 * electron_pairs);
  }
}

template <int CVMP3, template <typename, typename> typename Container, template <typename> typename Allocator>
void MP3_Functional<CVMP3, Container, Allocator>::call_helper(OVPS_Type& ovps, vector_double& j_op, vector_double& k_op, int offset) {
  mcmp3_helper(0 + offset,  2, ovps.v_set[0][0].s_11, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, j_op, k_op);
  mcmp3_helper(1 + offset,  2, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11, j_op, k_op);
  mcmp3_helper(2 + offset,  8, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_11, j_op, k_op);
  mcmp3_helper(3 + offset,  2, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_11, j_op, k_op);
  mcmp3_helper(4 + offset,  2, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11, j_op, k_op);
  mcmp3_helper(5 + offset,  2, ovps.o_set[0][0].s_21, ovps.o_set[0][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22, j_op, k_op);
  mcmp3_helper(0 + offset, -1, ovps.v_set[0][0].s_12, ovps.v_set[0][0].s_21, ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, j_op, k_op);
  mcmp3_helper(1 + offset, -4, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_11, j_op, k_op);
  mcmp3_helper(2 + offset, -4, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11, j_op, k_op);
  mcmp3_helper(3 + offset, -4, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11, j_op, k_op);
  mcmp3_helper(4 + offset, -4, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_11, j_op, k_op);
  mcmp3_helper(5 + offset, -1, ovps.o_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22, j_op, k_op);
}

template <int CVMP3, template <typename, typename> typename Container, template <typename> typename Allocator>
void MP3_Functional<CVMP3, Container, Allocator>::energy(double& emp, std::vector<double>& control, OVPS_Type& ovps, Electron_Pair_List_Type* electron_pair_list, Tau* tau) {
  this->blas_wrapper.fill(A_i.begin(), A_i.end(), 0.0);

  if (CVMP3 < 2) {
    call_helper(ovps, electron_pair_list->rv, electron_pair_list->rv, 0);
  } else {
    call_helper(ovps, electron_pair_list->rv, electron_pair_list->rv_inverse_weight, 0);
    if (CVMP3 == 3) {
      call_helper(ovps, electron_pair_list->inverse_weight, electron_pair_list->rv_inverse_weight, 12);
    }
  }

  if (CVMP3 == 0) {
    this->blas_wrapper.dgemv(true,
        electron_pairs, 6, 
        1.0,
        A_i, electron_pairs,
        electron_pair_list->rv, 1,
        0.0,
        A, 1);
  } else if (CVMP3 >= 1) {
    this->blas_wrapper.dgemm(true, false,
        A.size() / 2, 2, electron_pairs,
        1.0,
        A_i, electron_pairs,
        electron_pair_list->rv_inverse_weight, electron_pairs,
        0.0,
        A, A.size() / 2);
  }
#ifdef HAVE_CUDA
  thrust::copy(A.begin(), A.end(), ctrl.begin());
#else
  std::copy(A.begin(), A.end(), ctrl.begin());
#endif

  // divide by number of RW samples
  auto nsamp_tauwgt = tau->get_wgt(2);
  nsamp_tauwgt /= static_cast<double>(electron_pairs);
  nsamp_tauwgt /= static_cast<double>(electron_pairs - 1);
  nsamp_tauwgt /= static_cast<double>(electron_pairs - 2);
  std::transform(ctrl.begin(), ctrl.end(), ctrl.begin(), [&](double c) { return c * nsamp_tauwgt; });

  emp += std::accumulate(ctrl.begin(), ctrl.begin() + 6, 0.0);
  if (CVMP3 == 1 || CVMP3 == 2) {
    std::transform(ctrl.begin() + 6, ctrl.end(), control.begin(), control.begin(), std::plus<>());
  } else if (CVMP3 == 3) {
    std::transform(ctrl.begin() + 6, ctrl.begin() + 12, control.begin(), control.begin(), std::plus<>());
    std::transform(ctrl.begin() + 18, ctrl.end(), control.begin() + 6, control.begin() + 6, std::plus<>());
  }
}
