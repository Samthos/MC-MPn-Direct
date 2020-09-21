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
MP3_Functional<CVMP3, Container, Allocator>::MP3_Functional(int electron_pairs) : 
    Standard_MP_Functional<Container, Allocator>(3 * CVMP3 * (1 + CVMP3), 2, "23"),
    A_ij(electron_pairs * electron_pairs),
    A_ik(electron_pairs * electron_pairs),
    A_jk(electron_pairs * electron_pairs),
    A_i(electron_pairs * (6 - 3 * CVMP3 + 3 * CVMP3 * CVMP3)),
    A((6 + 5 * CVMP3 + CVMP3 * CVMP3 * CVMP3)) {
      std::cout << "MP3SIZE A_i  = " << A_i.size() << "\n";
      std::cout << "MP3SIZE A    = " << A.size() << "\n";
}

template <int CVMP3, template <typename, typename> typename Container, template <typename> typename Allocator>
void MP3_Functional<CVMP3, Container, Allocator>::mcmp3_helper(
    const int offset,
    unsigned int electron_pairs, double constant,
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
  /*
  en3 += constant * this->blas_wrapper.ddot(electron_pairs, rv, 1, A_i, 1);
  if (CVMP3 >= 1) {
    control[offset + 0] += constant * this->blas_wrapper.ddot(electron_pairs, wgt, 1, A_i, 1);
  }

  if (CVMP3 >= 2) {
    // A_ik . wgt
    this->blas_wrapper.dgemv(true,
        electron_pairs, electron_pairs,
        1.0,
        A_ik, electron_pairs,
        wgt, 1,
        0.0,
        A_i, 1);
    control[offset +  6] += constant * this->blas_wrapper.ddot(electron_pairs,  rv, 1, A_i, 1); // r * r * w
    control[offset + 12] += constant * this->blas_wrapper.ddot(electron_pairs, wgt, 1, A_i, 1); // w * r * w
  }

  if (CVMP3 >= 3) {
    // recompute A_jk
    this->blas_wrapper.multiplies(A_jk_1.begin(), A_jk_1.end(), A_jk_2.begin(), A_jk.begin());

    // scale A_jk by wgt
    this->blas_wrapper.ddgmm(true,
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
    this->blas_wrapper.multiplies(A_ik.begin(), A_ik.end(), A_ik_1.begin(), A_ik.begin());
    this->blas_wrapper.multiplies(A_ik.begin(), A_ik.end(), A_ik_2.begin(), A_ik.begin());

    // A_ik . rv
    this->blas_wrapper.dgemv(true,
        electron_pairs, electron_pairs,
        1.0,
        A_ik, electron_pairs,
        rv, 1,
        0.0,
        A_i, 1);
    control[offset + 18] += constant * this->blas_wrapper.ddot(electron_pairs, wgt, 1, A_i, 1); // w * w * r

    // A_ik . wgt
    this->blas_wrapper.dgemv(true,
        electron_pairs, electron_pairs,
        1.0,
        A_ik, electron_pairs,
        wgt, 1,
        0.0,
        A_i, 1);
    control[offset + 24] += constant * this->blas_wrapper.ddot(electron_pairs, wgt, 1, A_i, 1); // w * w * w
    control[offset + 30] += constant * this->blas_wrapper.ddot(electron_pairs,  rv, 1, A_i, 1); // r * w * w
  }
  */
}

template <int CVMP3, template <typename, typename> typename Container, template <typename> typename Allocator>
void MP3_Functional<CVMP3, Container, Allocator>::energy(double& emp, std::vector<double>& control, OVPS_Type& ovps, Electron_Pair_List_Type* electron_pair_list, Tau* tau) {
  double en3 = 0;
  std::vector<double> ctrl(control.size(), 0.0);

  vector_double* j_op = &electron_pair_list->rv;
  vector_double* k_op = &electron_pair_list->rv;
  if (CVMP3 >= 2) {
    k_op = &electron_pair_list->rv_inverse_weight;
  }
  this->blas_wrapper.fill(A_i.begin(), A_i.end(), 0.0);

  mcmp3_helper(0, electron_pair_list->size(),  2, ovps.v_set[0][0].s_11, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, *j_op, *k_op);
  mcmp3_helper(1, electron_pair_list->size(),  2, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11, *j_op, *k_op);
  mcmp3_helper(2, electron_pair_list->size(),  8, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_11, *j_op, *k_op);
  mcmp3_helper(3, electron_pair_list->size(),  2, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_11, *j_op, *k_op);
  mcmp3_helper(4, electron_pair_list->size(),  2, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11, *j_op, *k_op);
  mcmp3_helper(5, electron_pair_list->size(),  2, ovps.o_set[0][0].s_21, ovps.o_set[0][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22, *j_op, *k_op);
  mcmp3_helper(0, electron_pair_list->size(), -1, ovps.v_set[0][0].s_12, ovps.v_set[0][0].s_21, ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, *j_op, *k_op);
  mcmp3_helper(1, electron_pair_list->size(), -4, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_11, *j_op, *k_op);
  mcmp3_helper(2, electron_pair_list->size(), -4, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11, *j_op, *k_op);
  mcmp3_helper(3, electron_pair_list->size(), -4, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11, *j_op, *k_op);
  mcmp3_helper(4, electron_pair_list->size(), -4, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_11, *j_op, *k_op);
  mcmp3_helper(5, electron_pair_list->size(), -1, ovps.o_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22, *j_op, *k_op);

  if (CVMP3 == 3) {
    j_op = &electron_pair_list->inverse_weight;
    mcmp3_helper(12, electron_pair_list->size(),  2, ovps.v_set[0][0].s_11, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, *j_op, *k_op);
    mcmp3_helper(13, electron_pair_list->size(),  2, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11, *j_op, *k_op);
    mcmp3_helper(14, electron_pair_list->size(),  8, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_11, *j_op, *k_op);
    mcmp3_helper(15, electron_pair_list->size(),  2, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_11, *j_op, *k_op);
    mcmp3_helper(16, electron_pair_list->size(),  2, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11, *j_op, *k_op);
    mcmp3_helper(17, electron_pair_list->size(),  2, ovps.o_set[0][0].s_21, ovps.o_set[0][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22, *j_op, *k_op);
    mcmp3_helper(12, electron_pair_list->size(), -1, ovps.v_set[0][0].s_12, ovps.v_set[0][0].s_21, ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, *j_op, *k_op);
    mcmp3_helper(13, electron_pair_list->size(), -4, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_11, *j_op, *k_op);
    mcmp3_helper(14, electron_pair_list->size(), -4, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11, *j_op, *k_op);
    mcmp3_helper(15, electron_pair_list->size(), -4, ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11, *j_op, *k_op);
    mcmp3_helper(16, electron_pair_list->size(), -4, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[1][0].s_12, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_11, *j_op, *k_op);
    mcmp3_helper(17, electron_pair_list->size(), -1, ovps.o_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22, *j_op, *k_op);
  }

  for (int i = 0; i < A_i.size(); i++) {
    if (i % 8 == 0) {
      printf("\nMP3B %3i ", i);
    }
    printf("%16.8f ", A_i[i]);
  }
  printf("\n");

  if (CVMP3 == 0) {
    this->blas_wrapper.dgemv(true,
        electron_pair_list->size(), 6, 
        1.0,
        A_i, electron_pair_list->size(),
        electron_pair_list->rv, 1,
        0.0,
        A, 1);
#ifdef HAVE_CUDA
    std::vector<double> gr(A.size());
    thrust::copy(A.begin(), A.end(), gr.begin());
    en3 = std::accumulate(gr.begin(), gr.end(), 0.0);
#else
    en3 = std::accumulate(A.begin(), A.end(), 0.0);
#endif
  } else if (CVMP3 >= 1) {
    this->blas_wrapper.dgemm(true, false,
        A.size() / 2, 2, electron_pair_list->size(),
        1.0,
        A_i, electron_pair_list->size(),
        electron_pair_list->rv_inverse_weight, electron_pair_list->size(),
        0.0,
        A, A.size() / 2);
#ifdef HAVE_CUDA
    std::vector<double> gr(A.size());
    thrust::copy(A.begin(), A.end(), gr.begin());
    en3 = std::accumulate(gr.begin(), gr.end() + 6, 0.0);
#else
    en3 = std::accumulate(A.begin(), A.begin() + 6, 0.0);
#endif
  }

  if (CVMP3 == 1 || CVMP3 == 2) {
    std::copy(A.begin() + 6, A.end(), ctrl.begin());
  } else if (CVMP3 == 3) {
    std::copy(A.begin() + 6, A.begin() + 12, ctrl.begin());
    std::copy(A.begin() + 18, A.end(), ctrl.begin() + 6);
  }

  for (int i = 0; i < A.size(); i++) {
    if (i % 6 == 0) {
      printf("\nMP3A %3i ", i);
    }
    printf("%16.8f ", A[i]);
  }
  printf("\n");

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
