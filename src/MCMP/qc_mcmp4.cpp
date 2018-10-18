//
// Created by aedoran on 6/13/18.
//
#include "../qc_monte.h"
#include "cblas.h"

void Prep(
    std::vector<double>& A, bool transA,
    std::vector<double>& B, bool transB,
    std::vector<double>& rv,
    std::vector<double>& C,
    int mc_pair_num) {
  if (transA && transB) {
    for (int tidx = 0; tidx < mc_pair_num; ++tidx) {
      for (int tidy = 0; tidy < mc_pair_num; ++tidy) {
        int index = tidy * mc_pair_num + tidx;
        C[index] = 0;
        C[index] -= A[tidx * mc_pair_num + tidx] * B[tidx * mc_pair_num + tidy] * rv[tidx];
        C[index] -= A[tidx * mc_pair_num + tidy] * B[tidy * mc_pair_num + tidy] * rv[tidy];
      }
    }
  } else if (transA && !transB) {
    for (int tidx = 0; tidx < mc_pair_num; ++tidx) {
      for (int tidy = 0; tidy < mc_pair_num; ++tidy) {
        int index = tidy * mc_pair_num + tidx;
        C[index] = 0;
        C[index] -= A[tidx * mc_pair_num + tidx] * B[tidy * mc_pair_num + tidx] * rv[tidx];
        C[index] -= A[tidx * mc_pair_num + tidy] * B[tidy * mc_pair_num + tidy] * rv[tidy];
      }
    }
  } else if (!transA && transB) {
    for (int tidx = 0; tidx < mc_pair_num; ++tidx) {
      for (int tidy = 0; tidy < mc_pair_num; ++tidy) {
        int index = tidy * mc_pair_num + tidx;
        C[index] = 0;
        C[index] -= A[tidx * mc_pair_num + tidx] * B[tidx * mc_pair_num + tidy] * rv[tidx];
        C[index] -= A[tidy * mc_pair_num + tidx] * B[tidy * mc_pair_num + tidy] * rv[tidy];
      }
    }
  } else if (!transA && !transB) {
    for (int tidx = 0; tidx < mc_pair_num; ++tidx) {
      for (int tidy = 0; tidy < mc_pair_num; ++tidy) {
        int index = tidy * mc_pair_num + tidx;
        C[index] = 0;
        C[index] -= A[tidx * mc_pair_num + tidx] * B[tidy * mc_pair_num + tidx] * rv[tidx];
        C[index] -= A[tidy * mc_pair_num + tidx] * B[tidy * mc_pair_num + tidy] * rv[tidy];
      }
    }
  }
}
std::vector<double> saxpy(std::vector<double>& x, std::vector<double>& y) {
  std::vector<double> z(y.size());
  std::transform(x.begin(), x.end(), y.begin(), z.begin(), std::multiplies<>());
  return z;
}
std::vector<double> contract(int mc_pair_num, std::vector<double>& A, std::vector<double>& B, std::vector<double>& v) {
  std::vector<double> results(mc_pair_num *mc_pair_num);
  Prep(A, true, B, false, v, results, mc_pair_num);
  Ddgmm(DDGMM_SIDE_LEFT,
      mc_pair_num, mc_pair_num,
      A.data(), mc_pair_num,
      v.data(), 1,
      A.data(), mc_pair_num);
  cblas_dgemm(CblasColMajor,
      CblasTrans, CblasNoTrans,
      mc_pair_num, mc_pair_num, mc_pair_num,
      1.0,
      A.data(), mc_pair_num,
      B.data(), mc_pair_num,
      1.0,
      results.data(), mc_pair_num);
  Ddgmm(DDGMM_SIDE_LEFT,
      mc_pair_num, mc_pair_num,
      A.data(), mc_pair_num,
      v.data(), 1,
      A.data(), mc_pair_num, std::divides<>());
  return results;
}
double mcmp4_ij_helper(int mc_pair_num, double constant,
    double& emp4, std::vector<double>& control, int offset,
    std::vector<double>& rv, std::vector<double>& wgt,
    std::vector<double>& ik, std::vector<double>& jk,
    std::vector<double>& il, std::vector<double>& jl
) {
  auto i_kl = saxpy(ik, il);
  auto j_kl = saxpy(jk, jl);

  auto r_r = saxpy(rv, rv);
  auto r_w = saxpy(rv, wgt);

  auto ij_rk = contract(mc_pair_num, jk, ik, rv);

  auto ij_rl = contract(mc_pair_num, jl, il, rv);
  auto ij_wl = contract(mc_pair_num, jl, il, wgt);

  auto ij_rrkl = contract(mc_pair_num, j_kl, i_kl, r_r);
  auto ij_rwkl = contract(mc_pair_num, j_kl, i_kl, r_w);

  // combin ij_rk * ij_rl - ij_rrkl
  std::transform(std::begin(ij_rk), std::end(ij_rk), std::begin(ij_rl), std::begin(ij_rl), std::multiplies<>());
  std::transform(std::begin(ij_rl), std::end(ij_rl), std::begin(ij_rrkl), std::begin(ij_rl), std::minus<>());

  // combin ij_rk * ij_wl - ij_rwkl
  std::transform(std::begin(ij_rk), std::end(ij_rk), std::begin(ij_wl), std::begin(ij_wl), std::multiplies<>());
  std::transform(std::begin(ij_wl), std::end(ij_wl), std::begin(ij_rwkl), std::begin(ij_wl), std::minus<>());

  // zero diagonals
  for (int i = 0; i < mc_pair_num * mc_pair_num; i += mc_pair_num+1) {
    ij_rl[i] = 0;
    ij_wl[i] = 0;
  }

  // contract j
  cblas_dgemv(CblasColMajor,
      CblasTrans,
      mc_pair_num, mc_pair_num,
      1.0,
      ij_rl.data(), mc_pair_num,
      rv.data(), 1,
      0.0,
      r_r.data(), 1);
  // contract j
  cblas_dgemv(CblasColMajor,
      CblasTrans,
      mc_pair_num, mc_pair_num,
      1.0,
      ij_wl.data(), mc_pair_num,
      rv.data(), 1,
      0.0,
      r_w.data(), 1);
  emp4 += std::inner_product(rv.begin(), rv.end(), r_r.begin(), 0.0) * constant;
  control[0 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), r_r.begin(), 0.0) * constant;
  control[1 + offset] +=  std::inner_product(wgt.begin(), wgt.end(), r_w.begin(), 0.0) * constant;
  control[2 + offset] +=  std::inner_product(rv.begin(), rv.end(), r_w.begin(), 0.0) * constant;
}
double mcmp4_ij_helper_t1(int mc_pair_num, double constant,
    double& emp4, std::vector<double>& control, int offset,
    std::vector<double>& rv, std::vector<double>& wgt,
    std::vector<double>& ik_1, std::vector<double>& ik_2, std::vector<double>& ik_3,
    std::vector<double>& jk,
    std::vector<double>& il,
    std::vector<double>& jl_1, std::vector<double>& jl_2, std::vector<double>& jl_3
) {
  auto ik = saxpy(ik_1, ik_2);
  auto jl = saxpy(jl_1, jl_2);
  std::transform(ik.begin(), ik.end(), ik_3.begin(), ik.begin(), std::multiplies<>());
  std::transform(jl.begin(), jl.end(), jl_3.begin(), jl.begin(), std::multiplies<>());
  mcmp4_ij_helper(mc_pair_num, constant, emp4, control, offset, rv, wgt, ik, jk, il, jl);
}
double mcmp4_ij_helper_t2(int mc_pair_num, double constant,
    double& emp4, std::vector<double>& control, int offset,
    std::vector<double>& rv, std::vector<double>& wgt,
    std::vector<double>& ik_1, std::vector<double>& ik_2,
    std::vector<double>& jk_1, std::vector<double>& jk_2,
    std::vector<double>& il_1, std::vector<double>& il_2,
    std::vector<double>& jl_1, std::vector<double>& jl_2
) {
  auto ik = saxpy(ik_1, ik_2);
  auto jk = saxpy(jk_1, jk_2);
  auto il = saxpy(il_1, il_2);
  auto jl = saxpy(jl_1, jl_2);
  mcmp4_ij_helper(mc_pair_num, constant, emp4, control, offset, rv, wgt,  ik, jk, il, jl);
}
double mcmp4_ij_helper_t3(int mc_pair_num, double constant,
    double& emp4, std::vector<double>& control, int offset,
    std::vector<double>& rv, std::vector<double>& wgt,
    std::vector<double> ik,
    std::vector<double>& jk_1, std::vector<double>& jk_2, std::vector<double>& jk_3,
    std::vector<double>& il_1, std::vector<double>& il_2, std::vector<double>& il_3,
    std::vector<double> jl
) {
  auto jk = saxpy(jk_1, jk_2);
  std::transform(jk.begin(), jk.end(), jk_3.begin(), jk.begin(), std::multiplies<>());
  auto il = saxpy(il_1, il_2);
  std::transform(il.begin(), il.end(), il_3.begin(), il.begin(), std::multiplies<>());
  mcmp4_ij_helper(mc_pair_num, constant, emp4, control, offset, rv, wgt, ik, jk, il, jl);
}
void MP::mcmp4_energy_ij_fast(double& emp4, std::vector<double>& control) {
  std::vector<double> rv(iops.iopns[KEYS::MC_NPAIR]);
  std::vector<double> wgt(iops.iopns[KEYS::MC_NPAIR]);
  std::transform(el_pair_list.begin(), el_pair_list.end(), rv.begin(), [](el_pair_typ ept){return ept.rv;});
  std::transform(el_pair_list.begin(), el_pair_list.end(), wgt.begin(), [](el_pair_typ ept){return 1.0/ept.wgt;});

  /*
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], 1, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_22, ovps.v_set[1][1].s_11,
      ovps.v_set[2][0].s_21, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22);
  */
  mcmp4_ij_helper_t1(iops.iopns[KEYS::MC_NPAIR], -4, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t1(iops.iopns[KEYS::MC_NPAIR], -4, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t1(iops.iopns[KEYS::MC_NPAIR], -4, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t1(iops.iopns[KEYS::MC_NPAIR], -4, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t1(iops.iopns[KEYS::MC_NPAIR], 2, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t1(iops.iopns[KEYS::MC_NPAIR], 2, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t1(iops.iopns[KEYS::MC_NPAIR], 8, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.o_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t1(iops.iopns[KEYS::MC_NPAIR], 8, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11, ovps.v_set[2][1].s_22);

  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], 1, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_22, ovps.v_set[1][1].s_11,
      ovps.v_set[2][0].s_21, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], 1, emp4, control, 0, rv, wgt,
      ovps.v_set[1][0].s_12, ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_22, ovps.o_set[1][1].s_11,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], 2, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], 2, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], 2, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], 2, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], 8, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], 8, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], 8, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], 8, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], -2, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_11, ovps.o_set[1][0].s_22, ovps.v_set[1][1].s_22, ovps.v_set[1][1].s_11,
      ovps.v_set[2][0].s_22, ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.o_set[2][1].s_22);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], -2, emp4, control, 0, rv, wgt,
      ovps.v_set[1][0].s_11, ovps.v_set[1][0].s_22, ovps.o_set[1][1].s_22, ovps.o_set[1][1].s_11,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], -4, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], -4, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_11, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], -4, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], -4, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_22);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], -4, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_21);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], -4, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], -4, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_21, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_12);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], -16, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.v_set[1][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][1].s_11, ovps.v_set[2][1].s_11);

  mcmp4_ij_helper_t3(iops.iopns[KEYS::MC_NPAIR], 2, emp4, control, 0, rv, wgt,
      ovps.v_set[1][0].s_22, ovps.o_set[1][1].s_11, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_22,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][0].s_12, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t3(iops.iopns[KEYS::MC_NPAIR], 2, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.o_set[1][1].s_21, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22,
      ovps.v_set[2][0].s_11, ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11);
  mcmp4_ij_helper_t3(iops.iopns[KEYS::MC_NPAIR], 8, emp4, control, 0, rv, wgt,
      ovps.v_set[1][0].s_11, ovps.o_set[1][1].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_22,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][0].s_22, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t3(iops.iopns[KEYS::MC_NPAIR], 8, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.o_set[1][1].s_21, ovps.v_set[1][1].s_12, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_12, ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11);
  mcmp4_ij_helper_t3(iops.iopns[KEYS::MC_NPAIR], -4, emp4, control, 0, rv, wgt,
      ovps.v_set[1][0].s_12, ovps.o_set[1][1].s_11, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_22,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][0].s_22, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t3(iops.iopns[KEYS::MC_NPAIR], -4, emp4, control, 0, rv, wgt,
      ovps.v_set[1][0].s_21, ovps.o_set[1][1].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_22,
      ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][0].s_12, ovps.v_set[2][1].s_11);
  mcmp4_ij_helper_t3(iops.iopns[KEYS::MC_NPAIR], -4, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.o_set[1][1].s_21, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22,
      ovps.v_set[2][0].s_12, ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][1].s_11);
  mcmp4_ij_helper_t3(iops.iopns[KEYS::MC_NPAIR], -4, emp4, control, 0, rv, wgt,
      ovps.o_set[1][0].s_22, ovps.o_set[1][1].s_21, ovps.v_set[1][1].s_12, ovps.v_set[1][1].s_21,
      ovps.v_set[2][0].s_11, ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][1].s_11);
}
void MP::mcmp4_energy_ik_fast(double& emp4, std::vector<double>& control) {
  std::vector<double> rv(iops.iopns[KEYS::MC_NPAIR]);
  std::vector<double> wgt(iops.iopns[KEYS::MC_NPAIR]);
  std::transform(el_pair_list.begin(), el_pair_list.end(), rv.begin(), [](el_pair_typ ept){return ept.rv;});
  std::transform(el_pair_list.begin(), el_pair_list.end(), wgt.begin(), [](el_pair_typ ept){return 1.0/ept.wgt;});


  /*
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], 1, emp4, control, 3, rv, wgt,
      ovps.o_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.o_set[1][1].s_11, ovps.o_set[1][1].s_22,
      ovps.v_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22);
      */
  mcmp4_ij_helper_t1(iops.iopns[KEYS::MC_NPAIR],   4, emp4, control, 3, rv, wgt,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_21,
      ovps.v_set[2][0].s_22, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t1(iops.iopns[KEYS::MC_NPAIR],   4, emp4, control, 3, rv, wgt,
      ovps.v_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t1(iops.iopns[KEYS::MC_NPAIR],   4, emp4, control, 3, rv, wgt,
      ovps.v_set[0][0].s_12, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_21, ovps.v_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t1(iops.iopns[KEYS::MC_NPAIR],  -2, emp4, control, 3, rv, wgt,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_21,
      ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22, ovps.v_set[2][2].s_12);
  mcmp4_ij_helper_t1(iops.iopns[KEYS::MC_NPAIR],  -2, emp4, control, 3, rv, wgt,
      ovps.v_set[0][0].s_12, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_21, ovps.v_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t1(iops.iopns[KEYS::MC_NPAIR],  -8, emp4, control, 3, rv, wgt,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_21,
      ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t1(iops.iopns[KEYS::MC_NPAIR],  -8, emp4, control, 3, rv, wgt,
      ovps.v_set[0][0].s_11, ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21,
      ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t1(iops.iopns[KEYS::MC_NPAIR],   4, emp4, control, 3, rv, wgt,
      ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_21,
      ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22, ovps.v_set[2][2].s_12);

  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],   1, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_22, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.v_set[2][0].s_21, ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],   1, emp4, control, 3, rv, wgt,
    ovps.v_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_22, ovps.v_set[1][1].s_11,
    ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],   2, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],   2, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],   2, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],   2, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],   8, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],   8, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],   8, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],   8, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],  -2, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_22, ovps.o_set[0][0].s_11, ovps.o_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.v_set[2][0].s_22, ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.o_set[2][2].s_22);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],  -2, emp4, control, 3, rv, wgt,
    ovps.v_set[0][0].s_22, ovps.v_set[0][0].s_11, ovps.v_set[1][1].s_22, ovps.v_set[1][1].s_11,
    ovps.o_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],  -4, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],  -4, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],  -4, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],  -4, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_21, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],  -4, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_22, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_21);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],  -4, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_11, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_22);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR],  -4, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_12);
  mcmp4_ij_helper_t2(iops.iopns[KEYS::MC_NPAIR], -16, emp4, control, 3, rv, wgt,
    ovps.o_set[0][0].s_22, ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_21,
    ovps.o_set[2][0].s_12, ovps.v_set[2][0].s_12, ovps.o_set[2][2].s_11, ovps.v_set[2][2].s_11);

  mcmp4_ij_helper_t3(iops.iopns[KEYS::MC_NPAIR],   4, emp4, control, 3, rv, wgt,
      ovps.o_set[0][0].s_21, ovps.o_set[1][1].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
      ovps.v_set[2][0].s_11, ovps.v_set[2][0].s_22, ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11);
  mcmp4_ij_helper_t3(iops.iopns[KEYS::MC_NPAIR],   4, emp4, control, 3, rv, wgt,
      ovps.o_set[0][0].s_22, ovps.o_set[1][1].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
      ovps.v_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11);
  mcmp4_ij_helper_t3(iops.iopns[KEYS::MC_NPAIR],   4, emp4, control, 3, rv, wgt,
      ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_22, ovps.v_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t3(iops.iopns[KEYS::MC_NPAIR],   4, emp4, control, 3, rv, wgt,
      ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_22, ovps.v_set[2][0].s_12, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t3(iops.iopns[KEYS::MC_NPAIR],  -2, emp4, control, 3, rv, wgt,
      ovps.o_set[0][0].s_22, ovps.o_set[1][1].s_22, ovps.v_set[1][1].s_21, ovps.o_set[1][1].s_11,
      ovps.v_set[2][0].s_11, ovps.v_set[2][0].s_22, ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11);
  mcmp4_ij_helper_t3(iops.iopns[KEYS::MC_NPAIR],  -2, emp4, control, 3, rv, wgt,
      ovps.v_set[0][0].s_22, ovps.v_set[1][1].s_11, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_22, ovps.v_set[2][0].s_12, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11);
  mcmp4_ij_helper_t3(iops.iopns[KEYS::MC_NPAIR],  -8, emp4, control, 3, rv, wgt,
      ovps.o_set[0][0].s_21, ovps.o_set[1][1].s_22, ovps.v_set[1][1].s_22, ovps.o_set[1][1].s_11,
      ovps.v_set[2][0].s_12, ovps.v_set[2][0].s_21, ovps.o_set[2][0].s_12, ovps.o_set[2][2].s_11);
  mcmp4_ij_helper_t3(iops.iopns[KEYS::MC_NPAIR],  -8, emp4, control, 3, rv, wgt,
      ovps.v_set[0][0].s_12, ovps.v_set[1][1].s_21, ovps.v_set[1][1].s_12, ovps.o_set[1][1].s_21,
      ovps.o_set[2][0].s_22, ovps.v_set[2][0].s_22, ovps.o_set[2][0].s_11, ovps.v_set[2][2].s_11);
}

void MP::mcmp4_energy(double& emp4, std::vector<double>& control) {
  emp4 = 0.0;
  std::fill(control.begin(), control.end(), 0.0);

  emp4 = 0.0;
  mcmp4_energy_ij_fast(emp4, control);
  mcmp4_energy_ik_fast(emp4, control);
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
      auto it_jt = it * iops.iopns[KEYS::MC_NPAIR] + jt;
      std::array<double, 36> en_kt, ct_kt;
      en_kt.fill(0); ct_kt.fill(0);
      for (auto kt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++) {
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;

        std::array<double, 36> en;

        #include "qc_mcmp4_ij_k.h"

        std::transform(en_kt.begin(), en_kt.end(), en.begin(), en_kt.begin(), [&](double x, double y) {return x + y * el_pair_list[kt].rv;});
        std::transform(ct_kt.begin(), ct_kt.end(), en.begin(), ct_kt.begin(), [&](double x, double y) {return x + y / el_pair_list[kt].wgt;});
      }
      {
        auto kt = it;
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;

        std::array<double, 36> en;

#include "qc_mcmp4_ij_k.h"

        std::transform(en_kt.begin(), en_kt.end(), en.begin(), en_kt.begin(), [&](double x, double y) {return x - y * el_pair_list[kt].rv;});
        std::transform(ct_kt.begin(), ct_kt.end(), en.begin(), ct_kt.begin(), [&](double x, double y) {return x + y / el_pair_list[kt].wgt;});
      }
      {
        auto kt = jt;
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;

        std::array<double, 36> en;

#include "qc_mcmp4_ij_k.h"

        std::transform(en_kt.begin(), en_kt.end(), en.begin(), en_kt.begin(), [&](double x, double y) {return x - y * el_pair_list[kt].rv;});
        std::transform(ct_kt.begin(), ct_kt.end(), en.begin(), ct_kt.begin(), [&](double x, double y) {return x + y / el_pair_list[kt].wgt;});
      }

      std::array<double, 36> en_lt, ct_lt;
      en_lt.fill(0); ct_lt.fill(0);
      for (auto lt = 0; lt < iops.iopns[KEYS::MC_NPAIR]; lt++) {
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;

        #include "qc_mcmp4_ij_l.h"

        std::transform(en_lt.begin(), en_lt.end(), en.begin(), en_lt.begin(), [&](double x, double y) {return x + y * el_pair_list[lt].rv;});
        std::transform(ct_lt.begin(), ct_lt.end(), en.begin(), ct_lt.begin(), [&](double x, double y) {return x + y / el_pair_list[lt].wgt;});
      }
      {
        auto lt = it;
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;
#include "qc_mcmp4_ij_l.h"

        std::transform(en_lt.begin(), en_lt.end(), en.begin(), en_lt.begin(), [&](double x, double y) {return x - y * el_pair_list[lt].rv;});
        std::transform(ct_lt.begin(), ct_lt.end(), en.begin(), ct_lt.begin(), [&](double x, double y) {return x - y / el_pair_list[lt].wgt;});
      }
      {
        auto lt = jt;
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        std::array<double, 36> en;

#include "qc_mcmp4_ij_l.h"

        std::transform(en_lt.begin(), en_lt.end(), en.begin(), en_lt.begin(), [&](double x, double y) {return x - y * el_pair_list[lt].rv;});
        std::transform(ct_lt.begin(), ct_lt.end(), en.begin(), ct_lt.begin(), [&](double x, double y) {return x - y / el_pair_list[lt].wgt;});
      }

      double en_corr = 0;
      double ct_corr = 0;
      for (auto kt = 0, lt = 0; kt < iops.iopns[KEYS::MC_NPAIR]; kt++, lt++) {
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        double en = 0;

        #include "qc_mcmp4_ij.h"

        en_corr += en * el_pair_list[kt].rv * el_pair_list[lt].rv;
        ct_corr += en * el_pair_list[kt].rv / el_pair_list[lt].wgt;
      }
      {
        auto kt = it;
        auto lt = it;
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        double en = 0;

#include "qc_mcmp4_ij.h"

        en_corr -= en * el_pair_list[kt].rv * el_pair_list[lt].rv;
        ct_corr -= en * el_pair_list[kt].rv / el_pair_list[lt].wgt;
      }
      {
        auto kt = jt;
        auto lt = jt;
        auto it_kt = it * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto jt_kt = jt * iops.iopns[KEYS::MC_NPAIR] + kt;
        auto it_lt = it * iops.iopns[KEYS::MC_NPAIR] + lt;
        auto jt_lt = jt * iops.iopns[KEYS::MC_NPAIR] + lt;

        double en = 0;

#include "qc_mcmp4_ij.h"

        en_corr -= en * el_pair_list[kt].rv * el_pair_list[lt].rv;
        ct_corr -= en * el_pair_list[kt].rv / el_pair_list[lt].wgt;
      }
      //printf("%12.6f", (en_kt[0] * ct_lt[0] - ct_corr) * 1000000);

      en_i += std::inner_product(en_kt.begin(), en_kt.end(), en_lt.begin(), -en_corr) * el_pair_list[jt].rv * (it != jt);
      ct_i += std::inner_product(en_kt.begin(), en_kt.end(), ct_lt.begin(), -ct_corr) * el_pair_list[jt].rv * (it != jt);
    }
    //printf("\n");
    emp4 += en_i * el_pair_list[it].rv;
    control[0] += en_i / el_pair_list[it].wgt;
    control[1] += ct_i / el_pair_list[it].wgt;
    control[2] += ct_i * el_pair_list[it].rv;
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
        // std::transform(ct_jt.begin(), ct_jt.end(), en.begin(), ct_jt.begin(), [&](double x, double y) {return x + y / el_pair_list[jt].wgt;});
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
        ct_corr += en * el_pair_list[jt].rv / el_pair_list[lt].wgt;
      }

      en_i += std::inner_product(en_jt.begin(), en_jt.end(), en_lt.begin(), -en_corr) * el_pair_list[kt].rv;
      ct_i += std::inner_product(en_jt.begin(), en_jt.end(), ct_lt.begin(), -ct_corr) * el_pair_list[kt].rv;
    }
    emp4 += en_i * el_pair_list[it].rv;
    control[3] += en_i / el_pair_list[it].wgt;
    control[4] += ct_i / el_pair_list[it].wgt;
    control[5] += ct_i * el_pair_list[it].rv;
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
    control[6] += en_i / el_pair_list[it].wgt;
    control[7] += ct_i / el_pair_list[it].wgt;
    control[8] += ct_i * el_pair_list[it].rv;
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
        ct_kl += ct_l_t * el_pair_list[kt].rv;
      }
      en_jkl += en_kl * el_pair_list[jt].rv;
      ct_jkl += ct_kl * el_pair_list[jt].rv;
    }
    emp4       += en_jkl * el_pair_list[it].rv;

    control[9] += ct_jkl / el_pair_list[it].wgt;
    control[10] += ct_jkl / el_pair_list[it].wgt;
    control[11] += en_jkl * el_pair_list[it].rv;
  }
}

