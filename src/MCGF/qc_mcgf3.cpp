#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cblas.h"
#include "../blas_calls.h"
#include "../qc_monte.h"
#include "qc_mcgf3.h"


void print_out(bool col_major, double* A, int rows, int cols) {
  if (col_major) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        printf("%8.3f", A[i + j * rows]);
      }
      printf("\n");
    }
  } else {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        printf("%8.3f", A[i * cols + j]);
      }
      printf("\n");
    }
  }
}

void vector_multiply(const double* B, double* A, int mc_pair_num) {
  for (int tidy = 0; tidy < mc_pair_num; tidy++) {
    for (int tidx = 0; tidx < mc_pair_num; tidx++) {
      int index = tidy * mc_pair_num + tidx;
      A[index] = A[index] * B[tidy] * B[tidx];
    }
  }
  for (int tidy = 0; tidy < mc_pair_num; tidy++) {
    int index = tidy * mc_pair_num + tidy;
    A[index] = 0;
  }
}
void prep_Tk(const double* Ti, const double* Tj,  const double* rv, double* Tk, int mc_pair_num) {
  for (int tidy = 0; tidy < mc_pair_num; tidy++) {
    for (int tidx = 0; tidx < mc_pair_num; tidx++) {
      int index = tidy * mc_pair_num + tidx;
      Tk[index] = 0;
      Tk[index] -= Ti[tidx*mc_pair_num + tidx] * Tj[tidy*mc_pair_num + tidx] * rv[tidx];
      Tk[index] -= Ti[tidx*mc_pair_num + tidy] * Tj[tidy*mc_pair_num + tidy] * rv[tidy];
    }
  }
}
void m_v_mul(const double* A, const double *B, double* C, int mc_pair_num) {
  for (int tidx = 0; tidx < mc_pair_num; tidx++) {
    for (int tidy = 0; tidy < mc_pair_num; tidy++) {
      int index = tidx * mc_pair_num + tidy;
      C[index] = A[index] * B[tidy];
    }
  }
}
void m_m_add_mul(double alpha, const double* A, const double *B, double* C, int mc_pair_num) {
  for (int tidx = 0; tidx < mc_pair_num; tidx++) {
    for (int tidy = 0; tidy < mc_pair_num; tidy++) {
      int index = tidx * mc_pair_num + tidy;
      C[index] = alpha * A[index] * B[index] + C[index];
    }
  }
}

void gf3_helper(
    double* Ti_a, double* Ti_b, double* Ti,
    double* Tj_a, double* Tj_b, double* Tj,
    double* rv,
    double* Tk_a, double* Tk,
    double* en,
    double c, int mc_pair_num) {
  std::transform(Ti_a, Ti_a + mc_pair_num*mc_pair_num, Ti_b, Ti, std::multiplies<>());
  std::transform(Tj_a, Tj_a + mc_pair_num*mc_pair_num, Tj_b, Tj, std::multiplies<>());
  prep_Tk(Ti, Tj, rv, Tk, mc_pair_num);
  m_v_mul(Tj, rv, Tj, mc_pair_num);
  cblas_dgemm(CblasColMajor,
      CblasTrans, CblasNoTrans,
      mc_pair_num, mc_pair_num, mc_pair_num,
      1.0,
      Ti, mc_pair_num,
      Tj, mc_pair_num,
      1.0,
      Tk, mc_pair_num);
  m_m_add_mul(c, Tk, Tk_a, en, mc_pair_num);
}
void gf3_core_1(OVPS_Host& ovps, OVPS_ARRAY& d_ovps, double *rv, int mc_pair_num, std::array<double*, 4>& T) {
  std::fill(d_ovps.en3_1mCore.begin(), d_ovps.en3_1mCore.end(), 0.0);
  std::fill(d_ovps.en3_1pCore.begin(), d_ovps.en3_1pCore.end(), 0.0);

  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_12.data(), T[0], ovps.o_set[1][1].s_21.data(), ovps.v_set[1][1].s_11.data(), T[1], rv, ovps.v_set[0][0].s_12.data(), T[2], d_ovps.en3_1pCore.data(), -2.00, mc_pair_num);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_21.data(), T[0], ovps.o_set[1][1].s_21.data(), ovps.v_set[1][1].s_12.data(), T[1], rv, ovps.v_set[0][0].s_11.data(), T[2], d_ovps.en3_1pCore.data(), -2.00, mc_pair_num);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_22.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), T[1], rv, ovps.v_set[0][0].s_21.data(), T[2], d_ovps.en3_1pCore.data(), -2.00, mc_pair_num);
  gf3_helper(ovps.v_set[1][0].s_12.data(), ovps.v_set[1][0].s_21.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.o_set[1][1].s_22.data(), T[1], rv, ovps.o_set[0][0].s_11.data(), T[2], d_ovps.en3_1pCore.data(), -1.00, mc_pair_num);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_11.data(), T[0], ovps.o_set[1][1].s_21.data(), ovps.v_set[1][1].s_12.data(), T[1], rv, ovps.v_set[0][0].s_12.data(), T[2], d_ovps.en3_1pCore.data(),  1.00, mc_pair_num);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_21.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_12.data(), T[1], rv, ovps.v_set[0][0].s_21.data(), T[2], d_ovps.en3_1pCore.data(),  1.00, mc_pair_num);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_22.data(), T[0], ovps.o_set[1][1].s_21.data(), ovps.v_set[1][1].s_11.data(), T[1], rv, ovps.v_set[0][0].s_11.data(), T[2], d_ovps.en3_1pCore.data(),  1.00, mc_pair_num);
  gf3_helper(ovps.v_set[1][0].s_11.data(), ovps.v_set[1][0].s_22.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.o_set[1][1].s_22.data(), T[1], rv, ovps.o_set[0][0].s_11.data(), T[2], d_ovps.en3_1pCore.data(),  2.00, mc_pair_num);

  gf3_helper(ovps.o_set[1][0].s_11.data(), ovps.o_set[1][0].s_22.data(), T[0], ovps.v_set[1][1].s_11.data(), ovps.v_set[1][1].s_22.data(), T[1], rv, ovps.v_set[0][0].s_11.data(), T[2], d_ovps.en3_1mCore.data(), -2.00, mc_pair_num);
  gf3_helper(ovps.o_set[1][0].s_22.data(), ovps.v_set[1][0].s_11.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_12.data(), T[1], rv, ovps.o_set[0][0].s_21.data(), T[2], d_ovps.en3_1mCore.data(), -1.00, mc_pair_num);
  gf3_helper(ovps.o_set[1][0].s_22.data(), ovps.v_set[1][0].s_12.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_21.data(), T[1], rv, ovps.o_set[0][0].s_11.data(), T[2], d_ovps.en3_1mCore.data(), -1.00, mc_pair_num);
  gf3_helper(ovps.o_set[1][0].s_11.data(), ovps.o_set[1][0].s_22.data(), T[0], ovps.v_set[1][1].s_21.data(), ovps.v_set[1][1].s_12.data(), T[1], rv, ovps.v_set[0][0].s_11.data(), T[2], d_ovps.en3_1mCore.data(),  1.00, mc_pair_num);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_11.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_22.data(), T[1], rv, ovps.o_set[0][0].s_12.data(), T[2], d_ovps.en3_1mCore.data(), -1.00, mc_pair_num);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_12.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_21.data(), T[1], rv, ovps.o_set[0][0].s_12.data(), T[2], d_ovps.en3_1mCore.data(),  2.00, mc_pair_num);
  gf3_helper(ovps.o_set[1][0].s_22.data(), ovps.v_set[1][0].s_11.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_22.data(), T[1], rv, ovps.o_set[0][0].s_11.data(), T[2], d_ovps.en3_1mCore.data(),  2.00, mc_pair_num);
  gf3_helper(ovps.o_set[1][0].s_22.data(), ovps.v_set[1][0].s_12.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), T[1], rv, ovps.o_set[0][0].s_21.data(), T[2], d_ovps.en3_1mCore.data(),  2.00, mc_pair_num);

//gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_12.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), T[1], rv, ovps.v_set[0][0].s_22.data(), T[2], d_ovps.en3_1pCore.data(),  4.00, mc_pair_num);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_12.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), T[1], rv, ovps.o_set[0][0].s_22.data(), T[2], d_ovps.en3_1mCore.data(), -4.00, mc_pair_num);
  m_m_add_mul(4.0, T[2], ovps.v_set[0][0].s_22.data(), d_ovps.en3_1pCore.data(), mc_pair_num);

//gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_11.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_12.data(), T[1], rv, ovps.v_set[0][0].s_22.data(), T[2], d_ovps.en3_1pCore.data(), -2.00, mc_pair_num);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_11.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_12.data(), T[1], rv, ovps.o_set[0][0].s_22.data(), T[2], d_ovps.en3_1mCore.data(),  2.00, mc_pair_num);
  m_m_add_mul(-2.0, T[2], ovps.v_set[0][0].s_22.data(), d_ovps.en3_1pCore.data(), mc_pair_num);

  vector_multiply(rv, d_ovps.en3_1mCore.data(), mc_pair_num);
  vector_multiply(rv, d_ovps.en3_1pCore.data(), mc_pair_num);
}
void gf3_core_2(OVPS_Host& ovps, OVPS_ARRAY& d_ovps, double *rv, int mc_pair_num, std::array<double*, 4>& T) {
  std::fill(d_ovps.en3_2mCore.begin(), d_ovps.en3_2mCore.end(), 0.0);
  std::fill(d_ovps.en3_2pCore.begin(), d_ovps.en3_2pCore.end(), 0.0);

  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_21.data(), T[1], rv, ovps.v_set[1][1].s_12.data(), T[2], d_ovps.en3_2pCore.data(), -2.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_11.data(), T[0], ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_22.data(), T[1], rv, ovps.v_set[1][1].s_11.data(), T[2], d_ovps.en3_2pCore.data(), -2.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_11.data(), T[1], rv, ovps.v_set[1][1].s_21.data(), T[2], d_ovps.en3_2pCore.data(), -2.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.o_set[0][0].s_22.data(), T[0], ovps.v_set[1][0].s_21.data(), ovps.v_set[1][0].s_12.data(), T[1], rv, ovps.o_set[1][1].s_11.data(), T[2], d_ovps.en3_2pCore.data(), -1.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_11.data(), T[0], ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_22.data(), T[1], rv, ovps.v_set[1][1].s_12.data(), T[2], d_ovps.en3_2pCore.data(),  1.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_11.data(), T[0], ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_12.data(), T[1], rv, ovps.v_set[1][1].s_21.data(), T[2], d_ovps.en3_2pCore.data(),  1.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_21.data(), T[1], rv, ovps.v_set[1][1].s_11.data(), T[2], d_ovps.en3_2pCore.data(),  1.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.o_set[0][0].s_22.data(), T[0], ovps.v_set[1][0].s_11.data(), ovps.v_set[1][0].s_22.data(), T[1], rv, ovps.o_set[1][1].s_11.data(), T[2], d_ovps.en3_2pCore.data(),  2.00, mc_pair_num);

  gf3_helper(ovps.v_set[0][0].s_11.data(), ovps.v_set[0][0].s_22.data(), T[0], ovps.o_set[1][0].s_11.data(), ovps.o_set[1][0].s_22.data(), T[1], rv, ovps.v_set[1][1].s_11.data(), T[2], d_ovps.en3_2mCore.data(), -2.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_11.data(), T[0], ovps.o_set[1][0].s_21.data(), ovps.v_set[1][0].s_12.data(), T[1], rv, ovps.o_set[1][1].s_12.data(), T[2], d_ovps.en3_2mCore.data(), -1.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_21.data(), T[0], ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_12.data(), T[1], rv, ovps.o_set[1][1].s_21.data(), T[2], d_ovps.en3_2mCore.data(), -1.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_22.data(), T[0], ovps.o_set[1][0].s_21.data(), ovps.v_set[1][0].s_11.data(), T[1], rv, ovps.o_set[1][1].s_11.data(), T[2], d_ovps.en3_2mCore.data(), -1.00, mc_pair_num);
  gf3_helper(ovps.v_set[0][0].s_21.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][0].s_11.data(), ovps.o_set[1][0].s_22.data(), T[1], rv, ovps.v_set[1][1].s_11.data(), T[2], d_ovps.en3_2mCore.data(),  1.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][0].s_21.data(), ovps.v_set[1][0].s_11.data(), T[1], rv, ovps.o_set[1][1].s_12.data(), T[2], d_ovps.en3_2mCore.data(),  2.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_21.data(), T[0], ovps.o_set[1][0].s_21.data(), ovps.v_set[1][0].s_12.data(), T[1], rv, ovps.o_set[1][1].s_11.data(), T[2], d_ovps.en3_2mCore.data(),  2.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_22.data(), T[0], ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_11.data(), T[1], rv, ovps.o_set[1][1].s_21.data(), T[2], d_ovps.en3_2mCore.data(),  2.00, mc_pair_num);

//gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_11.data(), T[1], rv, ovps.v_set[1][1].s_22.data(), T[2], d_ovps.en3_2pCore.data(),  4.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_11.data(), T[1], rv, ovps.o_set[1][1].s_22.data(), T[2], d_ovps.en3_2mCore.data(), -4.00, mc_pair_num);
  m_m_add_mul(4.0, T[2], ovps.v_set[1][1].s_22.data(), d_ovps.en3_2pCore.data(), mc_pair_num);

//gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_11.data(), T[0], ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_12.data(), T[1], rv, ovps.v_set[1][1].s_22.data(), T[2], d_ovps.en3_2pCore.data(), -2.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_11.data(), T[0], ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_12.data(), T[1], rv, ovps.o_set[1][1].s_22.data(), T[2], d_ovps.en3_2mCore.data(),  2.00, mc_pair_num);
  m_m_add_mul(-2.0, T[2], ovps.v_set[1][1].s_22.data(), d_ovps.en3_2pCore.data(), mc_pair_num);

  vector_multiply(rv, d_ovps.en3_2mCore.data(), mc_pair_num);
  vector_multiply(rv, d_ovps.en3_2pCore.data(), mc_pair_num);
}
void gf3_core_12(OVPS_Host& ovps, OVPS_ARRAY& d_ovps, double *rv, int mc_pair_num, std::array<double*, 4>& T) {
  std::fill(d_ovps.en3_12mCore.begin(), d_ovps.en3_12mCore.end(), 0.0);
  std::fill(d_ovps.en3_12pCore.begin(), d_ovps.en3_12pCore.end(), 0.0);

  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.v_set[0][0].s_22.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_22.data(), T[1], rv, ovps.v_set[1][0].s_11.data(), T[2], d_ovps.en3_12pCore.data(), -2.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_21.data(), T[1], rv, ovps.v_set[1][0].s_12.data(), T[2], d_ovps.en3_12pCore.data(), -2.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_22.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), T[1], rv, ovps.v_set[1][0].s_21.data(), T[2], d_ovps.en3_12pCore.data(), -2.00, mc_pair_num);
  gf3_helper(ovps.v_set[0][0].s_12.data(), ovps.v_set[0][0].s_21.data(), T[0], ovps.v_set[1][1].s_11.data(), ovps.v_set[1][1].s_22.data(), T[1], rv, ovps.o_set[1][0].s_11.data(), T[2], d_ovps.en3_12pCore.data(), -1.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_22.data(), T[1], rv, ovps.v_set[1][0].s_12.data(), T[2], d_ovps.en3_12pCore.data(),  1.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_22.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_21.data(), T[1], rv, ovps.v_set[1][0].s_11.data(), T[2], d_ovps.en3_12pCore.data(),  1.00, mc_pair_num);
  gf3_helper(ovps.v_set[0][0].s_11.data(), ovps.v_set[0][0].s_22.data(), T[0], ovps.v_set[1][1].s_11.data(), ovps.v_set[1][1].s_22.data(), T[1], rv, ovps.o_set[1][0].s_11.data(), T[2], d_ovps.en3_12pCore.data(),  2.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.v_set[0][0].s_22.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_12.data(), T[1], rv, ovps.v_set[1][0].s_21.data(), T[2], d_ovps.en3_12pCore.data(),  1.00, mc_pair_num);

  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.o_set[0][0].s_22.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.o_set[1][1].s_22.data(), T[1], rv, ovps.v_set[1][0].s_11.data(), T[2], d_ovps.en3_12mCore.data(), -2.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][1].s_21.data(), ovps.v_set[1][1].s_12.data(), T[1], rv, ovps.o_set[1][0].s_12.data(), T[2], d_ovps.en3_12mCore.data(), -1.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_21.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_12.data(), T[1], rv, ovps.o_set[1][0].s_21.data(), T[2], d_ovps.en3_12mCore.data(), -1.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][1].s_21.data(), ovps.v_set[1][1].s_11.data(), T[1], rv, ovps.o_set[1][0].s_11.data(), T[2], d_ovps.en3_12mCore.data(), -1.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.o_set[0][0].s_22.data(), T[0], ovps.o_set[1][1].s_21.data(), ovps.o_set[1][1].s_12.data(), T[1], rv, ovps.v_set[1][0].s_11.data(), T[2], d_ovps.en3_12mCore.data(),  1.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][1].s_21.data(), ovps.v_set[1][1].s_11.data(), T[1], rv, ovps.o_set[1][0].s_12.data(), T[2], d_ovps.en3_12mCore.data(),  2.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_21.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][1].s_21.data(), ovps.v_set[1][1].s_12.data(), T[1], rv, ovps.o_set[1][0].s_11.data(), T[2], d_ovps.en3_12mCore.data(),  2.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), T[1], rv, ovps.o_set[1][0].s_21.data(), T[2], d_ovps.en3_12mCore.data(),  2.00, mc_pair_num);

//gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), T[1], rv, ovps.v_set[1][0].s_22.data(), T[2], d_ovps.en3_12pCore.data(),  4.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), T[1], rv, ovps.o_set[1][0].s_22.data(), T[2], d_ovps.en3_12mCore.data(), -4.00, mc_pair_num);
  m_m_add_mul(4.0, T[2], ovps.v_set[1][0].s_22.data(), d_ovps.en3_12pCore.data(), mc_pair_num);

//gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_12.data(), T[1], rv, ovps.v_set[1][0].s_22.data(), T[2], d_ovps.en3_12pCore.data(), -2.00, mc_pair_num);
  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.v_set[0][0].s_12.data(), T[0], ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_12.data(), T[1], rv, ovps.o_set[1][0].s_22.data(), T[2], d_ovps.en3_12mCore.data(),  2.00, mc_pair_num);
  m_m_add_mul(-2.0, T[2], ovps.v_set[1][0].s_22.data(), d_ovps.en3_12pCore.data(), mc_pair_num);

  vector_multiply(rv, d_ovps.en3_12mCore.data(), mc_pair_num);
  vector_multiply(rv, d_ovps.en3_12pCore.data(), mc_pair_num);
}

void gf3_helper_c(
    double* Ti,
    double* Tj_a, double* Tj_b, double* Tj_c, double* Tj,
    double* rv, double* Tk_a, double* Tk_b, double* Tk,
    double* en_12, double c12,
    double* en_22, double c22,
    int mc_pair_num) {
  std::transform(Tj_a, Tj_a + mc_pair_num*mc_pair_num, Tj_b, Tj, std::multiplies<>());
  std::transform(Tj_c, Tj_c + mc_pair_num*mc_pair_num, Tj, Tj, std::multiplies<>());
  prep_Tk(Ti, Tj, rv, Tk, mc_pair_num);
  m_v_mul(Tj, rv, Tj, mc_pair_num);
  cblas_dgemm(CblasColMajor,
      CblasTrans, CblasNoTrans,
      mc_pair_num, mc_pair_num, mc_pair_num,
      1.0,
      Ti, mc_pair_num,
      Tj, mc_pair_num,
      1.0,
      Tk, mc_pair_num);
  m_m_add_mul(c12, Tk, Tk_a, en_12, mc_pair_num);
  m_m_add_mul(c22, Tk, Tk_b, en_22, mc_pair_num);
}
void gf3_core_c(OVPS_Host& ovps, OVPS_ARRAY& d_ovps, double *rv, int mc_pair_num, std::array<double*, 4>& T) {
  std::fill(d_ovps.en3_12cCore.begin(), d_ovps.en3_12cCore.end(), 0.0);
  std::fill(d_ovps.en3_22cCore.begin(), d_ovps.en3_22cCore.end(), 0.0);

  gf3_helper_c(ovps.v_set[1][0].s_12.data(), ovps.o_set[1][1].s_11.data(), ovps.o_set[1][1].s_22.data(), ovps.v_set[1][1].s_11.data(), T[0], rv, ovps.o_set[0][0].s_22.data(), ovps.o_set[0][0].s_21.data(), T[1], d_ovps.en3_12cCore.data(),  2.00,  d_ovps.en3_22cCore.data(), -4.00, mc_pair_num);
  gf3_helper_c(ovps.v_set[1][0].s_11.data(), ovps.o_set[1][1].s_11.data(), ovps.o_set[1][1].s_22.data(), ovps.v_set[1][1].s_12.data(), T[0], rv, ovps.o_set[0][0].s_22.data(), ovps.o_set[0][0].s_21.data(), T[1], d_ovps.en3_12cCore.data(), -1.00,  d_ovps.en3_22cCore.data(),  2.00, mc_pair_num);
  gf3_helper_c(ovps.o_set[1][0].s_12.data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), ovps.v_set[1][1].s_22.data(), T[0], rv, ovps.v_set[0][0].s_22.data(), ovps.v_set[0][0].s_21.data(), T[1], d_ovps.en3_12cCore.data(), -2.00,  d_ovps.en3_22cCore.data(),  4.00, mc_pair_num);
  gf3_helper_c(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][1].s_12.data(), ovps.v_set[1][1].s_21.data(), ovps.o_set[1][1].s_11.data(), T[0], rv, ovps.v_set[0][0].s_22.data(), ovps.v_set[0][0].s_21.data(), T[1], d_ovps.en3_12cCore.data(),  1.00,  d_ovps.en3_22cCore.data(), -2.00, mc_pair_num);

  gf3_helper_c(ovps.o_set[1][1].s_11.data(), ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_11.data(), ovps.v_set[1][0].s_22.data(), T[0], rv, ovps.o_set[0][0].s_22.data(), ovps.o_set[0][0].s_21.data(), T[1], d_ovps.en3_12cCore.data(), -1.00,  d_ovps.en3_22cCore.data(),  2.00, mc_pair_num);
  gf3_helper_c(ovps.o_set[1][1].s_11.data(), ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_12.data(), ovps.v_set[1][0].s_21.data(), T[0], rv, ovps.o_set[0][0].s_22.data(), ovps.o_set[0][0].s_21.data(), T[1], d_ovps.en3_12cCore.data(),  2.00,  d_ovps.en3_22cCore.data(), -4.00, mc_pair_num);
  gf3_helper_c(ovps.v_set[1][1].s_11.data(), ovps.o_set[1][0].s_11.data(), ovps.o_set[1][0].s_22.data(), ovps.v_set[1][0].s_12.data(), T[0], rv, ovps.v_set[0][0].s_22.data(), ovps.v_set[0][0].s_21.data(), T[1], d_ovps.en3_12cCore.data(),  1.00,  d_ovps.en3_22cCore.data(), -2.00, mc_pair_num);
  gf3_helper_c(ovps.v_set[1][1].s_11.data(), ovps.o_set[1][0].s_11.data(), ovps.o_set[1][0].s_22.data(), ovps.v_set[1][0].s_22.data(), T[0], rv, ovps.v_set[0][0].s_12.data(), ovps.v_set[0][0].s_11.data(), T[1], d_ovps.en3_12cCore.data(), -2.00,  d_ovps.en3_22cCore.data(),  4.00, mc_pair_num);

  gf3_helper_c(ovps.o_set[1][1].s_11.data(), ovps.o_set[0][0].s_11.data(), ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_22.data(), T[0], rv, ovps.v_set[1][0].s_12.data(), ovps.v_set[1][0].s_11.data(), T[1], d_ovps.en3_12cCore.data(),  2.00,  d_ovps.en3_22cCore.data(), -4.00, mc_pair_num);
  gf3_helper_c(ovps.o_set[1][1].s_11.data(), ovps.o_set[0][0].s_11.data(), ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_12.data(), T[0], rv, ovps.v_set[1][0].s_22.data(), ovps.v_set[1][0].s_21.data(), T[1], d_ovps.en3_12cCore.data(), -1.00,  d_ovps.en3_22cCore.data(),  2.00, mc_pair_num);
  gf3_helper_c(ovps.v_set[1][1].s_11.data(), ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_11.data(), ovps.v_set[0][0].s_22.data(), T[0], rv, ovps.o_set[1][0].s_12.data(), ovps.o_set[1][0].s_11.data(), T[1], d_ovps.en3_12cCore.data(), -2.00,  d_ovps.en3_22cCore.data(),  4.00, mc_pair_num);
  gf3_helper_c(ovps.v_set[1][1].s_11.data(), ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_12.data(), ovps.v_set[0][0].s_21.data(), T[0], rv, ovps.o_set[1][0].s_12.data(), ovps.o_set[1][0].s_11.data(), T[1], d_ovps.en3_12cCore.data(),  1.00,  d_ovps.en3_22cCore.data(), -2.00, mc_pair_num);

  vector_multiply(rv, d_ovps.en3_12cCore.data(), mc_pair_num);
  vector_multiply(rv, d_ovps.en3_22cCore.data(), mc_pair_num);
}

void strided_transform(
    const size_t N,
    double alpha,
    const double *A, const size_t incA,
    const double *B, const size_t incB,
    double beta, double *C, const size_t incC) {
  for (size_t idx = 0, idxA = 0, idxB = 0, idxC = 0; idx < N; idx++, idxA += incA, idxB += incB, idxC += incC) {
    C[idxC] = alpha * A[idxA] * B[idxB] + beta * C[idxC];
  }
}

void GF::mcgf3_local_energy_core() {
  std::array<double*, 4> T{};
  T[0] = new double[iops.iopns[KEYS::ELECTRON_PAIRS] * iops.iopns[KEYS::ELECTRON_PAIRS]];
  T[1] = new double[iops.iopns[KEYS::ELECTRON_PAIRS] * iops.iopns[KEYS::ELECTRON_PAIRS]];
  T[2] = new double[iops.iopns[KEYS::ELECTRON_PAIRS] * iops.iopns[KEYS::ELECTRON_PAIRS]];
  T[3] = new double[iops.iopns[KEYS::ELECTRON_PAIRS]];

  std::copy(electron_pair_list->rv.begin(), electron_pair_list->rv.end(), T[3]);

  gf3_core_c (ovps, d_ovps, T[3], iops.iopns[KEYS::ELECTRON_PAIRS], T);
  gf3_core_1 (ovps, d_ovps, T[3], iops.iopns[KEYS::ELECTRON_PAIRS], T);
  gf3_core_2 (ovps, d_ovps, T[3], iops.iopns[KEYS::ELECTRON_PAIRS], T);
  gf3_core_12(ovps, d_ovps, T[3], iops.iopns[KEYS::ELECTRON_PAIRS], T);

  cblas_dgemv(CblasColMajor, CblasNoTrans,
      iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRON_PAIRS],
      1.0,
      d_ovps.en3_12cCore.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
      d_ovps.one.data(), 1,
      0.0,
      d_ovps.en3c12.data(), 1);

  cblas_dgemv(CblasColMajor, CblasNoTrans,
      iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRON_PAIRS],
      1.0,
      d_ovps.en3_22cCore.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
      d_ovps.one.data(), 1,
      0.0,
      d_ovps.en3c22.data(), 1);

  for (auto &it : T) {
    delete [] it;
  }
}
void GF::mcgf3_local_energy(std::vector<std::vector<double>>& egf3) {
  auto nsamp = static_cast<double>(iops.iopns[KEYS::ELECTRON_PAIRS]);
  nsamp = nsamp * (nsamp - 1.0) * (nsamp - 2.0);
  for (int band = 0; band < numBand; band++) {
    double en3 = 0;
    double alpha, beta;
    const double *psi1;
    const double *psi2;
    if (band-offBand < 0) {
      psi1 = wavefunctions[WC::electron_pairs_1].occ() + (band+iocc2-iocc1-offBand);
      psi2 = wavefunctions[WC::electron_pairs_2].occ() + (band+iocc2-iocc1-offBand);
    } else {
      psi1 = wavefunctions[WC::electron_pairs_1].vir() + (band-offBand);
      psi2 = wavefunctions[WC::electron_pairs_2].vir() + (band-offBand);
    }

    strided_transform(iops.iopns[KEYS::ELECTRON_PAIRS], 1.0, d_ovps.en3c12.data(), 1, psi1, wavefunctions[WC::electron_pairs_1].lda, 0.0, d_ovps.ent.data(), 1);
    strided_transform(iops.iopns[KEYS::ELECTRON_PAIRS], 1.0, d_ovps.en3c22.data(), 1, psi2, wavefunctions[WC::electron_pairs_2].lda, 1.0, d_ovps.ent.data(), 1);

    // ent = ovps.ovps.tg_val1[band] * en3_1pCore . psi
    alpha = tau->get_gfn_tau(0, 0, band-offBand, false);
    beta = 1.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRON_PAIRS],
        alpha,
        d_ovps.en3_1pCore.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta, d_ovps.ent.data(), 1);

    // ent = ovps.ovps.tg_val2[band] * en3_2pCore . psi + ent
    alpha = tau->get_gfn_tau(1, 1, band-offBand, false);
    beta = 1;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRON_PAIRS],
        alpha,
        d_ovps.en3_2pCore.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        d_ovps.ent.data(), 1);

    // ent = ovps.ovps.tg_val12[band] * en3_12pCore . psi + ent
    alpha = tau->get_gfn_tau(1, 0, band-offBand, false);
    beta = 1;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRON_PAIRS],
        alpha,
        d_ovps.en3_12pCore.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        d_ovps.ent.data(), 1);

    // ent = ovps.ovps.tgc_val1[band] * en3_1mCore . psi + ent
    alpha = tau->get_gfn_tau(0, 0, band-offBand, true);
    beta = 1;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRON_PAIRS],
        alpha,
        d_ovps.en3_1mCore.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        d_ovps.ent.data(), 1);

    // ent = ovps.ovps.tgc_val2[band] * en3_2mCore . psi + ent
    alpha = tau->get_gfn_tau(1, 1, band-offBand, true);
    beta = 1;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRON_PAIRS],
        alpha,
        d_ovps.en3_2mCore.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        d_ovps.ent.data(), 1);

    // ent = ovps.ovps.tgc_val12[band] * en3_12mCore . psi + ent
    alpha = tau->get_gfn_tau(1, 0, band-offBand, true);
    beta = 1;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRON_PAIRS],
        alpha,
        d_ovps.en3_12mCore.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        d_ovps.ent.data(), 1);

    // en2 = psi2 . ent
    en3 += cblas_ddot(iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        d_ovps.ent.data(), 1);

    en3 = en3 * tau->get_wgt(2) / nsamp;
    egf3[band].front() += en3;
  }
}
void GF::mcgf3_local_energy_diff(std::vector<std::vector<double>>& egf3) {
  auto nsamp = static_cast<double>(iops.iopns[KEYS::ELECTRON_PAIRS]);
  nsamp = nsamp * (nsamp - 1.0) * (nsamp - 2.0);
  for (int band = 0; band < numBand; band++) {
    int ip, dp;
    std::array<double, 7> en3{0, 0, 0, 0, 0, 0, 0};
    double en3t;
    double alpha, beta;
    const double *psi1;
    const double *psi2;
    if (band - offBand < 0) {
      psi1 = wavefunctions[WC::electron_pairs_1].occ() + (band + iocc2 - iocc1 - offBand);
      psi2 = wavefunctions[WC::electron_pairs_2].occ() + (band + iocc2 - iocc1 - offBand);
    } else {
      psi1 = wavefunctions[WC::electron_pairs_1].vir() + (band - offBand);
      psi2 = wavefunctions[WC::electron_pairs_2].vir() + (band - offBand);
    }

    strided_transform(iops.iopns[KEYS::ELECTRON_PAIRS], 1.0, d_ovps.en3c12.data(), 1, psi1, wavefunctions[WC::electron_pairs_1].lda, 0.0, d_ovps.ent.data(), 1);
    strided_transform(iops.iopns[KEYS::ELECTRON_PAIRS], 1.0, d_ovps.en3c22.data(), 1, psi2, wavefunctions[WC::electron_pairs_2].lda, 1.0, d_ovps.ent.data(), 1);
    en3[0] = cblas_ddot(iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        d_ovps.ent.data(), 1);

    // ent = en3_1pCore . psi
    alpha = 1.0;
    beta = 0.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRON_PAIRS],
        alpha,
        d_ovps.en3_1pCore.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        d_ovps.ent.data(), 1);
    // en2 = psi2 . ent
    en3t = cblas_ddot(iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        d_ovps.ent.data(), 1);
    en3[1] = en3[1] + en3t * tau->get_gfn_tau(0, 0, band - offBand, false);

    // ent = en3_2pCore . psi
    alpha = 1.0;
    beta = 0.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRON_PAIRS],
        alpha,
        d_ovps.en3_2pCore.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        d_ovps.ent.data(), 1);
    // en2 = psi2 . ent
    en3t = cblas_ddot(iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        d_ovps.ent.data(), 1);
    en3[2] = en3[2] + en3t * tau->get_gfn_tau(1, 1, band - offBand, false);

    // ent = en3_12pCore . psi
    alpha = 1.0;
    beta = 0.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRON_PAIRS],
        alpha,
        d_ovps.en3_12pCore.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        d_ovps.ent.data(), 1);
    // en2 = psi2 . ent
    en3t = cblas_ddot(iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        d_ovps.ent.data(), 1);
    en3[3] = en3[3] + en3t * tau->get_gfn_tau(1, 0, band - offBand, false);


    // ent = en3_1mCore . psi
    alpha = 1.0;
    beta = 0.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRON_PAIRS],
        alpha,
        d_ovps.en3_1mCore.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        d_ovps.ent.data(), 1);
    // en2 = psi2 . ent
    en3t = cblas_ddot(iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        d_ovps.ent.data(), 1);
    en3[4] = en3[4] + en3t * tau->get_gfn_tau(0, 0, band - offBand, true);

    // ent = en3_2mCore . psi
    alpha = 1.0;
    beta = 0.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRON_PAIRS],
        alpha,
        d_ovps.en3_2mCore.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        d_ovps.ent.data(), 1);
    // en2 = psi2 . ent
    en3t = cblas_ddot(iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        d_ovps.ent.data(), 1);
    en3[5] = en3[5] + en3t * tau->get_gfn_tau(1, 1, band - offBand, true);

    // ent = en3_12mCore . psi
    alpha = 1.0;
    beta = 0.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRON_PAIRS],
        alpha,
        d_ovps.en3_12mCore.data(), iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        d_ovps.ent.data(), 1);
    // en2 = psi2 . ent
    en3t = cblas_ddot(iops.iopns[KEYS::ELECTRON_PAIRS],
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        d_ovps.ent.data(), 1);
    en3[6] = en3[6] + en3t * tau->get_gfn_tau(1, 0, band - offBand, true);

    for (auto &it : en3) {
      it = it * tau->get_wgt(2) / nsamp;
    }

    for (ip = 0; ip < iops.iopns[KEYS::DIFFS]; ip++) {
      if (ip == 0) {
        for (dp = 0; dp < 3; dp++) {
          egf3[band][ip] += en3[dp + 1] + en3[dp + 4];
        }
        egf3[band][ip] += en3[0];
      } else if (ip % 2 == 1) {
        for (dp = 0; dp < 3; dp++) {
          egf3[band][ip] += en3[dp + 1] - en3[dp + 4];
        }
      } else if (ip % 2 == 0) {
        for (dp = 0; dp < 3; dp++) {
          egf3[band][ip] += en3[dp + 1] + en3[dp + 4];
        }
      }
      en3[1] = en3[1] * tau->get_tau(0);
      en3[2] = en3[2] * tau->get_tau(1);
      en3[3] = en3[3] * (tau->get_tau(0) + tau->get_tau(1));
      en3[4] = en3[4] * tau->get_tau(0);
      en3[5] = en3[5] * tau->get_tau(1);
      en3[6] = en3[6] * (tau->get_tau(0) + tau->get_tau(1));
    }
  }
}

void mcgf_full_helper(
    int m, int n,
    double a1, double b1, double a2, double b2,
    double* enCore,
    const double* psi, int psi_lda,
    double* ent,
    double* en) {
  // ent = alpha en3_2pCore . psi2
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
      m, n, m,
      a1,
      enCore, m,
      psi, psi_lda,
      b1,
      ent, m);

  // en3_2p = Tranpsose[psi2] . ent
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
      n, n, m,
      a2,
      psi, psi_lda,
      ent, m,
      b2,
      en, n);
}
void GF::mcgf3_local_energy_full(int band) {
  double nsamp;
  double alpha, beta;

  nsamp = static_cast<double>(iops.iopns[KEYS::ELECTRON_PAIRS]);
  nsamp = nsamp * (nsamp - 1.0);

  // enCore = alpha en2pCore + beta en2mCore
  alpha = tau->get_gfn_tau(0, 0, band - offBand, false) * tau->get_wgt(1) / nsamp;
  beta = tau->get_gfn_tau(0, 0, band - offBand, true) * tau->get_wgt(1) / nsamp;
  std::transform(d_ovps.en2pCore.begin(),
      d_ovps.en2pCore.end(),
      d_ovps.en2mCore.begin(),
      d_ovps.enCore.begin(),
      [&](double a, double b) {return alpha*a + beta*b;});

  nsamp = static_cast<double>(iops.iopns[KEYS::ELECTRON_PAIRS]);
  nsamp = nsamp * (nsamp - 1.0) * (nsamp - 2.0);

  // enCore = alpha en3_1p + enCore
  alpha = tau->get_gfn_tau(0, 0, band - offBand, false) * tau->get_wgt(2) / nsamp;
  std::transform(d_ovps.en3_1pCore.begin(),
      d_ovps.en3_1pCore.end(),
      d_ovps.enCore.begin(),
      d_ovps.enCore.begin(),
      [&](double a, double b) {return alpha*a + b;});

  // enCore = alpha en3_2p + enCore
  alpha = tau->get_gfn_tau(1, 1, band - offBand, false) * tau->get_wgt(2) / nsamp;
  std::transform(d_ovps.en3_2pCore.begin(),
      d_ovps.en3_2pCore.end(),
      d_ovps.enCore.begin(),
      d_ovps.enCore.begin(),
      [&](double a, double b) {return alpha*a + b;});

  // enCore = alpha en3_12p + enCore
  alpha = tau->get_gfn_tau(1, 0, band - offBand, false) * tau->get_wgt(2) / nsamp;
  std::transform(d_ovps.en3_12pCore.begin(),
      d_ovps.en3_12pCore.end(),
      d_ovps.enCore.begin(),
      d_ovps.enCore.begin(),
      [&](double a, double b) {return alpha*a + b;});

  // enCore = alpha en3_1m + enCore
  alpha = tau->get_gfn_tau(0, 0, band - offBand, true) * tau->get_wgt(2) / nsamp;
  std::transform(d_ovps.en3_1mCore.begin(),
      d_ovps.en3_1mCore.end(),
      d_ovps.enCore.begin(),
      d_ovps.enCore.begin(),
      [&](double a, double b) {return alpha*a + b;});

  // enCore = alpha en3_2m + enCore
  alpha = tau->get_gfn_tau(1, 1, band - offBand, true) * tau->get_wgt(2) / nsamp;
  std::transform(d_ovps.en3_2mCore.begin(),
      d_ovps.en3_2mCore.end(),
      d_ovps.enCore.begin(),
      d_ovps.enCore.begin(),
      [&](double a, double b) {return alpha*a + b;});

  // enCore = alpha en3_12m + enCore
  alpha = tau->get_gfn_tau(1, 0, band - offBand, true) * tau->get_wgt(2) / nsamp;
  std::transform(d_ovps.en3_12mCore.begin(),
      d_ovps.en3_12mCore.end(),
      d_ovps.enCore.begin(),
      d_ovps.enCore.begin(),
      [&](double a, double b) {return alpha*a + b;});

  mcgf_full_helper(
      iops.iopns[KEYS::ELECTRON_PAIRS], ivir2-iocc1,
      1.00, 0.00,
      1.00, 1.00,
      d_ovps.enCore.data(),
      wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
      d_ovps.ent.data(),
      d_ovps.enBlock[band][0].data());

  // ent = diag[enc12] . psi1
  Ddgmm(DDGMM_SIDE_RIGHT,
      ivir2-iocc1, iops.iopns[KEYS::ELECTRON_PAIRS], 
      wavefunctions[WC::electron_pairs_1].occ(), wavefunctions[WC::electron_pairs_1].lda,
      d_ovps.en3c12.data(), 1,
      d_ovps.ent.data(), ivir2 - iocc1);

  // en3 = Transpose[psi2] . ent + en3
  // alpha = 1.00;
  alpha = tau->get_wgt(2) / nsamp;
  beta  = 1.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 
      ivir2-iocc1, ivir2-iocc1, iops.iopns[KEYS::ELECTRON_PAIRS], alpha,
      wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
      d_ovps.ent.data(), ivir2 - iocc1,
      beta, d_ovps.enBlock[band][0].data(), ivir2-iocc1);

  // ent = diag[en3c22] . psi2
  Ddgmm(DDGMM_SIDE_RIGHT,
      ivir2-iocc1, iops.iopns[KEYS::ELECTRON_PAIRS], 
      wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
      d_ovps.en3c22.data(), 1,
      d_ovps.ent.data(), ivir2 - iocc1);

  // en3 = Transpose[psi2] . ent + en3
  alpha = tau->get_wgt(2) / nsamp;
  beta  = 1.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 
      ivir2-iocc1, ivir2-iocc1, iops.iopns[KEYS::ELECTRON_PAIRS], alpha,
      wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
      d_ovps.ent.data(), ivir2 - iocc1,
      beta, d_ovps.enBlock[band][0].data(), ivir2-iocc1);
}
void GF::mcgf3_local_energy_full_diff(int band) {
  double nsamp, nsamp2;
  double alpha, beta;

  nsamp = static_cast<double>(iops.iopns[KEYS::ELECTRON_PAIRS]);
  nsamp2 = nsamp * (nsamp - 1.0);
  nsamp = nsamp * (nsamp - 1.0) * (nsamp - 2.0);

  alpha = tau->get_gfn_tau(0, 0, band - offBand, false) * tau->get_wgt(1) / nsamp2;
  beta = tau->get_gfn_tau(0, 0, band - offBand, false) * tau->get_wgt(2) / nsamp;
  std::transform(d_ovps.en2pCore.begin(),
                 d_ovps.en2pCore.end(),
                 d_ovps.en3_1pCore.begin(),
                 d_ovps.enCore.begin(),
                 [&](double a, double b) {return alpha*a + beta*b;});
  mcgf_full_helper(
      iops.iopns[KEYS::ELECTRON_PAIRS], ivir2-iocc1,
      1.00, 0.00,
      1.00, 0.00,
      d_ovps.enCore.data(),
      wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
      d_ovps.ent.data(),
      d_ovps.en3_1p);

  // ent = alpha en3_2pCore.data() . psi2
  mcgf_full_helper(
      iops.iopns[KEYS::ELECTRON_PAIRS], ivir2-iocc1,
      tau->get_gfn_tau(1, 1, band - offBand, false) * tau->get_wgt(2) / nsamp, 0.00,
      1.00, 0.00,
      d_ovps.en3_2pCore.data(),
      wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
      d_ovps.ent.data(),
      d_ovps.en3_2p);

  // ent = alpha en3_12pCore.data() . psi2
  mcgf_full_helper(
      iops.iopns[KEYS::ELECTRON_PAIRS], ivir2-iocc1,
      tau->get_gfn_tau(1, 0, band - offBand, false) * tau->get_wgt(2) / nsamp, 0.00,
      1.00, 0.00,
      d_ovps.en3_12pCore.data(),
      wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
      d_ovps.ent.data(),
      d_ovps.en3_12p);

  alpha = tau->get_gfn_tau(0, 0, band - offBand, true) * tau->get_wgt(1) / nsamp2;
  beta = tau->get_gfn_tau(0, 0, band - offBand, true) * tau->get_wgt(2) / nsamp;
  std::transform(d_ovps.en2mCore.begin(),
                 d_ovps.en2mCore.end(),
                 d_ovps.en3_1mCore.begin(),
                 d_ovps.enCore.begin(),
                 [&](double a, double b) {return alpha*a + beta*b;});
  mcgf_full_helper(
      iops.iopns[KEYS::ELECTRON_PAIRS], ivir2-iocc1,
      1.00, 0.00,
      1.00, 0.00,
      d_ovps.enCore.data(),
      wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
      d_ovps.ent.data(),
      d_ovps.en3_1m);

  mcgf_full_helper(
      iops.iopns[KEYS::ELECTRON_PAIRS], ivir2-iocc1,
      tau->get_gfn_tau(1, 1, band - offBand, true) * tau->get_wgt(2) / nsamp, 0.00,
      1.00, 0.00,
      d_ovps.en3_2mCore.data(),
      wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
      d_ovps.ent.data(),
      d_ovps.en3_2m);

  mcgf_full_helper(
      iops.iopns[KEYS::ELECTRON_PAIRS], ivir2-iocc1,
      tau->get_gfn_tau(1, 0, band - offBand, true) * tau->get_wgt(2) / nsamp, 0.00,
      1.00, 0.00,
      d_ovps.en3_12mCore.data(),
      wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
      d_ovps.ent.data(),
      d_ovps.en3_12m);

  // ent = diag[enc12] . psi1
  Ddgmm(DDGMM_SIDE_RIGHT,
      ivir2-iocc1, iops.iopns[KEYS::ELECTRON_PAIRS], 
      wavefunctions[WC::electron_pairs_1].occ(), wavefunctions[WC::electron_pairs_1].lda,
      d_ovps.en3c12.data(), 1,
      d_ovps.ent.data(), ivir2 - iocc1);

  // en3 = Transpose[psi2] . ent + en3
  // alpha = 1.00;
  alpha = tau->get_wgt(2) / nsamp;
  beta  = 0.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 
      ivir2-iocc1, ivir2-iocc1, iops.iopns[KEYS::ELECTRON_PAIRS], alpha,
      wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
      d_ovps.ent.data(), ivir2 - iocc1,
      beta, d_ovps.en3_c, ivir2-iocc1);

  // ent = diag[en3c22] . psi2
  Ddgmm(DDGMM_SIDE_RIGHT,
      ivir2-iocc1, iops.iopns[KEYS::ELECTRON_PAIRS], 
      wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
      d_ovps.en3c22.data(), 1,
      d_ovps.ent.data(), ivir2 - iocc1);

  // en3 = Transpose[psi2] . ent + en3
  alpha = tau->get_wgt(2) / nsamp;
  beta  = 1.00;
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 
      ivir2-iocc1, ivir2-iocc1, iops.iopns[KEYS::ELECTRON_PAIRS], alpha,
      wavefunctions[WC::electron_pairs_2].occ(), wavefunctions[WC::electron_pairs_2].lda,
      d_ovps.ent.data(), ivir2 - iocc1,
      beta, d_ovps.en3_c, ivir2-iocc1);
}



GF3_Functional::GF3_Functional(IOPs& iops) :
  MCGF(iops, 2, "23", false),
  one(n_electron_pairs, 1.0),
  en3c12(n_electron_pairs),
  en3c22(n_electron_pairs),
  en3_1pCore (n_electron_pairs * n_electron_pairs),
  en3_2pCore (n_electron_pairs * n_electron_pairs),
  en3_12pCore(n_electron_pairs * n_electron_pairs),
  en3_1mCore (n_electron_pairs * n_electron_pairs),
  en3_2mCore (n_electron_pairs * n_electron_pairs),
  en3_12mCore(n_electron_pairs * n_electron_pairs),
  en3_12cCore(n_electron_pairs * n_electron_pairs),
  en3_22cCore(n_electron_pairs * n_electron_pairs)
{
  ent.resize(n_electron_pairs);
  nsamp = static_cast<double>(n_electron_pairs);
  nsamp = nsamp * (nsamp - 1.0) * (nsamp - 2.0);

  T[0].resize(n_electron_pairs * n_electron_pairs);
  T[1].resize(n_electron_pairs * n_electron_pairs);
  T[2].resize(n_electron_pairs * n_electron_pairs);
  T[3].resize(n_electron_pairs);

  if (iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULL || 
        iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULLDIFF) {
    std::cerr << "Full rountine not integrated into MCGF class\n";
    exit(0);
    // ent.resize((basis.ivir2 - basis.iocc1) * n_electron_pairs);
  }
}

void GF3_Functional::gf3_core_1(OVPS_Host& ovps, Electron_Pair_List* electron_pair_list) {
  std::fill(en3_1mCore.begin(), en3_1mCore.end(), 0.0);
  std::fill(en3_1pCore.begin(), en3_1pCore.end(), 0.0);

  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_21.data(), ovps.v_set[1][1].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[0][0].s_12.data(), T[2].data(), en3_1pCore.data(), -2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_21.data(), T[0].data(), ovps.o_set[1][1].s_21.data(), ovps.v_set[1][1].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[0][0].s_11.data(), T[2].data(), en3_1pCore.data(), -2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_22.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[0][0].s_21.data(), T[2].data(), en3_1pCore.data(), -2.00, n_electron_pairs);
  gf3_helper(ovps.v_set[1][0].s_12.data(), ovps.v_set[1][0].s_21.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.o_set[1][1].s_22.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[0][0].s_11.data(), T[2].data(), en3_1pCore.data(), -1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_11.data(), T[0].data(), ovps.o_set[1][1].s_21.data(), ovps.v_set[1][1].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[0][0].s_12.data(), T[2].data(), en3_1pCore.data(),  1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_21.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[0][0].s_21.data(), T[2].data(), en3_1pCore.data(),  1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_22.data(), T[0].data(), ovps.o_set[1][1].s_21.data(), ovps.v_set[1][1].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[0][0].s_11.data(), T[2].data(), en3_1pCore.data(),  1.00, n_electron_pairs);
  gf3_helper(ovps.v_set[1][0].s_11.data(), ovps.v_set[1][0].s_22.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.o_set[1][1].s_22.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[0][0].s_11.data(), T[2].data(), en3_1pCore.data(),  2.00, n_electron_pairs);

  gf3_helper(ovps.o_set[1][0].s_11.data(), ovps.o_set[1][0].s_22.data(), T[0].data(), ovps.v_set[1][1].s_11.data(), ovps.v_set[1][1].s_22.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[0][0].s_11.data(), T[2].data(), en3_1mCore.data(), -2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[1][0].s_22.data(), ovps.v_set[1][0].s_11.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[0][0].s_21.data(), T[2].data(), en3_1mCore.data(), -1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[1][0].s_22.data(), ovps.v_set[1][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_21.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[0][0].s_11.data(), T[2].data(), en3_1mCore.data(), -1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[1][0].s_11.data(), ovps.o_set[1][0].s_22.data(), T[0].data(), ovps.v_set[1][1].s_21.data(), ovps.v_set[1][1].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[0][0].s_11.data(), T[2].data(), en3_1mCore.data(),  1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_11.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_22.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[0][0].s_12.data(), T[2].data(), en3_1mCore.data(), -1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_21.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[0][0].s_12.data(), T[2].data(), en3_1mCore.data(),  2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[1][0].s_22.data(), ovps.v_set[1][0].s_11.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_22.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[0][0].s_11.data(), T[2].data(), en3_1mCore.data(),  2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[1][0].s_22.data(), ovps.v_set[1][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[0][0].s_21.data(), T[2].data(), en3_1mCore.data(),  2.00, n_electron_pairs);

//gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[0][0].s_22.data(), T[2].data(), en3_1pCore.data(),  4.00, n_electron_pairs);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[0][0].s_22.data(), T[2].data(), en3_1mCore.data(), -4.00, n_electron_pairs);
  m_m_add_mul(4.0, T[2].data(), ovps.v_set[0][0].s_22.data(), en3_1pCore.data(), n_electron_pairs);

//gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_11.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[0][0].s_22.data(), T[2].data(), en3_1pCore.data(), -2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_11.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[0][0].s_22.data(), T[2].data(), en3_1mCore.data(),  2.00, n_electron_pairs);
  m_m_add_mul(-2.0, T[2].data(), ovps.v_set[0][0].s_22.data(), en3_1pCore.data(), n_electron_pairs);

  vector_multiply(electron_pair_list->rv.data(), en3_1mCore.data(), n_electron_pairs);
  vector_multiply(electron_pair_list->rv.data(), en3_1pCore.data(), n_electron_pairs);
}
void GF3_Functional::gf3_core_2(OVPS_Host& ovps, Electron_Pair_List* electron_pair_list) {
  std::fill(en3_2mCore.begin(), en3_2mCore.end(), 0.0);
  std::fill(en3_2pCore.begin(), en3_2pCore.end(), 0.0);

  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_21.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][1].s_12.data(), T[2].data(), en3_2pCore.data(), -2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_11.data(), T[0].data(), ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_22.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][1].s_11.data(), T[2].data(), en3_2pCore.data(), -2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][1].s_21.data(), T[2].data(), en3_2pCore.data(), -2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.o_set[0][0].s_22.data(), T[0].data(), ovps.v_set[1][0].s_21.data(), ovps.v_set[1][0].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][1].s_11.data(), T[2].data(), en3_2pCore.data(), -1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_11.data(), T[0].data(), ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_22.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][1].s_12.data(), T[2].data(), en3_2pCore.data(),  1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_11.data(), T[0].data(), ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][1].s_21.data(), T[2].data(), en3_2pCore.data(),  1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_21.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][1].s_11.data(), T[2].data(), en3_2pCore.data(),  1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.o_set[0][0].s_22.data(), T[0].data(), ovps.v_set[1][0].s_11.data(), ovps.v_set[1][0].s_22.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][1].s_11.data(), T[2].data(), en3_2pCore.data(),  2.00, n_electron_pairs);

  gf3_helper(ovps.v_set[0][0].s_11.data(), ovps.v_set[0][0].s_22.data(), T[0].data(), ovps.o_set[1][0].s_11.data(), ovps.o_set[1][0].s_22.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][1].s_11.data(), T[2].data(), en3_2mCore.data(), -2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_11.data(), T[0].data(), ovps.o_set[1][0].s_21.data(), ovps.v_set[1][0].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][1].s_12.data(), T[2].data(), en3_2mCore.data(), -1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_21.data(), T[0].data(), ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][1].s_21.data(), T[2].data(), en3_2mCore.data(), -1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_22.data(), T[0].data(), ovps.o_set[1][0].s_21.data(), ovps.v_set[1][0].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][1].s_11.data(), T[2].data(), en3_2mCore.data(), -1.00, n_electron_pairs);
  gf3_helper(ovps.v_set[0][0].s_21.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][0].s_11.data(), ovps.o_set[1][0].s_22.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][1].s_11.data(), T[2].data(), en3_2mCore.data(),  1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][0].s_21.data(), ovps.v_set[1][0].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][1].s_12.data(), T[2].data(), en3_2mCore.data(),  2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_21.data(), T[0].data(), ovps.o_set[1][0].s_21.data(), ovps.v_set[1][0].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][1].s_11.data(), T[2].data(), en3_2mCore.data(),  2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_22.data(), T[0].data(), ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][1].s_21.data(), T[2].data(), en3_2mCore.data(),  2.00, n_electron_pairs);

//gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][1].s_22.data(), T[2].data(), en3_2pCore.data(),  4.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][1].s_22.data(), T[2].data(), en3_2mCore.data(), -4.00, n_electron_pairs);
  m_m_add_mul(4.0, T[2].data(), ovps.v_set[1][1].s_22.data(), en3_2pCore.data(), n_electron_pairs);

//gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_11.data(), T[0].data(), ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][1].s_22.data(), T[2].data(), en3_2pCore.data(), -2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_11.data(), T[0].data(), ovps.o_set[1][0].s_11.data(), ovps.v_set[1][0].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][1].s_22.data(), T[2].data(), en3_2mCore.data(),  2.00, n_electron_pairs);
  m_m_add_mul(-2.0, T[2].data(), ovps.v_set[1][1].s_22.data(), en3_2pCore.data(), n_electron_pairs);

  vector_multiply(electron_pair_list->rv.data(), en3_2mCore.data(), n_electron_pairs);
  vector_multiply(electron_pair_list->rv.data(), en3_2pCore.data(), n_electron_pairs);
}
void GF3_Functional::gf3_core_12(OVPS_Host& ovps, Electron_Pair_List* electron_pair_list) {
  std::fill(en3_12mCore.begin(), en3_12mCore.end(), 0.0);
  std::fill(en3_12pCore.begin(), en3_12pCore.end(), 0.0);

  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.v_set[0][0].s_22.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_22.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][0].s_11.data(), T[2].data(), en3_12pCore.data(), -2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_21.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][0].s_12.data(), T[2].data(), en3_12pCore.data(), -2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_22.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][0].s_21.data(), T[2].data(), en3_12pCore.data(), -2.00, n_electron_pairs);
  gf3_helper(ovps.v_set[0][0].s_12.data(), ovps.v_set[0][0].s_21.data(), T[0].data(), ovps.v_set[1][1].s_11.data(), ovps.v_set[1][1].s_22.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][0].s_11.data(), T[2].data(), en3_12pCore.data(), -1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_22.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][0].s_12.data(), T[2].data(), en3_12pCore.data(),  1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_22.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_21.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][0].s_11.data(), T[2].data(), en3_12pCore.data(),  1.00, n_electron_pairs);
  gf3_helper(ovps.v_set[0][0].s_11.data(), ovps.v_set[0][0].s_22.data(), T[0].data(), ovps.v_set[1][1].s_11.data(), ovps.v_set[1][1].s_22.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][0].s_11.data(), T[2].data(), en3_12pCore.data(),  2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.v_set[0][0].s_22.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][0].s_21.data(), T[2].data(), en3_12pCore.data(),  1.00, n_electron_pairs);

  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.o_set[0][0].s_22.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.o_set[1][1].s_22.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][0].s_11.data(), T[2].data(), en3_12mCore.data(), -2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_21.data(), ovps.v_set[1][1].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][0].s_12.data(), T[2].data(), en3_12mCore.data(), -1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_21.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][0].s_21.data(), T[2].data(), en3_12mCore.data(), -1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_21.data(), ovps.v_set[1][1].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][0].s_11.data(), T[2].data(), en3_12mCore.data(), -1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.o_set[0][0].s_22.data(), T[0].data(), ovps.o_set[1][1].s_21.data(), ovps.o_set[1][1].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][0].s_11.data(), T[2].data(), en3_12mCore.data(),  1.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_21.data(), ovps.v_set[1][1].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][0].s_12.data(), T[2].data(), en3_12mCore.data(),  2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_21.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_21.data(), ovps.v_set[1][1].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][0].s_11.data(), T[2].data(), en3_12mCore.data(),  2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][0].s_21.data(), T[2].data(), en3_12mCore.data(),  2.00, n_electron_pairs);

//gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][0].s_22.data(), T[2].data(), en3_12pCore.data(),  4.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_12.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][0].s_22.data(), T[2].data(), en3_12mCore.data(), -4.00, n_electron_pairs);
  m_m_add_mul(4.0, T[2].data(), ovps.v_set[1][0].s_22.data(), en3_12pCore.data(), n_electron_pairs);

//gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.v_set[1][0].s_22.data(), T[2].data(), en3_12pCore.data(), -2.00, n_electron_pairs);
  gf3_helper(ovps.o_set[0][0].s_11.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_12.data(), T[1].data(), electron_pair_list->rv.data(), ovps.o_set[1][0].s_22.data(), T[2].data(), en3_12mCore.data(),  2.00, n_electron_pairs);
  m_m_add_mul(-2.0, T[2].data(), ovps.v_set[1][0].s_22.data(), en3_12pCore.data(), n_electron_pairs);

  vector_multiply(electron_pair_list->rv.data(), en3_12mCore.data(), n_electron_pairs);
  vector_multiply(electron_pair_list->rv.data(), en3_12pCore.data(), n_electron_pairs);
}
void GF3_Functional::gf3_core_c(OVPS_Host& ovps, Electron_Pair_List* electron_pair_list) {
  std::fill(en3_12cCore.begin(), en3_12cCore.end(), 0.0);
  std::fill(en3_22cCore.begin(), en3_22cCore.end(), 0.0);

  gf3_helper_c(ovps.v_set[1][0].s_12.data(), ovps.o_set[1][1].s_11.data(), ovps.o_set[1][1].s_22.data(), ovps.v_set[1][1].s_11.data(), T[0].data(), electron_pair_list->rv.data(), ovps.o_set[0][0].s_22.data(), ovps.o_set[0][0].s_21.data(), T[1].data(), en3_12cCore.data(),  2.00,  en3_22cCore.data(), -4.00, n_electron_pairs);
  gf3_helper_c(ovps.v_set[1][0].s_11.data(), ovps.o_set[1][1].s_11.data(), ovps.o_set[1][1].s_22.data(), ovps.v_set[1][1].s_12.data(), T[0].data(), electron_pair_list->rv.data(), ovps.o_set[0][0].s_22.data(), ovps.o_set[0][0].s_21.data(), T[1].data(), en3_12cCore.data(), -1.00,  en3_22cCore.data(),  2.00, n_electron_pairs);
  gf3_helper_c(ovps.o_set[1][0].s_12.data(), ovps.o_set[1][1].s_11.data(), ovps.v_set[1][1].s_11.data(), ovps.v_set[1][1].s_22.data(), T[0].data(), electron_pair_list->rv.data(), ovps.v_set[0][0].s_22.data(), ovps.v_set[0][0].s_21.data(), T[1].data(), en3_12cCore.data(), -2.00,  en3_22cCore.data(),  4.00, n_electron_pairs);
  gf3_helper_c(ovps.o_set[1][0].s_12.data(), ovps.v_set[1][1].s_12.data(), ovps.v_set[1][1].s_21.data(), ovps.o_set[1][1].s_11.data(), T[0].data(), electron_pair_list->rv.data(), ovps.v_set[0][0].s_22.data(), ovps.v_set[0][0].s_21.data(), T[1].data(), en3_12cCore.data(),  1.00,  en3_22cCore.data(), -2.00, n_electron_pairs);

  gf3_helper_c(ovps.o_set[1][1].s_11.data(), ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_11.data(), ovps.v_set[1][0].s_22.data(), T[0].data(), electron_pair_list->rv.data(), ovps.o_set[0][0].s_22.data(), ovps.o_set[0][0].s_21.data(), T[1].data(), en3_12cCore.data(), -1.00,  en3_22cCore.data(),  2.00, n_electron_pairs);
  gf3_helper_c(ovps.o_set[1][1].s_11.data(), ovps.o_set[1][0].s_12.data(), ovps.v_set[1][0].s_12.data(), ovps.v_set[1][0].s_21.data(), T[0].data(), electron_pair_list->rv.data(), ovps.o_set[0][0].s_22.data(), ovps.o_set[0][0].s_21.data(), T[1].data(), en3_12cCore.data(),  2.00,  en3_22cCore.data(), -4.00, n_electron_pairs);
  gf3_helper_c(ovps.v_set[1][1].s_11.data(), ovps.o_set[1][0].s_11.data(), ovps.o_set[1][0].s_22.data(), ovps.v_set[1][0].s_12.data(), T[0].data(), electron_pair_list->rv.data(), ovps.v_set[0][0].s_22.data(), ovps.v_set[0][0].s_21.data(), T[1].data(), en3_12cCore.data(),  1.00,  en3_22cCore.data(), -2.00, n_electron_pairs);
  gf3_helper_c(ovps.v_set[1][1].s_11.data(), ovps.o_set[1][0].s_11.data(), ovps.o_set[1][0].s_22.data(), ovps.v_set[1][0].s_22.data(), T[0].data(), electron_pair_list->rv.data(), ovps.v_set[0][0].s_12.data(), ovps.v_set[0][0].s_11.data(), T[1].data(), en3_12cCore.data(), -2.00,  en3_22cCore.data(),  4.00, n_electron_pairs);

  gf3_helper_c(ovps.o_set[1][1].s_11.data(), ovps.o_set[0][0].s_11.data(), ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_22.data(), T[0].data(), electron_pair_list->rv.data(), ovps.v_set[1][0].s_12.data(), ovps.v_set[1][0].s_11.data(), T[1].data(), en3_12cCore.data(),  2.00,  en3_22cCore.data(), -4.00, n_electron_pairs);
  gf3_helper_c(ovps.o_set[1][1].s_11.data(), ovps.o_set[0][0].s_11.data(), ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_12.data(), T[0].data(), electron_pair_list->rv.data(), ovps.v_set[1][0].s_22.data(), ovps.v_set[1][0].s_21.data(), T[1].data(), en3_12cCore.data(), -1.00,  en3_22cCore.data(),  2.00, n_electron_pairs);
  gf3_helper_c(ovps.v_set[1][1].s_11.data(), ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_11.data(), ovps.v_set[0][0].s_22.data(), T[0].data(), electron_pair_list->rv.data(), ovps.o_set[1][0].s_12.data(), ovps.o_set[1][0].s_11.data(), T[1].data(), en3_12cCore.data(), -2.00,  en3_22cCore.data(),  4.00, n_electron_pairs);
  gf3_helper_c(ovps.v_set[1][1].s_11.data(), ovps.o_set[0][0].s_22.data(), ovps.v_set[0][0].s_12.data(), ovps.v_set[0][0].s_21.data(), T[0].data(), electron_pair_list->rv.data(), ovps.o_set[1][0].s_12.data(), ovps.o_set[1][0].s_11.data(), T[1].data(), en3_12cCore.data(),  1.00,  en3_22cCore.data(), -2.00, n_electron_pairs);

  vector_multiply(electron_pair_list->rv.data(), en3_12cCore.data(), n_electron_pairs);
  vector_multiply(electron_pair_list->rv.data(), en3_22cCore.data(), n_electron_pairs);
}
void GF3_Functional::core(OVPS_Host& ovps, Electron_Pair_List* electron_pair_list) {
  rv = electron_pair_list->rv.data();

  gf3_core_c(ovps, electron_pair_list);
  gf3_core_1(ovps, electron_pair_list);
  gf3_core_2(ovps, electron_pair_list);
  gf3_core_12(ovps, electron_pair_list);

  cblas_dgemv(CblasColMajor, CblasNoTrans,
      n_electron_pairs, n_electron_pairs,
      1.0,
      en3_12cCore.data(), n_electron_pairs,
      one.data(), 1,
      0.0,
      en3c12.data(), 1);

  cblas_dgemv(CblasColMajor, CblasNoTrans,
      n_electron_pairs, n_electron_pairs,
      1.0,
      en3_22cCore.data(), n_electron_pairs,
      one.data(), 1,
      0.0,
      en3c22.data(), 1);
}

void GF3_Functional::energy_no_diff(std::vector<std::vector<double>>& egf3, 
       std::unordered_map<int, Wavefunction_Type>& wavefunctions,
       Electron_Pair_List* electron_pair_list, Tau* tau) {
  for (int band = 0; band < numBand; band++) {
    double en3 = 0;
    double alpha, beta;
    const double *psi1;
    const double *psi2;
    psi1 = wavefunctions[WC::electron_pairs_1].vir() + (band-offBand);
    psi2 = wavefunctions[WC::electron_pairs_2].vir() + (band-offBand);

    strided_transform(n_electron_pairs, 1.0, en3c12.data(), 1, psi1, wavefunctions[WC::electron_pairs_1].lda, 0.0, ent.data(), 1);
    strided_transform(n_electron_pairs, 1.0, en3c22.data(), 1, psi2, wavefunctions[WC::electron_pairs_2].lda, 1.0, ent.data(), 1);

    // ent = ovps.ovps.tg_val1[band] * en3_1pCore . psi
    alpha = tau->get_gfn_tau(0, 0, band-offBand, false);
    beta = 1.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        n_electron_pairs, n_electron_pairs,
        alpha,
        en3_1pCore.data(), n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        ent.data(), 1);

    // ent = ovps.ovps.tg_val2[band] * en3_2pCore . psi + ent
    alpha = tau->get_gfn_tau(1, 1, band-offBand, false);
    beta = 1;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        n_electron_pairs, n_electron_pairs,
        alpha,
        en3_2pCore.data(), n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        ent.data(), 1);

    // ent = ovps.ovps.tg_val12[band] * en3_12pCore . psi + ent
    alpha = tau->get_gfn_tau(1, 0, band-offBand, false);
    beta = 1;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        n_electron_pairs, n_electron_pairs,
        alpha,
        en3_12pCore.data(), n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        ent.data(), 1);

    // ent = ovps.ovps.tgc_val1[band] * en3_1mCore . psi + ent
    alpha = tau->get_gfn_tau(0, 0, band-offBand, true);
    beta = 1;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        n_electron_pairs, n_electron_pairs,
        alpha,
        en3_1mCore.data(), n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        ent.data(), 1);

    // ent = ovps.ovps.tgc_val2[band] * en3_2mCore . psi + ent
    alpha = tau->get_gfn_tau(1, 1, band-offBand, true);
    beta = 1;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        n_electron_pairs, n_electron_pairs,
        alpha,
        en3_2mCore.data(), n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        ent.data(), 1);

    // ent = ovps.ovps.tgc_val12[band] * en3_12mCore . psi + ent
    alpha = tau->get_gfn_tau(1, 0, band-offBand, true);
    beta = 1;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        n_electron_pairs, n_electron_pairs,
        alpha,
        en3_12mCore.data(), n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        ent.data(), 1);

    // en2 = psi2 . ent
    en3 += cblas_ddot(n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        ent.data(), 1);

    en3 = en3 * tau->get_wgt(2) / nsamp;
    egf3[band].front() += en3;
  }
}

void GF3_Functional::energy_diff(std::vector<std::vector<double>>& egf3, 
       std::unordered_map<int, Wavefunction_Type>& wavefunctions,
       Electron_Pair_List* electron_pair_list, Tau* tau) {
  for (int band = 0; band < numBand; band++) {
    int ip, dp;
    std::array<double, 7> en3{0, 0, 0, 0, 0, 0, 0};
    double en3t;
    double alpha, beta;
    const double *psi1;
    const double *psi2;
    psi1 = wavefunctions[WC::electron_pairs_1].vir() + (band-offBand);
    psi2 = wavefunctions[WC::electron_pairs_2].vir() + (band-offBand);

    strided_transform(n_electron_pairs, 1.0, en3c12.data(), 1, psi1, wavefunctions[WC::electron_pairs_1].lda, 0.0, ent.data(), 1);
    strided_transform(n_electron_pairs, 1.0, en3c22.data(), 1, psi2, wavefunctions[WC::electron_pairs_2].lda, 1.0, ent.data(), 1);
    en3[0] = cblas_ddot(n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        ent.data(), 1);

    // ent.data() = en3_1pCore . psi
    alpha = 1.0;
    beta = 0.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        n_electron_pairs, n_electron_pairs,
        alpha,
        en3_1pCore.data(), n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        ent.data(), 1);
    // en2 = psi2 . ent.data()
    en3t = cblas_ddot(n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        ent.data(), 1);
    en3[1] = en3[1] + en3t * tau->get_gfn_tau(0, 0, band - offBand, false);

    // ent.data() = en3_2pCore . psi
    alpha = 1.0;
    beta = 0.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        n_electron_pairs, n_electron_pairs,
        alpha,
        en3_2pCore.data(), n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        ent.data(), 1);
    // en2 = psi2 . ent.data()
    en3t = cblas_ddot(n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        ent.data(), 1);
    en3[2] = en3[2] + en3t * tau->get_gfn_tau(1, 1, band - offBand, false);

    // ent.data() = en3_12pCore . psi
    alpha = 1.0;
    beta = 0.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        n_electron_pairs, n_electron_pairs,
        alpha,
        en3_12pCore.data(), n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        ent.data(), 1);
    // en2 = psi2 . ent.data()
    en3t = cblas_ddot(n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        ent.data(), 1);
    en3[3] = en3[3] + en3t * tau->get_gfn_tau(1, 0, band - offBand, false);


    // ent.data() = en3_1mCore . psi
    alpha = 1.0;
    beta = 0.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        n_electron_pairs, n_electron_pairs,
        alpha,
        en3_1mCore.data(), n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        ent.data(), 1);
    // en2 = psi2 . ent.data()
    en3t = cblas_ddot(n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        ent.data(), 1);
    en3[4] = en3[4] + en3t * tau->get_gfn_tau(0, 0, band - offBand, true);

    // ent.data() = en3_2mCore . psi
    alpha = 1.0;
    beta = 0.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        n_electron_pairs, n_electron_pairs,
        alpha,
        en3_2mCore.data(), n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        ent.data(), 1);
    // en2 = psi2 . ent.data()
    en3t = cblas_ddot(n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        ent.data(), 1);
    en3[5] = en3[5] + en3t * tau->get_gfn_tau(1, 1, band - offBand, true);

    // ent.data() = en3_12mCore . psi
    alpha = 1.0;
    beta = 0.0;
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        n_electron_pairs, n_electron_pairs,
        alpha,
        en3_12mCore.data(), n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        beta,
        ent.data(), 1);
    // en2 = psi2 . ent.data()
    en3t = cblas_ddot(n_electron_pairs,
        psi2, wavefunctions[WC::electron_pairs_2].lda,
        ent.data(), 1);
    en3[6] = en3[6] + en3t * tau->get_gfn_tau(1, 0, band - offBand, true);

    for (auto &it : en3) {
      it = it * tau->get_wgt(2) / nsamp;
    }

    for (ip = 0; ip < numDiff; ip++) {
      if (ip == 0) {
        for (dp = 0; dp < 3; dp++) {
          egf3[band][ip] += en3[dp + 1] + en3[dp + 4];
        }
        egf3[band][ip] += en3[0];
      } else if (ip % 2 == 1) {
        for (dp = 0; dp < 3; dp++) {
          egf3[band][ip] += en3[dp + 1] - en3[dp + 4];
        }
      } else if (ip % 2 == 0) {
        for (dp = 0; dp < 3; dp++) {
          egf3[band][ip] += en3[dp + 1] + en3[dp + 4];
        }
      }
      en3[1] = en3[1] * tau->get_tau(0);
      en3[2] = en3[2] * tau->get_tau(1);
      en3[3] = en3[3] * (tau->get_tau(0) + tau->get_tau(1));
      en3[4] = en3[4] * tau->get_tau(0);
      en3[5] = en3[5] * tau->get_tau(1);
      en3[6] = en3[6] * (tau->get_tau(0) + tau->get_tau(1));
    }
  }
}

void GF3_Functional::energy_f12(std::vector<std::vector<double>>&, 
   std::unordered_map<int, Wavefunction_Type>&,
   Electron_Pair_List*, Electron_List*) {}


/*
void GF::mc_gf3_func(double* en3, int ip, int jp, int kp, int band) {
  //  std::fill(en3,en3+7,0);
  //
  //  int ijIndex = ip * iops.iopns[KEYS::ELECTRON_PAIRS] + jp;
  //  int ikIndex = ip * iops.iopns[KEYS::ELECTRON_PAIRS] + kp;
  //  int jkIndex = jp * iops.iopns[KEYS::ELECTRON_PAIRS] + kp;
  //
  //  int ijbIndex = (band*iops.iopns[KEYS::ELECTRON_PAIRS] + ip)*iops.iopns[KEYS::ELECTRON_PAIRS] + jp;
  //  int ikbIndex = (band*iops.iopns[KEYS::ELECTRON_PAIRS] + ip)*iops.iopns[KEYS::ELECTRON_PAIRS] + kp;
  //  int jkbIndex = (band*iops.iopns[KEYS::ELECTRON_PAIRS] + jp)*iops.iopns[KEYS::ELECTRON_PAIRS] + kp;
  //
  //  int ibIndex = band*iops.iopns[KEYS::ELECTRON_PAIRS] + ip;
  //  int jbIndex = band*iops.iopns[KEYS::ELECTRON_PAIRS] + jp;
  //  int kbIndex = band*iops.iopns[KEYS::ELECTRON_PAIRS] + kp;

  //12/34
  //  en3[0] = en3[0] + 2.00 * ovps.vs_15[ikIndex] * ovps.vs_26[ikIndex] * ovps.os_24[ijIndex] * ovps.os_46[jkIndex] * ovps.os_35[jkIndex] * ovps.ps_13[ijbIndex];
  //  en3[0] = en3[0] - 1.00 * ovps.vs_15[ikIndex] * ovps.vs_26[ikIndex] * ovps.os_24[ijIndex] * ovps.os_45[jkIndex] * ovps.os_36[jkIndex] * ovps.ps_13[ijbIndex];
  //  en3[0] = en3[0] - 2.00 * ovps.vs_24[ijIndex] * ovps.vs_15[ikIndex] * ovps.vs_46[jkIndex] * ovps.os_26[ikIndex] * ovps.os_35[jkIndex] * ovps.ps_13[ijbIndex];
  //  en3[0] = en3[0] + 1.00 * ovps.vs_25[ikIndex] * ovps.vs_14[ijIndex] * ovps.vs_46[jkIndex] * ovps.os_26[ikIndex] * ovps.os_35[jkIndex] * ovps.ps_13[ijbIndex];
  //  en3[0] = en3[0] + 1.00 * ovps.vs_23[ijIndex] * ovps.vs_15[ikIndex] * ovps.vs_46[jkIndex] * ovps.os_26[ikIndex] * ovps.os_45[jkIndex] * ovps.ps_13[ijbIndex];
  //  en3[0] = en3[0] - 2.00 * ovps.vs_25[ikIndex] * ovps.vs_13[ijIndex] * ovps.vs_46[jkIndex] * ovps.os_26[ikIndex] * ovps.os_45[jkIndex] * ovps.ps_13[ijbIndex];
  //  en3[0] = en3[0] + 1.00 * ovps.vs_24[ijIndex] * ovps.vs_15[ikIndex] * ovps.vs_46[jkIndex] * ovps.os_25[ikIndex] * ovps.os_36[jkIndex] * ovps.ps_13[ijbIndex];
  //  en3[0] = en3[0] - 2.00 * ovps.vs_25[ikIndex] * ovps.vs_14[ijIndex] * ovps.vs_46[jkIndex] * ovps.os_25[ikIndex] * ovps.os_36[jkIndex] * ovps.ps_13[ijbIndex];
  //  en3[0] = en3[0] - 2.00 * ovps.vs_23[ijIndex] * ovps.vs_15[ikIndex] * ovps.vs_46[jkIndex] * ovps.os_25[ikIndex] * ovps.os_46[jkIndex] * ovps.ps_13[ijbIndex];
  //  en3[0] = en3[0] + 4.00 * ovps.vs_25[ikIndex] * ovps.vs_13[ijIndex] * ovps.vs_46[jkIndex] * ovps.os_25[ikIndex] * ovps.os_46[jkIndex] * ovps.ps_13[ijbIndex];

  //34/56
  //  en3[1] = en3[1] + 1.00 * ovps.vs_15[ikIndex] * ovps.vs_26[ikIndex] * ovps.os_13[ijIndex] * ovps.os_24[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_35[jkbIndex];
  //  en3[1] = en3[1] - 0.50 * ovps.vs_16[ikIndex] * ovps.vs_25[ikIndex] * ovps.os_13[ijIndex] * ovps.os_24[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_35[jkbIndex];
  //  en3[1] = en3[1] - 0.50 * ovps.vs_15[ikIndex] * ovps.vs_26[ikIndex] * ovps.os_14[ijIndex] * ovps.os_23[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_35[jkbIndex];
  //  en3[1] = en3[1] + 1.00 * ovps.vs_16[ikIndex] * ovps.vs_25[ikIndex] * ovps.os_14[ijIndex] * ovps.os_23[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_35[jkbIndex];
  //  en3[1] = en3[1] - 2.00 * ovps.vs_14[ijIndex] * ovps.vs_25[ikIndex] * ovps.vs_46[jkIndex] * ovps.os_16[ikIndex] * ovps.os_23[ijIndex] * ovps.ps_35[jkbIndex];
  //  en3[1] = en3[1] + 1.00 * ovps.vs_15[ikIndex] * ovps.vs_24[ijIndex] * ovps.vs_46[jkIndex] * ovps.os_16[ikIndex] * ovps.os_23[ijIndex] * ovps.ps_35[jkbIndex];
  //  en3[1] = en3[1] + 1.00 * ovps.vs_14[ijIndex] * ovps.vs_25[ikIndex] * ovps.vs_36[jkIndex] * ovps.os_16[ikIndex] * ovps.os_24[ijIndex] * ovps.ps_35[jkbIndex];
  //  en3[1] = en3[1] - 2.00 * ovps.vs_15[ikIndex] * ovps.vs_24[ijIndex] * ovps.vs_36[jkIndex] * ovps.os_16[ikIndex] * ovps.os_24[ijIndex] * ovps.ps_35[jkbIndex];
  //  en3[1] = en3[1] + 1.00 * ovps.vs_14[ijIndex] * ovps.vs_26[ikIndex] * ovps.vs_45[jkIndex] * ovps.os_16[ikIndex] * ovps.os_23[ijIndex] * ovps.ps_35[jkbIndex];
  //  en3[1] = en3[1] - 2.00 * ovps.vs_16[ikIndex] * ovps.vs_24[ijIndex] * ovps.vs_45[jkIndex] * ovps.os_16[ikIndex] * ovps.os_23[ijIndex] * ovps.ps_35[jkbIndex];
  //  en3[1] = en3[1] - 2.00 * ovps.vs_14[ijIndex] * ovps.vs_26[ikIndex] * ovps.vs_35[jkIndex] * ovps.os_16[ikIndex] * ovps.os_24[ijIndex] * ovps.ps_35[jkbIndex];
  //  en3[1] = en3[1] + 4.00 * ovps.vs_16[ikIndex] * ovps.vs_24[ijIndex] * ovps.vs_35[jkIndex] * ovps.os_16[ikIndex] * ovps.os_24[ijIndex] * ovps.ps_35[jkbIndex];

  //12/56
  //  en3[2] = en3[2] - 2.00 * ovps.vs_13[ijIndex] * ovps.vs_26[ikIndex] * ovps.vs_35[jkIndex] * ovps.os_24[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_15[ikbIndex];
  //  en3[2] = en3[2] + 1.00 * ovps.vs_16[ikIndex] * ovps.vs_23[ijIndex] * ovps.vs_35[jkIndex] * ovps.os_24[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_15[ikbIndex];
  //  en3[2] = en3[2] + 1.00 * ovps.vs_13[ijIndex] * ovps.vs_26[ikIndex] * ovps.vs_45[jkIndex] * ovps.os_23[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_15[ikbIndex];
  //  en3[2] = en3[2] - 2.00 * ovps.vs_16[ikIndex] * ovps.vs_23[ijIndex] * ovps.vs_45[jkIndex] * ovps.os_23[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_15[ikbIndex];
  //  en3[2] = en3[2] + 1.00 * ovps.vs_13[ijIndex] * ovps.vs_25[ikIndex] * ovps.vs_36[jkIndex] * ovps.os_24[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_15[ikbIndex];
  //  en3[2] = en3[2] - 2.00 * ovps.vs_15[ikIndex] * ovps.vs_23[ijIndex] * ovps.vs_36[jkIndex] * ovps.os_24[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_15[ikbIndex];
  //  en3[2] = en3[2] - 2.00 * ovps.vs_13[ijIndex] * ovps.vs_25[ikIndex] * ovps.vs_46[jkIndex] * ovps.os_23[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_15[ikbIndex];
  //  en3[2] = en3[2] + 4.00 * ovps.vs_15[ikIndex] * ovps.vs_23[ijIndex] * ovps.vs_46[jkIndex] * ovps.os_23[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_15[ikbIndex];
  //  en3[2] = en3[2] + 2.00 * ovps.vs_13[ijIndex] * ovps.vs_24[ijIndex] * ovps.vs_35[jkIndex] * ovps.vs_46[jkIndex] * ovps.os_26[ikIndex] * ovps.ps_15[ikbIndex];
  //  en3[2] = en3[2] - 1.00 * ovps.vs_13[ijIndex] * ovps.vs_24[ijIndex] * ovps.vs_36[jkIndex] * ovps.vs_45[jkIndex] * ovps.os_26[ikIndex] * ovps.ps_15[ikbIndex];

  //12/34c
  //  en3[3] = en3[3] + 2.00 * ovps.vs_25[ikIndex] * ovps.vs_36[jkIndex] * ovps.os_24[ijIndex] * ovps.os_16[ikIndex] * ovps.os_45[jkIndex] * ovps.ps_13c[ijbIndex];
  //  en3[3] = en3[3] - 1.00 * ovps.vs_25[ikIndex] * ovps.vs_36[jkIndex] * ovps.os_26[ikIndex] * ovps.os_14[ijIndex] * ovps.os_45[jkIndex] * ovps.ps_13c[ijbIndex];
  //  en3[3] = en3[3] - 1.00 * ovps.vs_25[ikIndex] * ovps.vs_46[jkIndex] * ovps.os_23[ijIndex] * ovps.os_16[ikIndex] * ovps.os_45[jkIndex] * ovps.ps_13c[ijbIndex];
  //  en3[3] = en3[3] + 2.00 * ovps.vs_25[ikIndex] * ovps.vs_46[jkIndex] * ovps.os_26[ikIndex] * ovps.os_13[ijIndex] * ovps.os_45[jkIndex] * ovps.ps_13c[ijbIndex];
  //  en3[3] = en3[3] - 1.00 * ovps.vs_25[ikIndex] * ovps.vs_36[jkIndex] * ovps.os_24[ijIndex] * ovps.os_15[ikIndex] * ovps.os_46[jkIndex] * ovps.ps_13c[ijbIndex];
  //  en3[3] = en3[3] + 2.00 * ovps.vs_25[ikIndex] * ovps.vs_36[jkIndex] * ovps.os_25[ikIndex] * ovps.os_14[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_13c[ijbIndex];
  //  en3[3] = en3[3] + 2.00 * ovps.vs_25[ikIndex] * ovps.vs_46[jkIndex] * ovps.os_23[ijIndex] * ovps.os_15[ikIndex] * ovps.os_46[jkIndex] * ovps.ps_13c[ijbIndex];
  //  en3[3] = en3[3] - 4.00 * ovps.vs_25[ikIndex] * ovps.vs_46[jkIndex] * ovps.os_25[ikIndex] * ovps.os_13[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_13c[ijbIndex];
  //  en3[3] = en3[3] - 2.00 * ovps.vs_24[ijIndex] * ovps.vs_45[jkIndex] * ovps.vs_36[jkIndex] * ovps.os_25[ikIndex] * ovps.os_16[ikIndex] * ovps.ps_13c[ijbIndex];
  //  en3[3] = en3[3] + 1.00 * ovps.vs_24[ijIndex] * ovps.vs_45[jkIndex] * ovps.vs_36[jkIndex] * ovps.os_26[ikIndex] * ovps.os_15[ikIndex] * ovps.ps_13c[ijbIndex];

  //34/56c
  //  en3[4] = en3[4] + 2.00 * ovps.vs_13[ijIndex] * ovps.vs_26[ikIndex] * ovps.os_15[ikIndex] * ovps.os_24[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_35c[jkbIndex];
  //  en3[4] = en3[4] - 1.00 * ovps.vs_16[ikIndex] * ovps.vs_23[ijIndex] * ovps.os_15[ikIndex] * ovps.os_24[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_35c[jkbIndex];
  //  en3[4] = en3[4] - 1.00 * ovps.vs_14[ijIndex] * ovps.vs_26[ikIndex] * ovps.os_15[ikIndex] * ovps.os_24[ijIndex] * ovps.os_36[jkIndex] * ovps.ps_35c[jkbIndex];
  //  en3[4] = en3[4] + 2.00 * ovps.vs_16[ikIndex] * ovps.vs_24[ijIndex] * ovps.os_15[ikIndex] * ovps.os_24[ijIndex] * ovps.os_36[jkIndex] * ovps.ps_35c[jkbIndex];
  //  en3[4] = en3[4] - 1.00 * ovps.vs_13[ijIndex] * ovps.vs_26[ikIndex] * ovps.os_16[ikIndex] * ovps.os_24[ijIndex] * ovps.os_45[jkIndex] * ovps.ps_35c[jkbIndex];
  //  en3[4] = en3[4] + 2.00 * ovps.vs_16[ikIndex] * ovps.vs_23[ijIndex] * ovps.os_16[ikIndex] * ovps.os_24[ijIndex] * ovps.os_45[jkIndex] * ovps.ps_35c[jkbIndex];
  //  en3[4] = en3[4] + 2.00 * ovps.vs_14[ijIndex] * ovps.vs_26[ikIndex] * ovps.os_16[ikIndex] * ovps.os_24[ijIndex] * ovps.os_35[jkIndex] * ovps.ps_35c[jkbIndex];
  //  en3[4] = en3[4] - 4.00 * ovps.vs_16[ikIndex] * ovps.vs_24[ijIndex] * ovps.os_16[ikIndex] * ovps.os_24[ijIndex] * ovps.os_35[jkIndex] * ovps.ps_35c[jkbIndex];
  //  en3[4] = en3[4] - 1.00 * ovps.vs_13[ijIndex] * ovps.vs_24[ijIndex] * ovps.vs_46[jkIndex] * ovps.os_15[ikIndex] * ovps.os_26[ikIndex] * ovps.ps_35c[jkbIndex];
  //  en3[4] = en3[4] + 0.50 * ovps.vs_14[ijIndex] * ovps.vs_23[ijIndex] * ovps.vs_46[jkIndex] * ovps.os_15[ikIndex] * ovps.os_26[ikIndex] * ovps.ps_35c[jkbIndex];
  //  en3[4] = en3[4] + 0.50 * ovps.vs_13[ijIndex] * ovps.vs_24[ijIndex] * ovps.vs_46[jkIndex] * ovps.os_16[ikIndex] * ovps.os_25[ikIndex] * ovps.ps_35c[jkbIndex];
  //  en3[4] = en3[4] - 1.00 * ovps.vs_14[ijIndex] * ovps.vs_23[ijIndex] * ovps.vs_46[jkIndex] * ovps.os_16[ikIndex] * ovps.os_25[ikIndex] * ovps.ps_35c[jkbIndex];

  //12/56c
  //  en3[5] = en3[5] - 2.00 * ovps.vs_26[ikIndex] * ovps.os_23[ijIndex] * ovps.os_14[ijIndex] * ovps.os_36[jkIndex] * ovps.os_45[jkIndex] * ovps.ps_15c[ikbIndex];
  //  en3[5] = en3[5] + 1.00 * ovps.vs_26[ikIndex] * ovps.os_23[ijIndex] * ovps.os_14[ijIndex] * ovps.os_35[jkIndex] * ovps.os_46[jkIndex] * ovps.ps_15c[ikbIndex];
  //  en3[5] = en3[5] + 2.00 * ovps.vs_23[ijIndex] * ovps.vs_36[jkIndex] * ovps.os_26[ikIndex] * ovps.os_14[ijIndex] * ovps.os_45[jkIndex] * ovps.ps_15c[ikbIndex];
  //  en3[5] = en3[5] - 1.00 * ovps.vs_23[ijIndex] * ovps.vs_36[jkIndex] * ovps.os_24[ijIndex] * ovps.os_16[ikIndex] * ovps.os_45[jkIndex] * ovps.ps_15c[ikbIndex];
  //  en3[5] = en3[5] - 1.00 * ovps.vs_23[ijIndex] * ovps.vs_46[jkIndex] * ovps.os_26[ikIndex] * ovps.os_13[ijIndex] * ovps.os_45[jkIndex] * ovps.ps_15c[ikbIndex];
  //  en3[5] = en3[5] + 2.00 * ovps.vs_23[ijIndex] * ovps.vs_46[jkIndex] * ovps.os_23[ijIndex] * ovps.os_16[ikIndex] * ovps.os_45[jkIndex] * ovps.ps_15c[ikbIndex];
  //  en3[5] = en3[5] - 1.00 * ovps.vs_23[ijIndex] * ovps.vs_36[jkIndex] * ovps.os_25[ikIndex] * ovps.os_14[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_15c[ikbIndex];
  //  en3[5] = en3[5] + 2.00 * ovps.vs_23[ijIndex] * ovps.vs_36[jkIndex] * ovps.os_24[ijIndex] * ovps.os_15[ikIndex] * ovps.os_46[jkIndex] * ovps.ps_15c[ikbIndex];
  //  en3[5] = en3[5] + 2.00 * ovps.vs_23[ijIndex] * ovps.vs_46[jkIndex] * ovps.os_25[ikIndex] * ovps.os_13[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_15c[ikbIndex];
  //  en3[5] = en3[5] - 4.00 * ovps.vs_23[ijIndex] * ovps.vs_46[jkIndex] * ovps.os_23[ijIndex] * ovps.os_15[ikIndex] * ovps.os_46[jkIndex] * ovps.ps_15c[ikbIndex];

  //constant
  //  en3[6] = en3[6] - 4.00 * ovps.vs_15[ikIndex] * ovps.vs_36[jkIndex] * ovps.os_14[ijIndex] * ovps.os_36[jkIndex] * ovps.os_45[jkIndex] * ovps.ps_22c[ibIndex];
  //  en3[6] = en3[6] + 2.00 * ovps.vs_15[ikIndex] * ovps.vs_36[jkIndex] * ovps.os_24[ijIndex] * ovps.os_36[jkIndex] * ovps.os_45[jkIndex] * ovps.ps_12c[ibIndex];
  //  en3[6] = en3[6] + 2.00 * ovps.vs_15[ikIndex] * ovps.vs_36[jkIndex] * ovps.os_14[ijIndex] * ovps.os_35[jkIndex] * ovps.os_46[jkIndex] * ovps.ps_22c[ibIndex];
  //  en3[6] = en3[6] - 1.00 * ovps.vs_15[ikIndex] * ovps.vs_36[jkIndex] * ovps.os_24[ijIndex] * ovps.os_35[jkIndex] * ovps.os_46[jkIndex] * ovps.ps_12c[ibIndex];
  //  en3[6] = en3[6] + 4.00 * ovps.vs_13[ijIndex] * ovps.vs_35[jkIndex] * ovps.vs_46[jkIndex] * ovps.os_15[ikIndex] * ovps.os_46[jkIndex] * ovps.ps_22c[ibIndex];
  //  en3[6] = en3[6] - 2.00 * ovps.vs_13[ijIndex] * ovps.vs_35[jkIndex] * ovps.vs_46[jkIndex] * ovps.os_25[ikIndex] * ovps.os_46[jkIndex] * ovps.ps_12c[ibIndex];
  //  en3[6] = en3[6] - 2.00 * ovps.vs_13[ijIndex] * ovps.vs_35[jkIndex] * ovps.vs_46[jkIndex] * ovps.os_16[ikIndex] * ovps.os_45[jkIndex] * ovps.ps_22c[ibIndex];
  //  en3[6] = en3[6] + 1.00 * ovps.vs_13[ijIndex] * ovps.vs_35[jkIndex] * ovps.vs_46[jkIndex] * ovps.os_26[ikIndex] * ovps.os_45[jkIndex] * ovps.ps_12c[ibIndex];
  //  en3[6] = en3[6] - 4.00 * ovps.vs_15[ikIndex] * ovps.vs_26[ikIndex] * ovps.os_13[ijIndex] * ovps.os_26[ikIndex] * ovps.os_35[jkIndex] * ovps.ps_44c[jbIndex];
  //  en3[6] = en3[6] + 2.00 * ovps.vs_15[ikIndex] * ovps.vs_26[ikIndex] * ovps.os_13[ijIndex] * ovps.os_25[ikIndex] * ovps.os_36[jkIndex] * ovps.ps_44c[jbIndex];
  //  en3[6] = en3[6] + 2.00 * ovps.vs_15[ikIndex] * ovps.vs_26[ikIndex] * ovps.os_13[ijIndex] * ovps.os_26[ikIndex] * ovps.os_45[jkIndex] * ovps.ps_34c[jbIndex];
  //  en3[6] = en3[6] - 1.00 * ovps.vs_15[ikIndex] * ovps.vs_26[ikIndex] * ovps.os_13[ijIndex] * ovps.os_25[ikIndex] * ovps.os_46[jkIndex] * ovps.ps_34c[jbIndex];
  //  en3[6] = en3[6] + 2.00 * ovps.vs_13[ijIndex] * ovps.vs_25[ikIndex] * ovps.vs_36[jkIndex] * ovps.os_16[ikIndex] * ovps.os_25[ikIndex] * ovps.ps_44c[jbIndex];
  //  en3[6] = en3[6] - 1.00 * ovps.vs_15[ikIndex] * ovps.vs_23[ijIndex] * ovps.vs_36[jkIndex] * ovps.os_16[ikIndex] * ovps.os_25[ikIndex] * ovps.ps_44c[jbIndex];
  //  en3[6] = en3[6] - 1.00 * ovps.vs_13[ijIndex] * ovps.vs_25[ikIndex] * ovps.vs_46[jkIndex] * ovps.os_16[ikIndex] * ovps.os_25[ikIndex] * ovps.ps_34c[jbIndex];
  //  en3[6] = en3[6] + 0.50 * ovps.vs_15[ikIndex] * ovps.vs_23[ijIndex] * ovps.vs_46[jkIndex] * ovps.os_16[ikIndex] * ovps.os_25[ikIndex] * ovps.ps_34c[jbIndex];
  //  en3[6] = en3[6] - 1.00 * ovps.vs_13[ijIndex] * ovps.vs_25[ikIndex] * ovps.vs_36[jkIndex] * ovps.os_15[ikIndex] * ovps.os_26[ikIndex] * ovps.ps_44c[jbIndex];
  //  en3[6] = en3[6] + 2.00 * ovps.vs_15[ikIndex] * ovps.vs_23[ijIndex] * ovps.vs_36[jkIndex] * ovps.os_15[ikIndex] * ovps.os_26[ikIndex] * ovps.ps_44c[jbIndex];
  //  en3[6] = en3[6] + 0.50 * ovps.vs_13[ijIndex] * ovps.vs_25[ikIndex] * ovps.vs_46[jkIndex] * ovps.os_15[ikIndex] * ovps.os_26[ikIndex] * ovps.ps_34c[jbIndex];
  //  en3[6] = en3[6] - 1.00 * ovps.vs_15[ikIndex] * ovps.vs_23[ijIndex] * ovps.vs_46[jkIndex] * ovps.os_15[ikIndex] * ovps.os_26[ikIndex] * ovps.ps_34c[jbIndex];
  //  en3[6] = en3[6] - 2.00 * ovps.vs_15[ikIndex] * ovps.vs_23[ijIndex] * ovps.os_14[ijIndex] * ovps.os_23[ijIndex] * ovps.os_45[jkIndex] * ovps.ps_66c[kbIndex];
  //  en3[6] = en3[6] + 1.00 * ovps.vs_13[ijIndex] * ovps.vs_25[ikIndex] * ovps.os_14[ijIndex] * ovps.os_23[ijIndex] * ovps.os_45[jkIndex] * ovps.ps_66c[kbIndex];
  //  en3[6] = en3[6] + 1.00 * ovps.vs_15[ikIndex] * ovps.vs_23[ijIndex] * ovps.os_13[ijIndex] * ovps.os_24[ijIndex] * ovps.os_45[jkIndex] * ovps.ps_66c[kbIndex];
  //  en3[6] = en3[6] - 2.00 * ovps.vs_13[ijIndex] * ovps.vs_25[ikIndex] * ovps.os_13[ijIndex] * ovps.os_24[ijIndex] * ovps.os_45[jkIndex] * ovps.ps_66c[kbIndex];
  //  en3[6] = en3[6] + 1.00 * ovps.vs_15[ikIndex] * ovps.vs_23[ijIndex] * ovps.os_14[ijIndex] * ovps.os_23[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_56c[kbIndex];
  //  en3[6] = en3[6] - 0.50 * ovps.vs_13[ijIndex] * ovps.vs_25[ikIndex] * ovps.os_14[ijIndex] * ovps.os_23[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_56c[kbIndex];
  //  en3[6] = en3[6] - 0.50 * ovps.vs_15[ikIndex] * ovps.vs_23[ijIndex] * ovps.os_13[ijIndex] * ovps.os_24[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_56c[kbIndex];
  //  en3[6] = en3[6] + 1.00 * ovps.vs_13[ijIndex] * ovps.vs_25[ikIndex] * ovps.os_13[ijIndex] * ovps.os_24[ijIndex] * ovps.os_46[jkIndex] * ovps.ps_56c[kbIndex];
  //  en3[6] = en3[6] + 4.00 * ovps.vs_13[ijIndex] * ovps.vs_24[ijIndex] * ovps.vs_35[jkIndex] * ovps.os_15[ikIndex] * ovps.os_24[ijIndex] * ovps.ps_66c[kbIndex];
  //  en3[6] = en3[6] - 2.00 * ovps.vs_13[ijIndex] * ovps.vs_24[ijIndex] * ovps.vs_45[jkIndex] * ovps.os_15[ikIndex] * ovps.os_23[ijIndex] * ovps.ps_66c[kbIndex];
  //  en3[6] = en3[6] - 2.00 * ovps.vs_13[ijIndex] * ovps.vs_24[ijIndex] * ovps.vs_35[jkIndex] * ovps.os_16[ikIndex] * ovps.os_24[ijIndex] * ovps.ps_56c[kbIndex];
  //  en3[6] = en3[6] + 1.00 * ovps.vs_13[ijIndex] * ovps.vs_24[ijIndex] * ovps.vs_45[jkIndex] * ovps.os_16[ikIndex] * ovps.os_23[ijIndex] * ovps.ps_56c[kbIndex];
}
 */
