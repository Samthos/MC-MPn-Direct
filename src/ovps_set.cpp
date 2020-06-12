#include "ovps_set.h"
#include "cblas.h"
#include "blas_calls.h"

OVPS_SET::OVPS_SET(int mc_pair_num_) {
  resize(mc_pair_num_);
}
void OVPS_SET::resize(int mc_pair_num_) {
  mc_pair_num = mc_pair_num_;
  s_11.resize(mc_pair_num * mc_pair_num);
  s_12.resize(mc_pair_num * mc_pair_num);
  s_21.resize(mc_pair_num * mc_pair_num);
  s_22.resize(mc_pair_num * mc_pair_num);
}
void OVPS_SET::update(double *psi1Tau, double *psi2Tau, size_t inner, size_t lda) {
  double alpha = 1.0;
  double beta = 0.0;

  cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
      mc_pair_num, inner,
      alpha,
      psi1Tau, lda,
      beta,
      s_11.data(), mc_pair_num);
  set_Upper_from_Lower(s_11.data(), mc_pair_num);
  cblas_dscal(mc_pair_num, 0.0, s_11.data(), mc_pair_num+1);

  cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
      mc_pair_num, inner,
      alpha,
      psi2Tau, lda,
      beta,
      s_22.data(), mc_pair_num);
  set_Upper_from_Lower(s_22.data(), mc_pair_num);
  cblas_dscal(mc_pair_num, 0.0, s_22.data(), mc_pair_num+1);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      mc_pair_num, mc_pair_num, inner,
      alpha,
      psi1Tau, lda,
      psi2Tau, lda,
      beta,
      s_21.data(), mc_pair_num);
  cblas_dscal(mc_pair_num, 0.0, s_21.data(), mc_pair_num+1);
  Transpose(s_21.data(), mc_pair_num, s_12.data());
}
