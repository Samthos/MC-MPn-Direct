#include <iostream>

#include "ovps_set.h"
#include "cblas.h"
#include "blas_calls.h"

template <class Container>
OVPS_SET_BASE<Container>::OVPS_SET_BASE(int mc_pair_num_) {
  resize(mc_pair_num_);
}

template <class Container>
void OVPS_SET_BASE<Container>::resize(int mc_pair_num_) {
  mc_pair_num = mc_pair_num_;
  s_11.resize(mc_pair_num * mc_pair_num);
  s_12.resize(mc_pair_num * mc_pair_num);
  s_21.resize(mc_pair_num * mc_pair_num);
  s_22.resize(mc_pair_num * mc_pair_num);
}

template <class Container>
void OVPS_SET_BASE<Container>::update(double *psi1Tau, double *psi2Tau, size_t inner, size_t lda) {
  std::cerr << "Default OVPS_SET_BASE<Container>.update not implemented";
  exit(0);
}

template <>
void OVPS_SET_BASE<std::vector<double>>::update(double *psi1Tau, double *psi2Tau, size_t inner, size_t lda) {
  double alpha = 1.0;
  double beta = 0.0;

// for (int i = 0; i < lda; i++) {
//   printf("DEBUGING: psi %2i %12.4f %12.4f\n", i, psi1Tau[i], psi1Tau[i + lda]);
// }
// printf("DEBUGING\n");
// printf("DEBUGING: inner = %i; lda = %i\n", inner, lda);

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
