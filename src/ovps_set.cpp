#include <iostream>

#include "ovps_set.h"
#include "cblas.h"
#include "blas_calls.h"

template <template <class, class> class Container, template <class> class Allocator>
OVPS_Set<Container, Allocator>::OVPS_Set(int mc_pair_num_) {
  resize(mc_pair_num_);
}

template <template <class, class> class Container, template <class> class Allocator>
void OVPS_Set<Container, Allocator>::resize(int mc_pair_num_) {
  mc_pair_num = mc_pair_num_;
  s_11.resize(mc_pair_num * mc_pair_num);
  s_12.resize(mc_pair_num * mc_pair_num);
  s_21.resize(mc_pair_num * mc_pair_num);
  s_22.resize(mc_pair_num * mc_pair_num);
}

template <template <class, class> class Container, template <class> class Allocator>
void OVPS_Set<Container, Allocator>::update(vector_double& psi1Tau, int psi1_offset, vector_double& psi2Tau, int psi2_offset, size_t inner, size_t lda) {
}

template <>
void OVPS_Set_Host::update(vector_double& psi1Tau, int psi1_offset, vector_double& psi2Tau, int psi2_offset, size_t inner, size_t lda) {
  double alpha = 1.0;
  double beta = 0.0;

  cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
      mc_pair_num, inner,
      alpha,
      psi1Tau.data() + psi1_offset, lda,
      beta,
      s_11.data(), mc_pair_num);
  set_Upper_from_Lower(s_11.data(), mc_pair_num);
  cblas_dscal(mc_pair_num, 0.0, s_11.data(), mc_pair_num+1);

  cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans,
      mc_pair_num, inner,
      alpha,
      psi2Tau.data() + psi2_offset, lda,
      beta,
      s_22.data(), mc_pair_num);
  set_Upper_from_Lower(s_22.data(), mc_pair_num);
  cblas_dscal(mc_pair_num, 0.0, s_22.data(), mc_pair_num+1);

  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
      mc_pair_num, mc_pair_num, inner,
      alpha,
      psi1Tau.data() + psi1_offset, lda,
      psi2Tau.data() + psi2_offset, lda,
      beta,
      s_21.data(), mc_pair_num);
  cblas_dscal(mc_pair_num, 0.0, s_21.data(), mc_pair_num+1);
  Transpose(s_21.data(), mc_pair_num, s_12.data());
}
