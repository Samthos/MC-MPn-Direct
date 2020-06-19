#include "cublas_v2.h"
#include "ovps_set.cpp"

template <>
void OVPS_Set<thrust::device_vector<double>>::update(thrust::device_vector<double>& psi1Tau, int psi1_offset, thrust::device_vector<double>& psi2Tau, int psi2_offset, size_t inner, size_t lda) {
  double alpha = 1.0;
  double beta = 0.0;

  cublasHandle_t handle;
  cublasCreate(&handle);

  // fill s_12 so upper triangle is zero
  thrust::fill(s_12.begin(), s_12.end(), 0.0);

  // use s_12 as temp storage to produce s_11
  cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
      mc_pair_num, inner, 
      &alpha,
      psi1Tau.data().get() + psi1_offset, lda,
      &beta,
      s_12.data().get(), mc_pair_num);

  // symmetric add s_12 into s_11
  cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      mc_pair_num, mc_pair_num,
      &alpha,
      s_12.data().get(), mc_pair_num,
      &alpha,
      s_12.data().get(), mc_pair_num,
      s_11.data().get(), mc_pair_num);

  // zero out diagonal
  cublasDscal(handle, mc_pair_num, &beta, s_11.data().get(), mc_pair_num+1);

  // use s_12 as temp storage to produce s_22
  cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
      mc_pair_num, inner,
      &alpha,
      psi2Tau.data().get() + psi2_offset, lda,
      &beta,
      s_12.data().get(), mc_pair_num);

  // symmetric add s_12 into s_22
  cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      mc_pair_num, mc_pair_num,
      &alpha,
      s_12.data().get(), mc_pair_num,
      &alpha,
      s_12.data().get(), mc_pair_num,
      s_22.data().get(), mc_pair_num);

  // zero out diagonal
  cublasDscal(handle, mc_pair_num, &beta, s_22.data().get(), mc_pair_num+1);

  // build s_21
  cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      mc_pair_num, mc_pair_num, inner,
      &alpha,
      psi1Tau.data().get() + psi1_offset, lda,
      psi2Tau.data().get() + psi2_offset, lda,
      &beta,
      s_21.data().get(), mc_pair_num);

  // zero out diagonal
  cublasDscal(handle, mc_pair_num, &beta, s_21.data().get(), mc_pair_num+1);

  // set s_12 as transpose of s_21
  cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      mc_pair_num, mc_pair_num,
      &alpha,
      s_21.data().get(), mc_pair_num,
      &beta,
      s_21.data().get(), mc_pair_num,
      s_12.data().get(), mc_pair_num);

  // destroy handle
  cublasDestroy(handle);
}

