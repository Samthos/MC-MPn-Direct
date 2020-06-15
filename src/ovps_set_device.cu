
#include "ovps_set.h"
#include "cublas_v2.h"

#include "ovps_set.cpp"

//__global__ void printit(double* v, int lda) {
//  for (int i = 0; i < lda; i++) {
//    printf("DEBUGING: psi %2i %12.4f %12.4f\n", i, v[i], v[i + lda]);
//  }
//  printf("DEBUGING\n");
//}

template <>
void OVPS_SET_BASE<thrust::device_vector<double>>::update(double *h_psi1Tau, double *h_psi2Tau, size_t inner, size_t lda) {
  double alpha = 1.0;
  double beta = 0.0;

  cublasHandle_t handle;
  cublasCreate(&handle);

  double *psi1Tau = nullptr;
  double *psi2Tau = nullptr;
  cudaMalloc((void**) &psi1Tau, sizeof(double) * (mc_pair_num)*lda);
  cudaMalloc((void**) &psi2Tau, sizeof(double) * (mc_pair_num)*lda);
  cudaMemset(psi1Tau, 0, sizeof(double) * mc_pair_num * lda);
  cudaMemset(psi2Tau, 0, sizeof(double) * mc_pair_num * lda);
  cudaMemcpy(psi1Tau, h_psi1Tau, sizeof(double) * (inner + (mc_pair_num - 1) * lda), cudaMemcpyHostToDevice);
  cudaMemcpy(psi2Tau, h_psi2Tau, sizeof(double) * (inner + (mc_pair_num - 1) * lda), cudaMemcpyHostToDevice);


// printit<<<1,1>>>(psi1Tau, lda);
// cudaThreadSynchronize();
// printf("DEBUGING: inner = %i; lda = %i\n", inner, lda);

  // fill s_12 so upper triangle is zero
  thrust::fill(s_12.begin(), s_12.end(), 0.0);

  // use s_12 as temp storage to produce s_11
  cublasDsyrk(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
      mc_pair_num, inner, 
      &alpha,
      psi1Tau, lda,
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
      psi2Tau, lda,
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
      psi1Tau, lda,
      psi2Tau, lda,
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
  cudaFree(psi1Tau);
  cudaFree(psi2Tau);
}

void copy_OVPS_HOST_TO_DEVICE(OVPS_SET& src, OVPS_SET_DEVICE& dest) {
  thrust::copy(src.s_11.begin(), src.s_11.end(), dest.s_11.begin());
  thrust::copy(src.s_12.begin(), src.s_12.end(), dest.s_12.begin());
  thrust::copy(src.s_21.begin(), src.s_21.end(), dest.s_21.begin());
  thrust::copy(src.s_22.begin(), src.s_22.end(), dest.s_22.begin());
}

void copy_OVPS_DEVICE_TO_HOST(OVPS_SET_DEVICE& src, OVPS_SET& dest) {
  thrust::copy(src.s_11.begin(), src.s_11.end(), dest.s_11.begin());
  thrust::copy(src.s_12.begin(), src.s_12.end(), dest.s_12.begin());
  thrust::copy(src.s_21.begin(), src.s_21.end(), dest.s_21.begin());
  thrust::copy(src.s_22.begin(), src.s_22.end(), dest.s_22.begin());
}
