#ifdef HAVE_CUDA
#include "cublas_v2.h"
#endif

#include "ovps_set.h"
#include "cblas.h"
#include "blas_calls.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
OVPS_Set<Container, Allocator>::OVPS_Set(int mc_pair_num_) {
  resize(mc_pair_num_);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void OVPS_Set<Container, Allocator>::resize(int mc_pair_num_) {
  mc_pair_num = mc_pair_num_;
  s_11.resize(mc_pair_num * mc_pair_num);
  s_12.resize(mc_pair_num * mc_pair_num);
  s_21.resize(mc_pair_num * mc_pair_num);
  s_22.resize(mc_pair_num * mc_pair_num);
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void OVPS_Set<Container, Allocator>::update(vector_double& psi1Tau, int psi1_offset, vector_double& psi2Tau, int psi2_offset, size_t inner, size_t lda, Blas_Wrapper_Type& blas_wrapper) {
  double alpha = 1.0;
  double beta = 0.0;

  blas_wrapper.dsyrk(BLAS_WRAPPER::FILL_FULL, true,
      mc_pair_num, inner,
      alpha,
      psi1Tau, psi1_offset, lda,
      beta,
      s_11, 0, mc_pair_num);
  blas_wrapper.dscal(mc_pair_num, 0.0, s_11, mc_pair_num+1);

  blas_wrapper.dsyrk(BLAS_WRAPPER::FILL_FULL, true,
      mc_pair_num, inner,
      alpha,
      psi2Tau, psi2_offset, lda,
      beta,
      s_22, 0, mc_pair_num);
  blas_wrapper.dscal(mc_pair_num, 0.0, s_22, mc_pair_num+1);

  blas_wrapper.dgemm(true, false,
      mc_pair_num, mc_pair_num, inner,
      alpha,
      psi1Tau, psi1_offset, lda,
      psi2Tau, psi2_offset, lda,
      beta,
      s_21, 0, mc_pair_num);
  blas_wrapper.dscal(mc_pair_num, 0.0, s_21, mc_pair_num+1);
  blas_wrapper.transpose(mc_pair_num, mc_pair_num, s_21, mc_pair_num, s_12, mc_pair_num);
}

#ifdef HAVE_CUDA
void copy_OVPS_Set(OVPS_Set_Host& src, OVPS_Set_Device& dest) {
  thrust::copy(src.s_11.begin(), src.s_11.end(), dest.s_11.begin());
  thrust::copy(src.s_12.begin(), src.s_12.end(), dest.s_12.begin());
  thrust::copy(src.s_21.begin(), src.s_21.end(), dest.s_21.begin());
  thrust::copy(src.s_22.begin(), src.s_22.end(), dest.s_22.begin());
}

void copy_OVPS_Set(OVPS_Set_Device& src, OVPS_Set_Host& dest) {
  thrust::copy(src.s_11.begin(), src.s_11.end(), dest.s_11.begin());
  thrust::copy(src.s_12.begin(), src.s_12.end(), dest.s_12.begin());
  thrust::copy(src.s_21.begin(), src.s_21.end(), dest.s_21.begin());
  thrust::copy(src.s_22.begin(), src.s_22.end(), dest.s_22.begin());
}
#endif
