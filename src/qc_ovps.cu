#include "cublas_v2.h"
#include "qc_ovps.cpp"

template <>
void OVPS<thrust::device_vector, thrust::device_allocator>::update(Wavefunction<thrust::device_vector, thrust::device_allocator>& electron_pair_psi1, Wavefunction<thrust::device_vector, thrust::device_allocator>& electron_pair_psi2, Tau* tau) {
  // update green's function trace objects

  auto iocc1 = electron_pair_psi1.iocc1;
  auto iocc2 = electron_pair_psi1.iocc2;
  auto ivir1 = electron_pair_psi1.ivir1;
  auto ivir2 = electron_pair_psi1.ivir2;
  auto lda = electron_pair_psi1.lda;

  cublasHandle_t handle;
  cublasCreate(&handle);

  thrust::device_vector<double> psi1 = electron_pair_psi1.psi;
  thrust::device_vector<double> psi2 = electron_pair_psi2.psi;
  thrust::device_vector<double> psi1Tau(psi1.size());
  thrust::device_vector<double> psi2Tau(psi2.size());

  for (auto stop = 0; stop < o_set.size(); stop++) {
    for (auto start = 0; start < o_set[stop].size(); start++) {
      auto t_val = tau->get_exp_tau(stop, start);
      std::transform(t_val.begin(), t_val.end(), t_val.begin(), [](double x){return sqrt(x);});
      thrust::device_vector<double> d_t_val = t_val;

      cublasDdgmm(handle, CUBLAS_SIDE_LEFT,
          ivir2 - iocc1, electron_pairs,
          psi1.data().get() + iocc1, lda,
          d_t_val.data().get() + iocc1, 1,
          psi1Tau.data().get() + iocc1, lda);
      cublasDdgmm(handle, CUBLAS_SIDE_LEFT,
          ivir2 - iocc1, electron_pairs, 
          psi2.data().get() + iocc1, lda, 
          d_t_val.data().get() + iocc1, 1,
          psi2Tau.data().get() + iocc1, lda);

      o_set[stop][start].update(psi1Tau, iocc1, psi2Tau, iocc1, iocc2 - iocc1, lda);
      v_set[stop][start].update(psi1Tau, ivir1, psi2Tau, ivir1, ivir2 - ivir1, lda);
    }
  }
  cublasDestroy(handle);
}

void copy_OVPS(OVPS_Host& src, OVPS_Device& dest) {
  for (int i = 0; i < src.o_set.size(); i++) {
    for (int j = 0; j < src.o_set[i].size(); j++) {
      copy_OVPS_Set(src.o_set[i][j], dest.o_set[i][j]);
      copy_OVPS_Set(src.v_set[i][j], dest.v_set[i][j]);
    }
  }
}

void copy_OVPS(OVPS_Device& src, OVPS_Host& dest) {
  for (int i = 0; i < src.o_set.size(); i++) {
    for (int j = 0; j < src.o_set[i].size(); j++) {
      copy_OVPS_Set(src.o_set[i][j], dest.o_set[i][j]);
      copy_OVPS_Set(src.v_set[i][j], dest.v_set[i][j]);
    }
  }
}
