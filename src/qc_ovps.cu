#include "cublas_v2.h"
#include "qc_ovps.cpp"

template <>
void OVPS_Device::update(Wavefunction_Type& electron_pair_psi1, Wavefunction_Type& electron_pair_psi2, Tau* tau) {
  // update green's function trace objects

  auto iocc1 = electron_pair_psi1.iocc1;
  auto iocc2 = electron_pair_psi1.iocc2;
  auto ivir1 = electron_pair_psi1.ivir1;
  auto ivir2 = electron_pair_psi1.ivir2;
  auto lda   = electron_pair_psi1.lda;

  cublasHandle_t handle;
  cublasCreate(&handle);

  for (auto stop = 0; stop < o_set.size(); stop++) {
    for (auto start = 0; start < o_set[stop].size(); start++) {
      auto t_val = tau->get_exp_tau(stop, start);
      std::transform(t_val.begin(), t_val.end(), t_val.begin(), [](double x){return sqrt(x);});
      thrust::device_vector<double> d_t_val = t_val;

      cublasDdgmm(handle, CUBLAS_SIDE_LEFT,
          ivir2 - iocc1, electron_pairs,
          electron_pair_psi1.data() + iocc1, lda,
          d_t_val.data().get() + iocc1, 1,
          electron_pair_psi1.dataTau() + iocc1, lda);
      cublasDdgmm(handle, CUBLAS_SIDE_LEFT,
          ivir2 - iocc1, electron_pairs, 
          electron_pair_psi2.data() + iocc1, lda, 
          d_t_val.data().get() + iocc1, 1,
          electron_pair_psi2.dataTau() + iocc1, lda);

      o_set[stop][start].update(electron_pair_psi1.psiTau, iocc1, electron_pair_psi2.psiTau, iocc1, iocc2 - iocc1, lda);
      v_set[stop][start].update(electron_pair_psi1.psiTau, ivir1, electron_pair_psi2.psiTau, ivir1, ivir2 - ivir1, lda);
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
