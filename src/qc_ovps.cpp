#include <iostream>
#include <functional>
#include <algorithm>
#include <cmath>

#ifdef HAVE_CUDA 
#include "cublas_v2.h"
#endif

#include "cblas.h"
#include "blas_calls.h"
#include "qc_ovps.h"

template <template <typename, typename> typename Container, template <typename> typename Allocator>
OVPS<Container, Allocator>::OVPS() {
  electron_pairs = 0;
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
OVPS<Container, Allocator>::~OVPS() {
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void OVPS<Container, Allocator>::init(const int dimm, const int electron_pairs_) {
  electron_pairs = electron_pairs_;

  o_set.resize(dimm);
  v_set.resize(dimm);
  for (auto stop = 0; stop < dimm; stop++) {
    o_set[stop].resize(stop + 1);
    v_set[stop].resize(stop + 1);
    for (auto start = 0; start < stop + 1; start++) {
      o_set[stop][start].resize(electron_pairs);
      v_set[stop][start].resize(electron_pairs);
    }
  }
}

template <template <typename, typename> typename Container, template <typename> typename Allocator>
void OVPS<Container, Allocator>::update(Wavefunction_Type& electron_pair_psi1, Wavefunction_Type& electron_pair_psi2, Tau* tau) {
  std::cerr << "Default OVPS update_ovsp not implemented\n";
  exit(0);
}

template <>
void OVPS_Host::update(Wavefunction_Type& electron_pair_psi1, Wavefunction_Type& electron_pair_psi2, Tau* tau) {
  auto iocc1 = electron_pair_psi1.iocc1;
  auto iocc2 = electron_pair_psi1.iocc2;
  auto ivir1 = electron_pair_psi1.ivir1;
  auto ivir2 = electron_pair_psi1.ivir2;
  auto lda =  electron_pair_psi1.lda;
  for (auto stop = 0; stop < o_set.size(); stop++) {
    for (auto start = 0; start < o_set[stop].size(); start++) {
      auto t_val = tau->get_exp_tau(stop, start);
      std::transform(t_val.begin(), t_val.end(), t_val.begin(), [](double x){return sqrt(x);});

      blas_wrapper.ddgmm(BLAS_WRAPPER::LEFT_SIDE,
          ivir2 - iocc1, electron_pairs,
          electron_pair_psi1.psi, iocc1, lda,
          t_val, iocc1, 1,
          electron_pair_psi1.psiTau, iocc1, lda);
      blas_wrapper.ddgmm(BLAS_WRAPPER::LEFT_SIDE,
          ivir2 - iocc1, electron_pairs, 
          electron_pair_psi2.psi, iocc1, lda, 
          t_val, iocc1, 1,
          electron_pair_psi2.psiTau, iocc1, lda);

      o_set[stop][start].update(electron_pair_psi1.psiTau, iocc1, electron_pair_psi2.psiTau, iocc1, iocc2 - iocc1, lda, blas_wrapper);
      v_set[stop][start].update(electron_pair_psi1.psiTau, ivir1, electron_pair_psi2.psiTau, ivir1, ivir2 - ivir1, lda, blas_wrapper);
    }
  }
}

#ifdef HAVE_CUDA 
template <>
void OVPS_Device::update(Wavefunction_Type& electron_pair_psi1, Wavefunction_Type& electron_pair_psi2, Tau* tau) {
  auto iocc1 = electron_pair_psi1.iocc1;
  auto iocc2 = electron_pair_psi1.iocc2;
  auto ivir1 = electron_pair_psi1.ivir1;
  auto ivir2 = electron_pair_psi1.ivir2;
  auto lda   = electron_pair_psi1.lda;

  for (auto stop = 0; stop < o_set.size(); stop++) {
    for (auto start = 0; start < o_set[stop].size(); start++) {
      auto t_val = tau->get_exp_tau(stop, start);
      std::transform(t_val.begin(), t_val.end(), t_val.begin(), [](double x){return sqrt(x);});
      thrust::device_vector<double> d_t_val = t_val;

      blas_wrapper.ddgmm(BLAS_WRAPPER::LEFT_SIDE,
          ivir2 - iocc1, electron_pairs,
          electron_pair_psi1.psi, iocc1, lda,
          d_t_val, iocc1, 1,
          electron_pair_psi1.psiTau, iocc1, lda);
      blas_wrapper.ddgmm(BLAS_WRAPPER::LEFT_SIDE,
          ivir2 - iocc1, electron_pairs, 
          electron_pair_psi2.psi, iocc1, lda, 
          d_t_val, iocc1, 1,
          electron_pair_psi2.psiTau, iocc1, lda);

      o_set[stop][start].update(electron_pair_psi1.psiTau, iocc1, electron_pair_psi2.psiTau, iocc1, iocc2 - iocc1, lda, blas_wrapper);
      v_set[stop][start].update(electron_pair_psi1.psiTau, ivir1, electron_pair_psi2.psiTau, ivir1, ivir2 - ivir1, lda, blas_wrapper);
    }
  }
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
#endif
