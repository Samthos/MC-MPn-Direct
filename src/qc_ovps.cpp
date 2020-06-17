#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

#include "cblas.h"
#include "blas_calls.h"
#include "qc_ovps.h"

void OVPs::init(const int dimm, const int mc_pair_num_) {
  mc_pair_num = mc_pair_num_;

  o_set.resize(dimm);
  v_set.resize(dimm);
  for (auto stop = 0; stop < dimm; stop++) {
    o_set[stop].resize(stop + 1);
    v_set[stop].resize(stop + 1);
    for (auto start = 0; start < stop + 1; start++) {
      o_set[stop][start].resize(mc_pair_num);
      v_set[stop][start].resize(mc_pair_num);
    }
  }
}
void OVPs::update_ovps(Wavefunction& electron_pair_psi1, Wavefunction& electron_pair_psi2, Tau* tau) {
  // update green's function trace objects

#ifdef HAVE_CUDA
  OVPS_SET_DEVICE temp(mc_pair_num);
#endif // HAVE_CUDA

  auto iocc1 = electron_pair_psi1.iocc1;
  auto iocc2 = electron_pair_psi1.iocc2;
  auto ivir1 = electron_pair_psi1.ivir1;
  auto ivir2 = electron_pair_psi1.ivir2;
  for (auto stop = 0; stop < o_set.size(); stop++) {
    for (auto start = 0; start < o_set[stop].size(); start++) {
      auto t_val = tau->get_exp_tau(stop, start);
      std::transform(t_val.begin(), t_val.end(), t_val.begin(), [](double x){return sqrt(x);});

      Ddgmm(DDGMM_SIDE_LEFT, ivir2 - iocc1, mc_pair_num, electron_pair_psi1.occ(), electron_pair_psi1.lda, &t_val[iocc1], 1, electron_pair_psi1.occTau(), electron_pair_psi1.lda);
      Ddgmm(DDGMM_SIDE_LEFT, ivir2 - iocc1, mc_pair_num, electron_pair_psi2.occ(), electron_pair_psi2.lda, &t_val[iocc1], 1, electron_pair_psi2.occTau(), electron_pair_psi2.lda);

#ifdef HAVE_CUDA
      temp.update(electron_pair_psi1.occTau(), electron_pair_psi2.occTau(), electron_pair_psi2.iocc2 - electron_pair_psi2.iocc1, electron_pair_psi2.lda);
      copy_OVPS_DEVICE_TO_HOST(temp, o_set[stop][start]);

      temp.update(electron_pair_psi1.virTau(), electron_pair_psi2.virTau(), electron_pair_psi2.ivir2 - electron_pair_psi2.ivir1, electron_pair_psi2.lda);
      copy_OVPS_DEVICE_TO_HOST(temp, v_set[stop][start]);
#else
      o_set[stop][start].update(electron_pair_psi1.occTau(), electron_pair_psi2.occTau(), electron_pair_psi2.iocc2 - electron_pair_psi2.iocc1, electron_pair_psi2.lda);
      v_set[stop][start].update(electron_pair_psi1.virTau(), electron_pair_psi2.virTau(), electron_pair_psi2.ivir2 - electron_pair_psi2.ivir1, electron_pair_psi2.lda);
#endif // HAVE_CUDA
    }
  }
}

