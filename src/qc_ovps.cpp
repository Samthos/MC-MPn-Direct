#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

#include "cblas.h"
#include "blas_calls.h"
#include "qc_ovps.h"

template <template <class, class> class Container, template <class> class Allocator>
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

template <template <class, class> class Container, template <class> class Allocator>
void OVPS<Container, Allocator>::update(Wavefunction& electron_pair_psi1, Wavefunction& electron_pair_psi2, Tau* tau) {
  std::cerr << "Default OVPS update_ovsp not implemented\n";
  exit(0);
}

template <>
void OVPS<std::vector, std::allocator>::update(Wavefunction& electron_pair_psi1, Wavefunction& electron_pair_psi2, Tau* tau) {
  auto iocc1 = electron_pair_psi1.iocc1;
  auto iocc2 = electron_pair_psi1.iocc2;
  auto ivir1 = electron_pair_psi1.ivir1;
  auto ivir2 = electron_pair_psi1.ivir2;
  for (auto stop = 0; stop < o_set.size(); stop++) {
    for (auto start = 0; start < o_set[stop].size(); start++) {
      auto t_val = tau->get_exp_tau(stop, start);
      std::transform(t_val.begin(), t_val.end(), t_val.begin(), [](double x){return sqrt(x);});

      Ddgmm(DDGMM_SIDE_LEFT, ivir2 - iocc1, electron_pairs, electron_pair_psi1.occ(), electron_pair_psi1.lda, &t_val[iocc1], 1, electron_pair_psi1.occTau(), electron_pair_psi1.lda);
      Ddgmm(DDGMM_SIDE_LEFT, ivir2 - iocc1, electron_pairs, electron_pair_psi2.occ(), electron_pair_psi2.lda, &t_val[iocc1], 1, electron_pair_psi2.occTau(), electron_pair_psi2.lda);

      o_set[stop][start].update(electron_pair_psi1.psiTau, iocc1, electron_pair_psi2.psiTau, iocc1, iocc2 - iocc1, electron_pair_psi2.lda);
      v_set[stop][start].update(electron_pair_psi1.psiTau, ivir1, electron_pair_psi2.psiTau, ivir1, ivir2 - ivir1, electron_pair_psi2.lda);
    }
  }
}

