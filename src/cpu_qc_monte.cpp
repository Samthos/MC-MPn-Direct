#include "qc_monte.h"

template <class Container>
void QC_monte<Container>::update_wavefunction() {
  for (auto &it : wavefunction_groups) {
    if (it.second.size() == 1) {
      basis.build_contractions(*wavefunctions[it.second.front()].pos);
    } else {
      basis.build_contractions_with_derivatives(*wavefunctions[it.second.front()].pos);
    }
    for (auto &jt : it.second) {
      Wavefunction_Type& wavefunction = wavefunctions[jt];
      auto code = jt & WT::mask;
      switch (code) {
        case WT::normal: basis.host_psi_get(wavefunction, *wavefunction.pos); break;
        case WT::dx: basis.host_psi_get_dx(wavefunction, *wavefunction.pos); break;
        case WT::dy: basis.host_psi_get_dy(wavefunction, *wavefunction.pos); break;
        case WT::dz: basis.host_psi_get_dz(wavefunction, *wavefunction.pos); break;
      }
    }
  }
}
