#include "qc_monte.h"

void QC_monte::update_wavefunction() {
  for (auto &it : wavefunctions) {
    auto code = it.first & WT::mask;
    switch (code) {
      case WT::normal: basis.host_psi_get(it.second, *it.second.pos); break;
      case WT::dx: basis.host_psi_get_dx(it.second, *it.second.pos); break;
      case WT::dy: basis.host_psi_get_dy(it.second, *it.second.pos); break;
      case WT::dz: basis.host_psi_get_dz(it.second, *it.second.pos); break;
    }
  }
}
