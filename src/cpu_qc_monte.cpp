#include "qc_monte.h"

void QC_monte::update_wavefunction() {
  for (auto &it : el_pair_list) {
    basis.host_psi_get(it.pos1.data(), it.psi1.data());
    basis.host_psi_get(it.pos2.data(), it.psi2.data());
  }
}
