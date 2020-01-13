#include "qc_monte.h"

void QC_monte::update_wavefunction() {
  basis.host_psi_get(electron_pair_psi1, electron_pair_list->pos1);
  basis.host_psi_get(electron_pair_psi2, electron_pair_list->pos2);
  if (electron_list != nullptr) {
    basis.host_psi_get(electron_psi, electron_list->pos);
  }
}
