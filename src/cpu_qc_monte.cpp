#include "qc_monte.h"

void QC_monte::update_wavefunction() {
  basis.host_psi_get(electron_pair_psi1, electron_pair_psi2, electron_pair_list);
}
