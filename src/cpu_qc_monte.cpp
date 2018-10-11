#include "qc_monte.h"

void QC_monte::update_wavefunction() {
  basis.host_psi_get(el_pair_list);
}
