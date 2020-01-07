#include "qc_monte.h"

void QC_monte::update_wavefunction() {
  basis.host_psi_get(basis.h_basis.wfn_psi1, basis.h_basis.wfn_psi2, el_pair_list);
}
